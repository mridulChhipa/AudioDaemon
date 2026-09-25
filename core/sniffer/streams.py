"""Place media pieces on disk, judge completeness, assemble whole files.

Everything the sniffer receives lands here, whether or not anyone wants it
yet: the listener decides what a playback *is* only after it has started, and
by then the opening segments have long been fetched. Pieces are grouped as

    item    -- the page they belong to (`detect.page_id`), e.g. yt:dQw4w9WgXcQ
    stream  -- one rendition of one track within it (a DASH representation,
               an HLS variant, a YouTube itag, a progressive file)

and a stream is placed one of two ways:

    bytes   -- by byte offset into a file (Range requests, progressive media)
    seq     -- by segment number, with timing (HLS, DASH, UMP)

Pieces arrive from what the player fetched. When a play ends with a stream
still incomplete, `missing()` lists what to fetch to finish its best
rendition, and the sniffer has the page fetch exactly that.
"""
import hashlib
import json
import logging
import shutil
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from core.config import (
    COVERAGE_TOLERANCE_SECONDS,
    FETCH_CHUNK_BYTES,
    PARTIAL_MAX_AGE_DAYS,
    STREAM_PARTIALS_DIR,
)
from core.sniffer import detect, ump
from core.sniffer.detect import MUXED, VIDEO

log = logging.getLogger(__name__)

BYTES, SEQ = "bytes", "seq"
# What ingest() did with a response.
STORED, MANIFEST, UNPLACED, IGNORED = "stored", "manifest", "unplaced", "ignored"
UNPLACED_MAX_BYTES = 256 * 1024 * 1024
UNPLACED_MAX_AGE_SECONDS = 300
_ITEM_MANIFEST = "item.json"
_STREAM_MANIFEST = "stream.json"
# Unclaimed items are media nobody asked for: autoplay previews, muted
# background tabs, ads. They're kept only long enough for a claim to arrive.
UNCLAIMED_MAX_AGE_SECONDS = 3600


def digest(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:20]


def merge_intervals(
    intervals: list[tuple[float, float]], tolerance: float = 0.0
) -> list[tuple[float, float]]:
    """Union of intervals, merging any that touch within `tolerance`."""
    if not intervals:
        return []
    ordered = sorted(intervals)
    merged = [list(ordered[0])]
    for start, end in ordered[1:]:
        if start <= merged[-1][1] + tolerance:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(s, e) for s, e in merged]


def coverage_gaps(
    intervals: list[tuple[float, float]],
    duration: float,
    tolerance: float = COVERAGE_TOLERANCE_SECONDS,
) -> list[tuple[float, float]]:
    """Parts of [0, duration] that no interval covers."""
    if duration <= 0:
        return []
    gaps: list[tuple[float, float]] = []
    cursor = 0.0
    for start, end in merge_intervals(intervals):
        if start - cursor > tolerance:
            gaps.append((cursor, start))
        cursor = max(cursor, end)
    if duration - cursor > tolerance:
        gaps.append((cursor, duration))
    return gaps


def duration_matches(actual: float, expected: float, *, loose: bool = False) -> bool:
    """Whether a stream is the item and not an ad or a UI sound on the same page.

    `loose` is for a stream that is the page's only candidate, where a site
    padding or trimming its stream is likelier than an ad of nearly the same
    length.
    """
    if expected <= 0 or actual <= 0:
        return True  # nothing to compare against
    slack = max(10.0, 0.10 * expected) if loose else max(3.0, 0.03 * expected)
    return abs(actual - expected) <= slack


@dataclass
class Response:
    """One finished HTTP response, as the sniffer hands it over."""

    page_url: str
    url: str
    status: int
    mime: str
    headers: dict[str, str]   # lower-cased names
    body: bytes
    received: float = field(default_factory=time.time)


@dataclass
class ManifestRef:
    """Where a URL sits, according to a manifest the page fetched earlier."""

    item: str
    stream_key: str
    kind: str | None
    encrypted: bool
    drm: tuple[str, ...] = ()
    mode: str = SEQ
    sequence: int = 0
    start: float = 0.0
    duration: float = 0.0
    is_init: bool = False
    expected_first: int | None = None
    expected_count: int | None = None
    total_duration: float = 0.0


@dataclass
class Chunk:
    file: str
    size: int
    offset: int = 0           # BYTES mode
    seq: int = 0              # SEQ mode
    start: float = 0.0        # seconds, when known
    duration: float = 0.0


@dataclass
class StreamState:
    key: str
    mode: str
    kind: str | None = None
    kind_certain: bool = False               # kind came from an init segment
    encrypted: bool = False
    total: int | None = None                 # BYTES: file size
    expected_first: int | None = None        # SEQ: manifest's first segment
    expected_count: int | None = None        # SEQ: ...and how many there are
    duration: float = 0.0                    # media duration, when known
    url: str = ""                            # BYTES: a URL the file was fetched from
    init: str | None = None                  # init segment filename
    needs_init: bool = False
    chunks: list[Chunk] = field(default_factory=list)
    updated: float = field(default_factory=time.time)

    # -- coverage ----------------------------------------------------------

    def byte_intervals(self) -> list[tuple[float, float]]:
        return [(c.offset, c.offset + c.size) for c in self.chunks]

    def time_intervals(self) -> list[tuple[float, float]]:
        return [(c.start, c.start + c.duration) for c in self.chunks if c.duration > 0]

    def seqs(self) -> set[int]:
        return {c.seq for c in self.chunks}

    def expected_seqs(self) -> set[int]:
        """SEQ: the segments a finished manifest lists (empty if it's unknown)."""
        first = self.expected_first or 0
        return set(range(first, first + (self.expected_count or 0)))

    def is_complete(self, expected_duration: float = 0.0) -> bool:
        if self.encrypted or not self.chunks:
            return False
        if self.mode == BYTES:
            if not self.total:
                return False
            merged = merge_intervals(self.byte_intervals())
            return len(merged) == 1 and merged[0][0] <= 0 and merged[0][1] >= self.total
        if self.needs_init and self.init is None:
            return False
        if self.expected_count:
            return self.expected_seqs() <= self.seqs()
        target = self.duration or expected_duration
        if target <= 0 or not self.time_intervals():
            return False
        return not coverage_gaps(self.time_intervals(), target, COVERAGE_TOLERANCE_SECONDS)

    def describe_coverage(self, expected_duration: float = 0.0) -> str:
        """Human-readable progress, for the log."""
        if self.mode == BYTES:
            have = sum(e - s for s, e in merge_intervals(self.byte_intervals()))
            total = f"{self.total / 1e6:.1f}" if self.total else "?"
            return f"{have / 1e6:.1f}/{total} MB"
        if self.expected_count:
            have = len(self.expected_seqs() & self.seqs())
            return f"{have}/{self.expected_count} segments"
        target = self.duration or expected_duration
        covered = sum(e - s for s, e in merge_intervals(self.time_intervals()))
        gaps = coverage_gaps(self.time_intervals(), target) if target > 0 else []
        missing = ", ".join(f"{s:.0f}-{e:.0f}s" for s, e in gaps[:4])
        return f"{covered:.0f}/{target:.0f}s" + (f" (missing {missing})" if missing else "")

    @property
    def bytes_stored(self) -> int:
        return sum(c.size for c in self.chunks)

    # -- persistence -------------------------------------------------------

    def to_json(self) -> dict:
        data = asdict(self)
        data["chunks"] = [asdict(c) for c in self.chunks]
        return data

    @classmethod
    def from_json(cls, data: dict) -> "StreamState":
        chunks = [Chunk(**c) for c in data.pop("chunks", [])]
        # Manifests written by older versions may carry fields since dropped.
        known = {f.name for f in fields(cls)}
        state = cls(**{k: v for k, v in data.items() if k in known})
        state.chunks = chunks
        return state


@dataclass
class ItemSummary:
    has_video: bool
    drm: bool
    drm_systems: tuple[str, ...]   # which DRM, as far as known
    streams: int
    clear: int = 0             # streams that aren't encrypted


@dataclass
class Candidate:
    """A complete, clean stream the pipeline may turn into a library file."""

    item: str
    key: str
    kind: str | None
    bytes: int


@dataclass(frozen=True)
class FetchRequest:
    """One request that fills part of a stream, as the page should make it."""

    url: str
    range: tuple[int, int] | None = None   # inclusive bytes, sent as a Range header

    @property
    def header(self) -> str | None:
        return f"bytes={self.range[0]}-{self.range[1]}" if self.range else None


class StreamStore:
    """Thread-safe: the sniffer ingests from worker threads while the pipeline
    reads and assembles."""

    def __init__(self, root: Path | None = None):
        self.root = root or STREAM_PARTIALS_DIR
        self._lock = threading.RLock()
        self._refs: dict[str, ManifestRef] = {}        # segment URL -> placement
        self._variant_kind: dict[str, str | None] = {}  # variant playlist URL -> kind
        self._states: dict[tuple[str, str], StreamState] = {}
        # Where each listed segment of a manifest stream can be fetched:
        # (item, stream) -> {seq: (url, (offset, length) or None)}, plus init.
        self._seq_urls: dict[tuple[str, str], dict[int, tuple[str, tuple[int, int] | None]]] = {}
        self._init_urls: dict[tuple[str, str], str] = {}
        # URLs asked for by missing(), and the item each fills: the page may
        # have navigated since, so a response's own page says nothing.
        self._requested: dict[str, str] = {}
        self._unplaced: deque[Response] = deque()
        self._unplaced_bytes = 0

    # -- paths -------------------------------------------------------------

    def _item_dir(self, item: str) -> Path:
        return self.root / digest(item)

    def _stream_dir(self, item: str, key: str) -> Path:
        return self._item_dir(item) / digest(key)

    # -- ingest ------------------------------------------------------------

    def ingest(self, resp: Response) -> str:
        """Take one response. Returns what became of it (STORED, MANIFEST,
        UNPLACED, IGNORED). Never raises: a bad body costs only itself."""
        try:
            outcome = self._ingest(resp)
        except Exception:
            log.warning("Could not ingest %s", resp.url[:120], exc_info=True)
            return IGNORED
        if outcome == UNPLACED:
            self._hold(resp)
        elif outcome == MANIFEST:
            self._retry_unplaced()
        return outcome

    def _hold(self, resp: Response) -> None:
        """Keep a segment that arrived before its manifest, for a while.

        A page already playing when the daemon attached fetched its manifest
        unseen; the sniffer asks the page to fetch it again, and whatever was
        held meanwhile is placed when it lands.
        """
        with self._lock:
            now = time.time()
            self._unplaced.append(resp)
            self._unplaced_bytes += len(resp.body)
            while self._unplaced and (
                self._unplaced_bytes > UNPLACED_MAX_BYTES
                or now - self._unplaced[0].received > UNPLACED_MAX_AGE_SECONDS
            ):
                dropped = self._unplaced.popleft()
                self._unplaced_bytes -= len(dropped.body)

    def _retry_unplaced(self) -> None:
        with self._lock:
            held = list(self._unplaced)
            self._unplaced.clear()
            self._unplaced_bytes = 0
        for resp in held:
            try:
                outcome = self._ingest(resp)
            except Exception:
                continue
            if outcome == UNPLACED:
                with self._lock:
                    self._unplaced.append(resp)
                    self._unplaced_bytes += len(resp.body)

    def _ingest(self, resp: Response) -> str:
        if not 200 <= resp.status < 300 or not resp.body:
            return IGNORED
        container = detect.sniff_container(resp.body, resp.mime)
        item = detect.page_id(resp.page_url)

        if container == detect.HLS:
            self._register_hls(resp.body.decode("utf-8", "replace"), resp.url, item)
            return MANIFEST
        if container == detect.DASH:
            self._register_dash(resp.body.decode("utf-8", "replace"), resp.url, item)
            return MANIFEST
        if container == detect.UMP:
            self._ingest_ump(resp.body, item)
            return STORED

        cr = detect.content_range(resp.headers.get("content-range"))
        qr = detect.query_range(resp.url)
        start = cr[0] if cr else qr[0] if qr else 0
        if start > 0 and container not in (detect.HLS, detect.DASH):
            # Magic bytes mean nothing mid-file: a range that happens to start
            # with 0x47 is not MPEG-TS. Only the MIME type is a hint here.
            container = detect.UNKNOWN
            facts = detect.MediaFacts(detect.UNKNOWN, detect.kind_from_mime(resp.mime))
        else:
            facts = detect.media_facts(resp.body, resp.mime)
        ref = self._lookup(resp.url, cr[0] if cr else None)
        if container == detect.UNKNOWN and ref is None and not (
            detect.kind_from_mime(resp.mime)
            and (cr or detect.query_range(resp.url))
        ):
            # A range from the middle of a file has no magic bytes, so an
            # unknown body is kept only if the MIME says media and it is a range.
            return IGNORED

        if ref is not None:
            if ref.mode == BYTES:
                offset, total = (cr[0], cr[2]) if cr else (0, len(resp.body))
                self._add_bytes(ref.item, ref.stream_key, offset, total, resp.body, facts,
                                ref=ref, url=resp.url)
            else:
                self._add_seq(ref.item, ref.stream_key, ref.sequence, ref.start,
                              ref.duration, resp.body, facts, is_init=ref.is_init, ref=ref)
            return STORED

        # No manifest placed it: a file, or a byte range of one.
        item = self._requested.get(resp.url, item)
        key = detect.stream_url_key(resp.url)
        if cr:
            offset, total = cr[0], cr[2]
        elif qr:
            offset = qr[0]
            clen = detect.query_value(resp.url, "clen")
            total = int(clen) if clen and clen.isdigit() else None
        elif resp.status == 200 and self._looks_like_whole_file(resp.body, container, facts):
            offset, total = 0, len(resp.body)
        else:
            # A segment whose manifest we haven't seen (yet).
            log.debug("Unplaced media response: %s", resp.url[:120])
            return UNPLACED
        self._add_bytes(item, key, offset, total, resp.body, facts, url=resp.url)
        return STORED

    @staticmethod
    def _looks_like_whole_file(body: bytes, container: str, facts) -> bool:
        """A 200 with no range is a whole file only if it starts like one.

        A bare fMP4 segment (styp/moof) or an init segment (ftyp+moov, no
        media) would otherwise masquerade as a complete one-piece file.
        """
        if container == detect.MP4:
            return body[4:8] == b"ftyp" and facts.has_media
        if container == detect.WEBM:
            return body.startswith(b"\x1a\x45\xdf\xa3") and facts.has_media
        return container in (detect.MP3, detect.ADTS)

    def _lookup(self, url: str, offset: int | None) -> ManifestRef | None:
        with self._lock:
            if offset is not None:
                ref = self._refs.get(f"{url}#{offset}")
                if ref:
                    return ref
            ref = self._refs.get(url)
            if ref:
                return ref
            # Tokens appended after the manifest was written.
            return self._refs.get(url.split("?", 1)[0])

    def _remember(self, url: str, ref: ManifestRef, offset: int | None = None) -> None:
        with self._lock:
            if offset is not None:
                self._refs[f"{url}#{offset}"] = ref
            else:
                self._refs[url] = ref
                bare = url.split("?", 1)[0]
                self._refs.setdefault(bare, ref)
            if len(self._refs) > 200_000:  # a long session; forget the oldest half
                for k in list(self._refs)[: len(self._refs) // 2]:
                    del self._refs[k]

    # -- manifests ---------------------------------------------------------

    def _register_hls(self, text: str, url: str, item: str) -> None:
        playlist = detect.parse_hls(text, url)
        if playlist.is_master:
            with self._lock:
                for v in playlist.variants:
                    self._variant_kind[v.uri] = detect.kind_from_codecs(v.codecs)
            return
        if not playlist.segments:
            return
        if playlist.encrypted:
            self.mark_drm(item, f"HLS encryption in {url[:80]}", playlist.drm)
        key = "hls:" + detect.stream_url_key(url)
        ended = playlist.ended
        common = dict(
            item=item, stream_key=key, kind=self._variant_kind.get(url),
            encrypted=playlist.encrypted, drm=tuple(playlist.drm),
            expected_first=playlist.segments[0].sequence if ended else None,
            expected_count=len(playlist.segments) if ended else None,
            total_duration=playlist.duration if ended else 0.0,
        )
        if playlist.init_uri:
            self._remember(playlist.init_uri, ManifestRef(is_init=True, **common))
        urls = {}
        for seg in playlist.segments:
            ref = ManifestRef(sequence=seg.sequence, start=seg.start, duration=seg.duration,
                              **common)
            self._remember(seg.uri, ref, seg.byterange[0] if seg.byterange else None)
            urls[seg.sequence] = (seg.uri, seg.byterange)
        self._list_urls(item, key, urls, playlist.init_uri)

    def _register_dash(self, text: str, url: str, item: str) -> None:
        try:
            manifest = detect.parse_dash(text, url)
        except Exception:
            log.debug("Unparseable MPD at %s", url[:120], exc_info=True)
            return
        for rep in manifest.representations:
            if rep.encrypted:
                self.mark_drm(item, f"DASH ContentProtection in {url[:80]}", rep.drm)
            key = f"dash:{detect.stream_url_key(url)}:{rep.id}"
            seqs = [s for s, _, _ in rep.segments.values()]
            common = dict(
                item=item, stream_key=key, kind=rep.kind, encrypted=rep.encrypted,
                drm=tuple(rep.drm),
                expected_first=min(seqs) if seqs else None,
                expected_count=len(seqs) if seqs else None,
                total_duration=manifest.duration,
            )
            if rep.base_url:
                self._remember(rep.base_url, ManifestRef(mode=BYTES, **common))
                continue
            if rep.init_url:
                self._remember(rep.init_url, ManifestRef(is_init=True, **common))
            for seg_url, (seq, start, duration) in rep.segments.items():
                self._remember(seg_url, ManifestRef(sequence=seq, start=start,
                                                    duration=duration, **common))
            self._list_urls(item, key, {seq: (seg_url, None) for seg_url, (seq, _, _)
                                        in rep.segments.items()}, rep.init_url)

    def _list_urls(self, item: str, key: str,
                   urls: dict[int, tuple[str, tuple[int, int] | None]],
                   init_url: str | None) -> None:
        with self._lock:
            # A live playlist lists a sliding window; keep every URL it named.
            self._seq_urls.setdefault((item, key), {}).update(urls)
            if init_url:
                self._init_urls[(item, key)] = init_url

    def _ingest_ump(self, body: bytes, page_item: str) -> None:
        for chunk in ump.extract_chunks(body):
            item = f"yt:{chunk.video_id}" if chunk.video_id else page_item
            if item != page_item and not page_item.startswith("yt:"):
                # A YouTube player embedded in another site's page. (On
                # YouTube itself another video id is an ad, and stays apart.)
                self._link(page_item, item)
            key = f"yt:{chunk.video_id}:{chunk.itag}:{chunk.xtags}"
            facts = detect.media_facts(chunk.data)
            self._add_seq(item, key, chunk.sequence, chunk.start, chunk.duration,
                          chunk.data, facts, is_init=chunk.is_init)

    # -- storing pieces ----------------------------------------------------

    def _state(self, item: str, key: str, mode: str) -> StreamState:
        cache_key = (item, key)
        state = self._states.get(cache_key)
        if state is None:
            path = self._stream_dir(item, key) / _STREAM_MANIFEST
            if path.exists():
                try:
                    state = StreamState.from_json(json.loads(path.read_text("utf-8")))
                except (ValueError, TypeError):
                    log.warning("Unreadable stream manifest %s; starting over", path)
            if state is None:
                state = StreamState(key=key, mode=mode)
            self._states[cache_key] = state
        return state

    def _save(self, item: str, state: StreamState) -> None:
        sdir = self._stream_dir(item, state.key)
        sdir.mkdir(parents=True, exist_ok=True)
        state.updated = time.time()
        tmp = sdir / (_STREAM_MANIFEST + ".tmp")
        tmp.write_text(json.dumps(state.to_json()), "utf-8")
        tmp.replace(sdir / _STREAM_MANIFEST)
        self._touch_item(item)

    def _touch_item(self, item: str, **changes) -> dict:
        idir = self._item_dir(item)
        idir.mkdir(parents=True, exist_ok=True)
        path = idir / _ITEM_MANIFEST
        data = {"item": item, "drm": False, "claimed": False, "created": time.time()}
        if path.exists():
            try:
                data.update(json.loads(path.read_text("utf-8")))
            except ValueError:
                pass
        data.update(changes)
        data["updated"] = time.time()
        tmp = idir / (_ITEM_MANIFEST + ".tmp")
        tmp.write_text(json.dumps(data), "utf-8")
        tmp.replace(path)
        return data

    @staticmethod
    def _write(path: Path, data: bytes) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_bytes(data)
        tmp.replace(path)  # a half-written chunk must never look finished

    def _absorb_facts(self, item: str, state: StreamState, facts, ref: ManifestRef | None):
        # What a stream carries, most trustworthy first: its own init segment,
        # then the manifest's CODECS, then the response's MIME type.
        if facts.is_init and facts.kind:
            state.kind, state.kind_certain = facts.kind, True
        elif not state.kind_certain:
            state.kind = (ref.kind if ref else None) or state.kind or facts.kind
        if facts.duration and not state.duration:
            state.duration = facts.duration
        if ref is not None:
            if ref.expected_count and not state.expected_count:
                state.expected_first, state.expected_count = ref.expected_first, ref.expected_count
            if ref.total_duration and not state.duration:
                state.duration = ref.total_duration
        if facts.encrypted or (ref and ref.encrypted):
            # Every encrypted piece may name more of the DRM (a pssh in a
            # later segment), so each is reported; only new names are logged.
            self.mark_drm(item, f"encrypted media in stream {state.key[:60]}",
                          facts.drm + (ref.drm if ref else ()))
        if (facts.encrypted or (ref and ref.encrypted)) and not state.encrypted:
            state.encrypted = True
            # Encrypted bytes are of no use to anyone here; don't keep them.
            sdir = self._stream_dir(item, state.key)
            for chunk in state.chunks:
                (sdir / chunk.file).unlink(missing_ok=True)
            if state.init:
                (sdir / state.init).unlink(missing_ok=True)
            state.chunks, state.init = [], None

    def _add_bytes(self, item, key, offset, total, body, facts, ref=None, url="") -> None:
        with self._lock:
            state = self._state(item, key, BYTES)
            self._absorb_facts(item, state, facts, ref)
            if total and not state.total:
                state.total = total
            if url:
                state.url = url
            if state.encrypted:
                self._save(item, state)
                return
            if any(c.offset == offset and c.size >= len(body) for c in state.chunks):
                return  # already have it (replays re-fetch the same ranges)
            name = f"b_{offset:012d}_{len(body)}.bin"
            sdir = self._stream_dir(item, key)
            sdir.mkdir(parents=True, exist_ok=True)
            self._write(sdir / name, body)
            state.chunks.append(Chunk(file=name, size=len(body), offset=offset))
            self._save(item, state)

    def _add_seq(self, item, key, seq, start, duration, body, facts, *, is_init,
                 ref=None) -> None:
        with self._lock:
            state = self._state(item, key, SEQ)
            self._absorb_facts(item, state, facts, ref)
            if state.encrypted:
                self._save(item, state)
                return
            sdir = self._stream_dir(item, key)
            sdir.mkdir(parents=True, exist_ok=True)
            if is_init or (facts.is_init and not facts.has_media):
                if state.init is None:
                    state.init = "init.bin"
                    self._write(sdir / state.init, body)
                    self._save(item, state)
                return
            if facts.container in (detect.MP4, detect.WEBM) and not facts.is_init:
                state.needs_init = True
            if seq in state.seqs():
                return
            name = f"s_{seq:08d}.bin"
            self._write(sdir / name, body)
            state.chunks.append(Chunk(file=name, size=len(body), seq=seq, start=start,
                                      duration=duration))
            self._save(item, state)

    # -- what the listener and pipeline ask --------------------------------

    def _meta(self, item: str) -> dict:
        """An item's item.json, or {} if it has none yet."""
        try:
            return json.loads((self._item_dir(item) / _ITEM_MANIFEST).read_text("utf-8"))
        except (OSError, ValueError):
            return {}

    def mark_drm(self, item: str, reason: str = "", systems=()) -> None:
        """Flag an item as DRM-protected, noting which DRM when it's known.

        Logged the first time, and again whenever a DRM system not yet seen
        for the item turns up. A more specific label replaces a bare one:
        "Widevine L3 (...)" from the page supersedes "Widevine" from a pssh,
        and a bare name adds nothing once the specific one is known.
        """
        with self._lock:
            meta = self._meta(item)
            known = list(meta.get("drm_systems") or [])
            new = [s for s in dict.fromkeys(systems) if s and s not in known
                   and not any(k.startswith(s + " ") for k in known)]
            if meta.get("drm") and not new:
                return
            known = [k for k in known if not any(n.startswith(k + " ") for n in new)] + new
            self._touch_item(item, drm=True, drm_systems=known)
            log.info("DRM-protected media on %s: %s (%s); it will not be archived",
                     item, ", ".join(known) or "DRM system not named yet", reason)

    def mark_playback(self, item: str, secure: str, how: str) -> None:
        """Record how securely the item's live player decrypts, as it reports.

        This is the playback's own security level, which neither the stream
        nor a licence says; logged whenever it changes.
        """
        with self._lock:
            playback = {"secure": secure, "how": how}
            if self._meta(item).get("drm_playback") == playback:
                return
            self._touch_item(item, drm=True, drm_playback=playback)
            log.info("DRM-protected media on %s: %s (the playback); it will not be archived",
                     item, ", ".join(self.drm_systems(item)))

    def is_drm(self, item: str) -> bool:
        return bool(self._meta(item).get("drm"))

    def drm_systems(self, item: str) -> list[str]:
        """Which DRM, with the playback's level where it is known.

        A single system named without a level ("Widevine", from a pssh) gets
        the level of the playback ("Widevine L3 (playback: ...)"). With
        several candidates, or none, the level is given for each system that
        has a name for it, as one entry.
        """
        meta = self._meta(item)
        labels = list(meta.get("drm_systems") or [])
        playback = meta.get("drm_playback")
        if not playback:
            return labels
        secure, how = playback["secure"], playback["how"]
        if any(label.split(" ")[0] in detect.DRM_SYSTEM_NAMES and " " in label
               and not label.startswith("unknown") for label in labels):
            return labels  # the page's own MediaKeys already said the level
        bare = [label for label in labels if detect.level_name(label, secure)]
        if len(bare) == 1:
            level = detect.level_name(bare[0], secure)
            return [f"{label} {level} (playback: {how})" if label == bare[0] else label
                    for label in labels]
        levels = " / ".join(f"{s} {detect.level_name(s, secure)}" for s in bare)
        return labels + [f"playback {secure}-secure{': ' + levels if levels else ''} ({how})"]

    def _link(self, page_item: str, embedded: str) -> None:
        """Record that a page's player is filed under another item."""
        with self._lock:
            embeds = self._embeds(page_item)
            if embedded not in embeds:
                self._touch_item(page_item, embeds=embeds + [embedded])

    def _embeds(self, item: str) -> list[str]:
        return list(self._meta(item).get("embeds") or [])

    def members(self, item: str) -> list[str]:
        """The item and every item its page embeds."""
        return [item] + [e for e in self._embeds(item) if e != item]

    def claim(self, item: str, title: str = "") -> None:
        """Mark an item as wanted, so the unclaimed-media purge leaves it be."""
        with self._lock:
            for member in self.members(item):
                self._touch_item(member, claimed=True, title=title)

    def streams(self, item: str) -> list[StreamState]:
        idir = self._item_dir(item)
        if not idir.exists():
            return []
        out = []
        with self._lock:
            for sdir in idir.iterdir():
                manifest = sdir / _STREAM_MANIFEST
                if not manifest.exists():
                    continue
                try:
                    data = json.loads(manifest.read_text("utf-8"))
                except ValueError:
                    continue
                state = self._states.get((item, data.get("key")))
                out.append(state if state is not None else StreamState.from_json(data))
        return out

    def _all_streams(self, item: str) -> list[tuple[str, StreamState]]:
        return [(member, s) for member in self.members(item) for s in self.streams(member)]

    def summary(self, item: str) -> ItemSummary:
        streams = [s for _, s in self._all_streams(item)]
        kinds = {s.kind for s in streams}
        return ItemSummary(
            has_video=bool(kinds & {VIDEO, MUXED}),
            drm=(any(self.is_drm(m) for m in self.members(item))
                 or any(s.encrypted for s in streams)),
            drm_systems=tuple(dict.fromkeys(
                s for m in self.members(item) for s in self.drm_systems(m))),
            streams=len(streams),
            clear=sum(not s.encrypted for s in streams),
        )

    def candidates(self, item: str, expected_duration: float = 0.0) -> list[Candidate]:
        """Complete, unencrypted streams, largest first."""
        found = [
            Candidate(member, s.key, s.kind, s.bytes_stored)
            for member, s in self._all_streams(item) if s.is_complete(expected_duration)
        ]
        return sorted(found, key=lambda c: c.bytes, reverse=True)

    def progress(self, item: str, expected_duration: float = 0.0) -> list[str]:
        """One line per stream, for the log when nothing is complete."""
        return [
            f"{s.kind or '?'} {s.key[:40]}: " + (
                "encrypted" if s.encrypted else s.describe_coverage(expected_duration))
            for _, s in self._all_streams(item)
        ]

    # -- completion ----------------------------------------------------------

    def missing(self, item: str, expected_duration: float = 0.0) -> list[FetchRequest]:
        """What to fetch to complete the best rendition of each kind.

        The player may have stopped early, never fetched the init segment, or
        switched quality mid-play so that no rendition is whole. Of the
        renditions that can be finished from here, the highest-bitrate one of
        each kind (audio, video, muxed) is chosen and its gaps listed.
        YouTube's UMP streams can't be: SABR requests can't be replayed.
        """
        best: dict[str | None, tuple[float, str, list[FetchRequest]]] = {}
        for member, state in self._all_streams(item):
            if state.encrypted or not duration_matches(state.duration, expected_duration):
                continue
            plan = self._plan(member, state, expected_duration)
            if plan is None:
                continue
            score = self._quality(state)
            if state.kind not in best or score > best[state.kind][0]:
                best[state.kind] = (score, member, plan)

        requests: list[FetchRequest] = []
        with self._lock:
            if len(self._requested) > 50_000:
                self._requested.clear()
            for _, member, plan in best.values():
                for request in plan:
                    self._requested[request.url] = member
                requests += plan
        return requests

    def _plan(self, item: str, state: StreamState,
              expected_duration: float) -> list[FetchRequest] | None:
        """The requests that complete one stream: [] if it's whole, None if it
        can't be completed from what's known about it."""
        if state.is_complete(expected_duration):
            return []
        if state.mode == BYTES:
            return _plan_bytes(state) or None
        with self._lock:
            urls = dict(self._seq_urls.get((item, state.key), {}))
            init_url = self._init_urls.get((item, state.key))
        if not urls:
            return None  # UMP, or its manifest was seen before a restart
        plan: list[FetchRequest] = []
        if state.needs_init and state.init is None:
            if not init_url:
                return None
            plan.append(FetchRequest(init_url))
        if not state.expected_seqs() <= set(urls):
            return None
        for seq in sorted(set(urls) - state.seqs()):
            url, byterange = urls[seq]
            plan.append(FetchRequest(url, (byterange[0], byterange[0] + byterange[1] - 1)
                                     if byterange else None))
        return plan or None

    @staticmethod
    def _quality(state: StreamState) -> float:
        """Roughly a stream's bitrate, for choosing among renditions."""
        if state.mode == BYTES:
            return float(state.total or state.bytes_stored)
        seconds = sum(c.duration for c in state.chunks)
        if seconds > 0:
            return state.bytes_stored / seconds
        return state.bytes_stored / max(1, len(state.chunks))

    def assemble(self, item: str, key: str, dest: Path) -> bool:
        """Write a complete stream out as one file. Returns whether it worked."""
        with self._lock:
            state = next((s for s in self.streams(item) if s.key == key), None)
            if state is None or state.encrypted:
                return False
            chunks = list(state.chunks)
            init = state.init
            mode, total = state.mode, state.total
        sdir = self._stream_dir(item, key)
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(dest, "wb") as out:
                if mode == BYTES:
                    cursor = 0
                    for chunk in sorted(chunks, key=lambda c: c.offset):
                        if chunk.offset > cursor:
                            log.error("Hole at byte %d in %s", cursor, key[:60])
                            return False
                        skip = cursor - chunk.offset
                        if skip >= chunk.size:
                            continue
                        data = (sdir / chunk.file).read_bytes()[skip:]
                        if total:
                            data = data[: max(0, total - cursor)]
                        out.write(data)
                        cursor += len(data)
                else:
                    if init:
                        out.write((sdir / init).read_bytes())
                    seen = set()
                    for chunk in sorted(chunks, key=lambda c: c.seq):
                        if chunk.seq in seen:
                            continue
                        seen.add(chunk.seq)
                        out.write((sdir / chunk.file).read_bytes())
        except OSError:
            log.exception("Could not assemble %s", key[:60])
            dest.unlink(missing_ok=True)
            return False
        return True

    def _drop(self, item: str) -> None:
        """Forget an item: its cached state and everything on disk."""
        for cache_key in [k for k in self._states if k[0] == item]:
            del self._states[cache_key]
        shutil.rmtree(self._item_dir(item), ignore_errors=True)

    def discard(self, item: str) -> None:
        with self._lock:
            for member in self.members(item):
                self._drop(member)

    def purge(self, unclaimed_age: float = UNCLAIMED_MAX_AGE_SECONDS,
              claimed_age_days: float = PARTIAL_MAX_AGE_DAYS) -> int:
        """Delete stale items. Returns how many."""
        if not self.root.exists():
            return 0
        now = time.time()
        removed = 0
        with self._lock:
            for idir in self.root.iterdir():
                if not idir.is_dir():
                    continue
                try:
                    data = json.loads((idir / _ITEM_MANIFEST).read_text("utf-8"))
                except (OSError, ValueError):
                    data = {"claimed": False, "updated": idir.stat().st_mtime}
                age = now - float(data.get("updated", 0))
                limit = claimed_age_days * 86400 if data.get("claimed") else unclaimed_age
                if age > limit:
                    if data.get("item"):
                        self._drop(data["item"])
                    else:
                        shutil.rmtree(idir, ignore_errors=True)
                    removed += 1
        return removed


def _plan_bytes(state: StreamState) -> list[FetchRequest]:
    """Ranged requests for the holes in a file, in pieces the browser keeps."""
    if not state.url:
        return []
    merged = merge_intervals(state.byte_intervals())
    if state.total:
        gaps, cursor = [], 0
        for start, end in merged:
            if start > cursor:
                gaps.append((cursor, int(start)))
            cursor = max(cursor, int(end))
        if cursor < state.total:
            gaps.append((cursor, state.total))
    else:
        # Size unknown: one request past what's held. Its Content-Range says
        # the size, and the next round asks for the rest.
        cursor = int(merged[0][1]) if merged and merged[0][0] <= 0 else 0
        gaps = [(cursor, cursor + FETCH_CHUNK_BYTES)]
    return [_range_request(state.url, start, min(start + FETCH_CHUNK_BYTES, end) - 1)
            for gap_start, end in gaps
            for start in range(gap_start, end, FETCH_CHUNK_BYTES)]


def _range_request(url: str, first: int, last: int) -> FetchRequest:
    """Bytes first..last of a file, the way its URL asks for ranges.

    Some servers take the range as a `range=` query parameter rather than a
    header; the URL the player used says which.
    """
    if detect.query_range(url):
        parts = urlsplit(url)
        query = [(k, f"{first}-{last}" if k == "range" else v)
                 for k, v in parse_qsl(parts.query, keep_blank_values=True)]
        return FetchRequest(urlunsplit(parts._replace(query=urlencode(query))))
    return FetchRequest(url, (first, last))
