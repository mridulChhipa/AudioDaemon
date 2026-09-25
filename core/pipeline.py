"""The background worker that turns finished plays into library files.

Assembly, remuxing and tagging run here on a single FIFO worker, so a long mux
never delays noticing the next play.

For each play:

1. let responses still in flight land;
2. if no stream of the item is complete, have the page fetch what's missing
   (the sniffer's `complete()`);
3. assemble and probe every complete stream, drop the ones whose length
   isn't the play's (ads, UI sounds), and choose the best audio and video;
4. remux (`-c copy`) into the music library, the video library, or both.

A music video goes to both libraries. What can't be made -- DRM-protected, or
a stream that couldn't be completed -- is logged, and its pieces kept for a
later play.
"""
import asyncio
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

from core import database, encoder
from core.agent import TrackVerdict, route
from core.config import MIN_VIDEO_SECONDS, STAGING_DIR, STREAM_SETTLE_SECONDS
from core.database import MUSIC, VIDEO
from core.sniffer.cdp import Sniffer
from core.sniffer.streams import StreamStore, digest, duration_matches

log = logging.getLogger(__name__)


@dataclass
class MediaJob:
    """One finished play of a claimed stream item."""

    verdict: TrackVerdict
    album: str
    album_artist: str
    track_number: int
    duration: float
    item: str                   # StreamStore item (the page)
    site: str
    has_video: bool | None
    cover: tuple[bytes, str] | None
    source: str
    closed_at: float = field(default_factory=time.time)

    @property
    def artist(self) -> str:
        return self.verdict.clean_artist

    @property
    def title(self) -> str:
        return self.verdict.clean_title


async def already_owned(media_hash: str, kind: str = MUSIC) -> bool:
    """Whether the library really holds this item.

    A row pointing at a deleted file would block the item forever, so drop it
    and let the next play re-acquire.
    """
    recorded = await asyncio.to_thread(database.get_file_path, media_hash, kind=kind)
    if recorded is None:
        return False
    if Path(recorded).exists():
        return True

    log.info("Recorded file is missing (%s); forgetting it", recorded)
    await asyncio.to_thread(database.forget_media, media_hash, kind=kind)
    return False


@dataclass
class _Assembled:
    path: Path
    info: encoder.ProbeInfo
    bytes: int

    @property
    def area(self) -> int:
        return self.info.width * self.info.height


@dataclass
class Selection:
    """The best complete streams of an item, assembled to files."""

    audio: _Assembled | None = None       # its first audio track is the music
    video: _Assembled | None = None
    video_audio: _Assembled | None = None  # separate audio to mux with `video`
    files: list[Path] = field(default_factory=list)

    def cleanup(self) -> None:
        for path in self.files:
            path.unlink(missing_ok=True)


def choose(assembled: list[_Assembled]) -> Selection:
    """Best audio and best video among complete, probed streams.

    Audio: the largest audio-only stream (the highest bitrate), else a muxed
    one's audio. Video: the largest picture, video-only or muxed; a video-only
    winner is paired with the chosen audio.
    """
    audio_only = [a for a in assembled if a.info.has_audio and not a.info.has_video]
    video_only = [a for a in assembled if a.info.has_video and not a.info.has_audio]
    muxed = [a for a in assembled if a.info.has_video and a.info.has_audio]

    sel = Selection(files=[a.path for a in assembled])
    best_audio = max(audio_only, key=lambda a: a.bytes, default=None)
    best_muxed = max(muxed, key=lambda a: (a.area, a.bytes), default=None)
    sel.audio = best_audio or best_muxed

    best_video = max(video_only, key=lambda a: (a.area, a.bytes), default=None)
    if best_video and (best_muxed is None or best_video.area > best_muxed.area):
        sel.video = best_video
        sel.video_audio = sel.audio
    elif best_muxed:
        sel.video = best_muxed
    return sel


class Pipeline:
    """FIFO worker for finished plays."""

    def __init__(self, store: StreamStore, sniffer: Sniffer) -> None:
        self.store = store
        self.sniffer = sniffer      # has the page fetch what a stream lacks
        self.queue: asyncio.Queue[MediaJob | None] = asyncio.Queue()
        self._task: asyncio.Task | None = None

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run(), name="pipeline")

    async def stop(self) -> None:
        if self._task is None:
            return
        await self.queue.put(None)
        await self._task
        self._task = None

    def submit(self, job: MediaJob) -> None:
        self.queue.put_nowait(job)
        log.debug("Queued %s - %s (%d waiting)", job.artist, job.title, self.queue.qsize())

    async def _run(self) -> None:
        while True:
            job = await self.queue.get()
            if job is None:
                return
            try:
                await self.process_media(job)
            except Exception:
                log.exception("Processing failed for %s - %s", job.artist, job.title)

    # -- a whole play --------------------------------------------------------

    async def process_media(self, job: MediaJob) -> None:
        # Let responses still in flight when the play ended land first.
        wait = job.closed_at + STREAM_SETTLE_SECONDS - time.time()
        if wait > 0:
            await asyncio.sleep(wait)

        summary = self.store.summary(job.item)
        has_video = bool(job.has_video or summary.has_video)
        want_music, want_video = route(job.verdict, has_video)
        if want_video and not job.verdict.is_music and 0 < job.duration < MIN_VIDEO_SECONDS:
            log.info("Too short to keep as a video (%.0fs): %s", job.duration, job.title)
            want_video = False
        if not (want_music or want_video):
            return

        await self._complete(job)
        selection = await self._select(job)
        try:
            music_done = await self._music(job, selection) if want_music else True
            video_done = await self._video(job, selection) if want_video else True
        finally:
            selection.cleanup()

        if music_done and video_done:
            await asyncio.to_thread(self.store.discard, job.item)

    async def _complete(self, job: MediaJob) -> None:
        """Have the page fetch whatever the item's best renditions lack."""
        try:
            await self.sniffer.complete(job.item, job.duration)
        except Exception:
            log.warning("Fetching the rest of %s failed", job.item, exc_info=True)

    async def _select(self, job: MediaJob) -> Selection:
        """Assemble and probe every complete stream of the item."""
        candidates = await asyncio.to_thread(self.store.candidates, job.item, job.duration)
        probed: list[_Assembled] = []
        for cand in candidates:
            path = STAGING_DIR / f"asm_{digest(cand.item + cand.key)}.bin"
            if not await asyncio.to_thread(self.store.assemble, cand.item, cand.key, path):
                continue
            info = await asyncio.to_thread(encoder.probe, path)
            if info is None:
                path.unlink(missing_ok=True)
                continue
            probed.append(_Assembled(path, info, cand.bytes))

        # The page's only stream is let off with a looser length check.
        loose = len(probed) == 1
        assembled = []
        for a in probed:
            if duration_matches(a.info.duration, job.duration, loose=loose):
                assembled.append(a)
            else:
                log.debug("Skipping %s: %.1fs, expected %.1fs", a.path.name,
                          a.info.duration, job.duration)
                a.path.unlink(missing_ok=True)
        return choose(assembled)

    def _progress(self, job: MediaJob) -> str:
        summary = self.store.summary(job.item)
        if summary.drm:
            return (f"DRM-protected ({', '.join(summary.drm_systems) or 'DRM system not named'});"
                    " not archived")
        lines = self.store.progress(job.item, job.duration)
        return "; ".join(lines[:4]) or "nothing captured from the network"

    async def _music(self, job: MediaJob, selection: Selection) -> bool:
        """Produce the track. Returns whether the music part is settled."""
        track_hash = database.make_track_hash(job.artist, job.title)
        if await already_owned(track_hash, MUSIC):
            log.info("Already in the music library: %s - %s", job.artist, job.title)
            return True

        if selection.audio is not None:
            codec = selection.audio.info.audio_codec
            dest = encoder.music_path(job.artist, job.title, encoder.audio_extension(codec))
            if await encoder.remux_audio(selection.audio.path, dest):
                await asyncio.to_thread(
                    encoder.write_tags, dest, job.artist, job.title, job.album,
                    job.album_artist, job.track_number,
                    job.cover[0] if job.cover else None,
                    job.cover[1] if job.cover else "image/jpeg",
                )
                await self._record(MUSIC, track_hash, job, dest)
                return True

        log.info("%s - %s: not complete yet (%s)", job.artist, job.title, self._progress(job))
        return False

    async def _video(self, job: MediaJob, selection: Selection) -> bool:
        """Produce the video. Returns whether the video part is settled."""
        video_hash = database.make_video_hash(job.item)
        if await already_owned(video_hash, VIDEO):
            log.info("Already in the video library: %s", job.title)
            return True
        folder = job.artist or job.site
        meta = {"title": job.title, "artist": job.artist, "comment": job.item}

        if selection.video is not None:
            v, a = selection.video, selection.video_audio
            ext = encoder.video_extension(v.info.video_codec,
                                          (a.info if a else v.info).audio_codec)
            dest = encoder.video_path(folder, job.title, ext)
            if await encoder.mux_video(v.path, a.path if a else None, dest, meta):
                await self._record(VIDEO, video_hash, job, dest)
                return True

        log.info("%s: video not complete yet (%s)", job.title, self._progress(job))
        return False

    async def _record(self, kind: str, media_hash: str, job: MediaJob, dest: Path) -> None:
        recorded = await asyncio.to_thread(
            database.add_media, media_hash, job.artist, job.title, job.source, str(dest),
            kind=kind,
        )
        if recorded:
            log.info("%s library updated: %s (%s)", kind.capitalize(), dest.name,
                     job.site or job.source)
        else:
            log.info("%s was recorded concurrently: %s", kind.capitalize(), media_hash)
