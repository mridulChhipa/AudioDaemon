import re
import struct
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from urllib.parse import parse_qsl, urlencode, urljoin, urlsplit, urlunsplit

UMP_MIME = "application/vnd.yt-ump"
HLS_MIMES = ("application/vnd.apple.mpegurl", "application/x-mpegurl", "audio/mpegurl",
             "audio/x-mpegurl")
DASH_MIMES = ("application/dash+xml",)
_OPAQUE_MIMES = ("application/octet-stream", "binary/octet-stream", "")
_MEDIA_EXTENSIONS = (
    ".mp4", ".m4s", ".m4v", ".m4a", ".mp4a", ".cmfv", ".cmfa", ".webm", ".weba", ".mkv",
    ".ts", ".aac", ".mp3", ".ogg", ".opus", ".m3u8", ".mpd", ".m4f", ".fmp4",
)

_MEDIA_RESOURCE_TYPES = ("Media", "XHR", "Fetch", "Other")


def base_mime(mime: str | None) -> str:
    return (mime or "").split(";")[0].strip().lower()


def is_candidate(url: str, mime: str | None, resource_type: str | None) -> bool:
    """Whether a response might carry media or a media manifest."""
    if resource_type and resource_type not in _MEDIA_RESOURCE_TYPES:
        return False
    m = base_mime(mime)
    if m.startswith(("video/", "audio/")) or m in (UMP_MIME, *HLS_MIMES, *DASH_MIMES):
        return True
    path = urlsplit(url).path.lower()
    if path.endswith(_MEDIA_EXTENSIONS):
        return True
    return m in _OPAQUE_MIMES and resource_type in ("Media", "XHR", "Fetch")

# DRM system IDs, as a `pssh` box or a DASH `urn:uuid:` scheme names them.
_DRM_SYSTEM_IDS = {
    "edef8ba979d64acea3c827dcd51d21ed": "Widevine",
    "9a04f07998404286ab92e65be0885f95": "PlayReady",
    "94ce86fb07ff4f43adb893d2fa968ca2": "FairPlay",
    "1077efecc0b24d02ace33c1e52e2fb4b": "ClearKey",
    "e2719d58a985b3c9781ab030af78d30e": "ClearKey",
    "5e629af538da4063897797ffbd9902d4": "Marlin",
    "f239e769efa348509c16a903c6932efb": "Adobe Primetime",
}
# EME key systems, as a page asks for them.
_KEY_SYSTEMS = {
    "com.widevine": "Widevine",
    "com.microsoft.playready": "PlayReady",
    "com.apple.fps": "FairPlay",
    "org.w3.clearkey": "ClearKey",
}
# HLS KEYFORMATs.
_HLS_KEYFORMATS = {
    "com.apple.streamingkeydelivery": "FairPlay",
    "com.microsoft.playready": "PlayReady",
    "urn:uuid:edef8ba9-79d6-4ace-a3c8-27dcd51d21ed": "Widevine",
}


def drm_system_name(system_id: str) -> str:
    """A DRM system's name from its ID (hex, with or without dashes or `urn:uuid:`)."""
    hex_id = system_id.lower().removeprefix("urn:uuid:").replace("-", "")
    return _DRM_SYSTEM_IDS.get(hex_id, f"unknown DRM {hex_id}")


def key_system_name(key_system: str) -> str | None:
    """A DRM system's name from an EME key system such as `com.widevine.alpha`."""
    key_system = (key_system or "").lower()
    return next((name for prefix, name in _KEY_SYSTEMS.items()
                 if key_system.startswith(prefix)), key_system or None)


# Widevine's EME robustness strings, strongest first, and the security level
# each means. Nothing requested means the software CDM: L3.
_WIDEVINE_ROBUSTNESS = (
    ("HW_SECURE_ALL", "L1"),
    ("HW_SECURE_DECODE", "L1"),
    ("HW_SECURE_CRYPTO", "L2"),
    ("SW_SECURE_DECODE", "L3"),
    ("SW_SECURE_CRYPTO", "L3"),
    ("", "L3"),
)
# PlayReady's security levels, as robustness strings or key-system suffixes.
_PLAYREADY_LEVELS = ("3000", "2000", "150")


def eme_drm_label(key_system: str, video: list[str], audio: list[str]) -> str | None:
    """The DRM system and security level a page's MediaKeys were granted.

    `video` and `audio` are the robustness strings of the configuration the
    browser granted (`MediaKeySystemAccess.getConfiguration()`). The picture's
    robustness is what the level is quoted by; audio's only when there is no
    video.
    """
    name = key_system_name(key_system)
    if name is None:
        return None
    robustness = [r.upper() for r in (video or audio)]
    if name == "Widevine":
        for value, level in _WIDEVINE_ROBUSTNESS:
            if value in robustness or (not robustness and not value):
                shown = value or "no robustness requested"
                return f"Widevine {level} ({shown})"
        return f"Widevine ({', '.join(robustness)})"
    if name == "PlayReady":
        ks = key_system.lower()
        if ks.endswith(".hardware"):
            return "PlayReady SL3000 (hardware)"
        for level in _PLAYREADY_LEVELS:
            if ks.endswith("." + level) or level in robustness:
                return f"PlayReady SL{level}"
    return name


# Every DRM system name the labels above can start with.
DRM_SYSTEM_NAMES = set(_DRM_SYSTEM_IDS.values()) | set(_KEY_SYSTEMS.values())
# What each system calls a hardware-secure and a software-secure playback.
_PLAYBACK_LEVELS = {
    "Widevine": {"hardware": "L1", "software": "L3"},
    "PlayReady": {"hardware": "SL3000", "software": "SL2000"},
}


def level_name(system: str, secure: str) -> str | None:
    """A system's name for a playback's security ("hardware"/"software")."""
    return _PLAYBACK_LEVELS.get(system, {}).get(secure)


def playback_security(props: dict[str, str]) -> tuple[str, str] | None:
    """How securely a live media player decrypts, from its media-log properties.

    The properties are Chromium's (chrome://media-internals, and the DevTools
    Media domain). A protected playback takes one of two paths:

    - hardware-secure: the MediaFoundationRenderer decrypts and decodes in
      the platform's protected path (Widevine L1, PlayReady SL3000);
    - software-secure: the CDM decrypts inside the renderer, through a
      DecryptingDemuxerStream or a Decrypting*Decoder (Widevine L3).

    Returns ("hardware" | "software", how it was seen), or None when the
    properties don't show a protected path (yet).
    """
    def true(name: str) -> bool:
        return str(props.get(name, "")).strip('"').lower() == "true"

    renderer = str(props.get("kRendererName", "")).strip('"')
    if "MediaFoundationRenderer" in renderer:
        return "hardware", f"{renderer}, the hardware-secure path"
    for kind in ("Video", "Audio"):
        if true(f"kIs{kind}DecryptingDemuxerStream"):
            return "software", f"{kind.lower()} decrypted by the software CDM (DecryptingDemuxerStream)"
        decoder = str(props.get(f"k{kind}DecoderName", "")).strip('"')
        if decoder.startswith("Decrypting"):
            return "software", f"{kind.lower()} decrypted and decoded by the software CDM ({decoder})"
    return None


def _add(labels: list[str], label: str) -> None:
    if label not in labels:
        labels.append(label)


MP4, WEBM, MPEGTS, ADTS, MP3, HLS, DASH, UMP, UNKNOWN = (
    "mp4", "webm", "mpegts", "adts", "mp3", "hls", "dash", "ump", "unknown",
)
_MP4_TOP_BOXES = {b"ftyp", b"styp", b"moof", b"moov", b"sidx", b"emsg", b"free", b"prft", b"mdat"}
_EBML_MAGIC = b"\x1a\x45\xdf\xa3"
_EBML_CLUSTER = b"\x1f\x43\xb6\x75"


def sniff_container(body: bytes, mime: str | None = None) -> str:
    """Identify a body's container from its first bytes (and MIME for UMP)."""
    m = base_mime(mime)
    if m == UMP_MIME:
        return UMP
    if not body:
        return UNKNOWN

    head = body[:1024].lstrip(b"\xef\xbb\xbf \t\r\n")
    if head.startswith(b"#EXTM3U"):
        return HLS
    if b"<MPD" in head:
        return DASH
    if len(body) >= 8 and body[4:8] in _MP4_TOP_BOXES:
        return MP4
    if body.startswith((_EBML_MAGIC, _EBML_CLUSTER)):
        return WEBM
    if body[0] == 0x47 and (len(body) < 189 or body[188] == 0x47):
        return MPEGTS
    if body.startswith(b"ID3"):
        return MP3
    if len(body) >= 2 and body[0] == 0xFF:
        if body[1] & 0xF6 == 0xF0:   # sync + layer 00: ADTS
            return ADTS
        if body[1] & 0xE0 == 0xE0:   # sync + an MPEG audio layer
            return MP3
    if m in HLS_MIMES:
        return HLS
    return UNKNOWN


# --------------------------------------------------------------------------
# ISO BMFF (MP4) boxes
# --------------------------------------------------------------------------

# Boxes whose payload is itself a sequence of boxes.
_CONTAINER_BOXES = {
    b"moov", b"trak", b"mdia", b"minf", b"stbl", b"mvex", b"moof", b"traf", b"edts",
    b"dinf", b"sinf", b"schi", b"udta",
}
_CODEC_NAMES = {
    b"avc1": "h264", b"avc3": "h264", b"hvc1": "hevc", b"hev1": "hevc", b"dvh1": "hevc",
    b"dvhe": "hevc", b"vp08": "vp8", b"vp09": "vp9", b"av01": "av1", b"mp4v": "mpeg4",
    b"mp4a": "aac", b"Opus": "opus", b"opus": "opus", b"ac-3": "ac3", b"ec-3": "eac3",
    b"fLaC": "flac", b"mp3 ": "mp3", b".mp3": "mp3", b"alac": "alac",
}


def iter_boxes(data: bytes, start: int = 0, end: int | None = None):
    """Yield (type, payload_start, box_end) for each box in data[start:end].

    Stops quietly at a truncated or malformed box: segments are often cut
    mid-box by byte-range requests.
    """
    end = len(data) if end is None else min(end, len(data))
    pos = start
    while pos + 8 <= end:
        size, kind = struct.unpack_from(">I4s", data, pos)
        header = 8
        if size == 1:
            if pos + 16 > end:
                return
            size = struct.unpack_from(">Q", data, pos + 8)[0]
            header = 16
        elif size == 0:
            size = end - pos
        if size < header:
            return
        box_end = pos + size
        yield kind, pos + header, min(box_end, end)
        if box_end > end:
            return
        pos = box_end


@dataclass
class Mp4Info:
    has_init: bool = False          # a moov was present
    has_media: bool = False         # a moof or mdat was present
    handlers: list[str] = field(default_factory=list)   # "vide" / "soun" / ...
    codecs: list[str] = field(default_factory=list)
    encrypted: bool = False
    drm: list[str] = field(default_factory=list)  # systems (pssh) and scheme (schm)
    duration: float = 0.0           # from mvhd, when it's filled in


def _walk_mp4(data: bytes, start: int, end: int, info: Mp4Info, depth: int = 0) -> None:
    if depth > 8:
        return
    for kind, payload, box_end in iter_boxes(data, start, end):
        if kind == b"moov":
            info.has_init = True
        elif kind in (b"moof", b"mdat"):
            info.has_media = True
        if kind in (b"pssh", b"tenc", b"senc"):
            # pssh (key system data), senc (per-sample IVs), tenc: all CENC.
            info.encrypted = True
        if kind == b"pssh" and payload + 20 <= box_end:
            # A full box: version and flags, then the 16-byte DRM system ID.
            _add(info.drm, drm_system_name(data[payload + 4 : payload + 20].hex()))
        elif kind == b"schm" and payload + 8 <= box_end:
            # The protection scheme: cenc / cbcs (AES-CTR / AES-CBC), or others.
            scheme = data[payload + 4 : payload + 8].decode("latin-1").strip()
            _add(info.drm, f"scheme {scheme}")
        if kind == b"hdlr" and payload + 12 <= box_end:
            info.handlers.append(data[payload + 8 : payload + 12].decode("latin-1"))
        elif kind == b"mvhd" and payload + 4 <= box_end:
            version = data[payload]
            try:
                if version == 1:
                    scale, dur = struct.unpack_from(">IQ", data, payload + 20)
                else:
                    scale, dur = struct.unpack_from(">II", data, payload + 12)
                if scale and dur not in (0, 0xFFFFFFFF, 0xFFFFFFFFFFFFFFFF):
                    info.duration = dur / scale
            except struct.error:
                pass
        elif kind == b"stsd" and payload + 8 <= box_end:
            # Sample entries: the fourcc is the codec, and encv/enca mean CENC.
            # Their `sinf` (with the scheme) follows the sample entry's own
            # fields: 78 bytes for video, 28 for audio.
            for entry, entry_payload, entry_end in iter_boxes(data, payload + 8, box_end):
                if entry in (b"encv", b"enca"):
                    info.encrypted = True
                    skip = 78 if entry == b"encv" else 28
                    _walk_mp4(data, entry_payload + skip, entry_end, info, depth + 1)
                name = _CODEC_NAMES.get(entry)
                if name:
                    info.codecs.append(name)
        elif kind in _CONTAINER_BOXES:
            _walk_mp4(data, payload, box_end, info, depth + 1)


def mp4_info(data: bytes) -> Mp4Info:
    info = Mp4Info()
    _walk_mp4(data, 0, len(data), info)
    return info


# --------------------------------------------------------------------------
# Matroska / WebM
# --------------------------------------------------------------------------

_EBML_IDS = {
    "segment": 0x18538067, "info": 0x1549A966, "tracks": 0x1654AE6B, "track": 0xAE,
    "track_type": 0x83, "encodings": 0x6D80, "encoding": 0x6240,
    "encryption": 0x5035, "cluster": 0x1F43B675, "duration": 0x4489, "scale": 0x2AD7B1,
}
_UNKNOWN_SIZE = -1


def _read_vint(data: bytes, pos: int, keep_marker: bool) -> tuple[int, int] | None:
    if pos >= len(data):
        return None
    first = data[pos]
    length = 1
    mask = 0x80
    while length <= 8 and not first & mask:
        mask >>= 1
        length += 1
    if length > 8 or pos + length > len(data):
        return None
    value = first if keep_marker else first & (mask - 1)
    all_ones = (first & (mask - 1)) == mask - 1
    for i in range(1, length):
        value = (value << 8) | data[pos + i]
        all_ones = all_ones and data[pos + i] == 0xFF
    if not keep_marker and all_ones:
        return _UNKNOWN_SIZE, pos + length
    return value, pos + length


@dataclass
class WebmInfo:
    has_init: bool = False
    has_media: bool = False
    track_types: list[int] = field(default_factory=list)  # 1 video, 2 audio
    encrypted: bool = False
    duration: float = 0.0


def webm_info(data: bytes) -> WebmInfo:
    info = WebmInfo()
    ids = _EBML_IDS
    masters = {ids["segment"], ids["tracks"], ids["track"], ids["encodings"],
               ids["encoding"], ids["info"]}
    scale = 1_000_000
    raw_duration = 0.0

    def walk(pos: int, end: int, depth: int) -> None:
        nonlocal scale, raw_duration
        while pos < end and depth < 8:
            got_id = _read_vint(data, pos, keep_marker=True)
            if got_id is None:
                return
            el_id, pos = got_id
            got_size = _read_vint(data, pos, keep_marker=False)
            if got_size is None:
                return
            size, pos = got_size
            el_end = end if size == _UNKNOWN_SIZE else min(pos + size, end)

            if el_id == ids["cluster"]:
                info.has_media = True
                return  # media from here on; nothing more to learn
            if el_id == ids["tracks"]:
                info.has_init = True
            if el_id == ids["encryption"]:
                info.encrypted = True
            if el_id in masters:
                walk(pos, el_end, depth + 1)
            elif el_id == ids["track_type"] and size > 0:
                info.track_types.append(int.from_bytes(data[pos:el_end], "big"))
            elif el_id == ids["scale"] and size > 0:
                scale = int.from_bytes(data[pos:el_end], "big") or scale
            elif el_id == ids["duration"] and size in (4, 8):
                fmt = ">f" if size == 4 else ">d"
                raw_duration = struct.unpack(fmt, data[pos:el_end])[0]
            if size == _UNKNOWN_SIZE:
                return
            pos = el_end

    if data.startswith(_EBML_CLUSTER):
        info.has_media = True
        return info

    pos = 0
    while pos < len(data):
        got_id = _read_vint(data, pos, keep_marker=True)
        if got_id is None:
            break
        el_id, after_id = got_id
        got_size = _read_vint(data, after_id, keep_marker=False)
        if got_size is None:
            break
        size, body = got_size
        if el_id == ids["segment"]:
            walk(body, len(data) if size == _UNKNOWN_SIZE else body + size, 1)
            break
        if size == _UNKNOWN_SIZE:
            break
        pos = body + size

    if raw_duration:
        info.duration = raw_duration * scale / 1e9
    return info


# --------------------------------------------------------------------------
# One summary, whatever the container
# --------------------------------------------------------------------------

AUDIO, VIDEO, MUXED = "audio", "video", "muxed"


@dataclass
class MediaFacts:
    """What the bytes themselves say about a piece of media."""

    container: str
    kind: str | None = None         # AUDIO / VIDEO / MUXED, when known
    encrypted: bool = False
    drm: tuple[str, ...] = ()       # which DRM, when the bytes say
    is_init: bool = False           # carries the decoder setup (moov / Tracks)
    has_media: bool = False
    duration: float = 0.0


def _kind(has_video: bool, has_audio: bool) -> str | None:
    if has_video and has_audio:
        return MUXED
    if has_video:
        return VIDEO
    if has_audio:
        return AUDIO
    return None


def kind_from_mime(mime: str | None) -> str | None:
    m = base_mime(mime)
    if m.startswith("video/"):
        return VIDEO
    if m.startswith("audio/"):
        return AUDIO
    return None


_VIDEO_CODECS = {"h264", "hevc", "vp8", "vp9", "av1", "mpeg4"}
_AUDIO_CODECS = {"aac", "opus", "ac3", "eac3", "flac", "mp3", "alac"}


def media_facts(body: bytes, mime: str | None = None) -> MediaFacts:
    container = sniff_container(body, mime)
    if container == MP4:
        info = mp4_info(body)
        has_video = "vide" in info.handlers or bool(_VIDEO_CODECS & set(info.codecs))
        has_audio = "soun" in info.handlers or bool(_AUDIO_CODECS & set(info.codecs))
        return MediaFacts(container, _kind(has_video, has_audio) or kind_from_mime(mime),
                          info.encrypted, tuple(info.drm), info.has_init, info.has_media,
                          info.duration)
    if container == WEBM:
        info = webm_info(body)
        kind = _kind(1 in info.track_types, 2 in info.track_types)
        # Matroska names only the cipher; the DRM system is the page's EME.
        drm = ("Matroska AES encryption",) if info.encrypted else ()
        return MediaFacts(container, kind or kind_from_mime(mime), info.encrypted, drm,
                          info.has_init, info.has_media, info.duration)
    if container in (ADTS, MP3):
        return MediaFacts(container, AUDIO, has_media=True)
    if container == MPEGTS:
        return MediaFacts(container, kind_from_mime(mime), has_media=True)
    return MediaFacts(container, kind_from_mime(mime))


# --------------------------------------------------------------------------
# HLS
# --------------------------------------------------------------------------

@dataclass
class HlsVariant:
    uri: str
    codecs: str = ""


@dataclass
class HlsSegment:
    uri: str
    sequence: int
    start: float
    duration: float
    byterange: tuple[int, int] | None = None   # (offset, length)


@dataclass
class HlsPlaylist:
    variants: list[HlsVariant] = field(default_factory=list)
    segments: list[HlsSegment] = field(default_factory=list)
    init_uri: str | None = None
    ended: bool = False
    drm: list[str] = field(default_factory=list)   # e.g. "FairPlay (SAMPLE-AES)"

    @property
    def encrypted(self) -> bool:
        return bool(self.drm)

    @property
    def is_master(self) -> bool:
        return bool(self.variants)

    @property
    def duration(self) -> float:
        return sum(s.duration for s in self.segments)


_ATTR = re.compile(r'([A-Z0-9-]+)=("[^"]*"|[^,]*)')


def _attrs(line: str) -> dict[str, str]:
    body = line.split(":", 1)[1] if ":" in line else ""
    return {k: v.strip('"') for k, v in _ATTR.findall(body)}


def parse_hls(text: str, url: str) -> HlsPlaylist:
    """Parse a master or media playlist, resolving URIs against `url`."""
    playlist = HlsPlaylist()
    lines = [line.strip() for line in text.splitlines()]
    sequence = 0
    start = 0.0
    pending_duration: float | None = None
    pending_variant: dict[str, str] | None = None
    pending_range: tuple[int, int] | None = None
    next_offset = 0

    for line in lines:
        if not line:
            continue
        if line.startswith("#EXT-X-MEDIA-SEQUENCE:"):
            try:
                sequence = int(line.split(":", 1)[1])
            except ValueError:
                pass
        elif line.startswith("#EXTINF:"):
            try:
                pending_duration = float(line.split(":", 1)[1].split(",")[0])
            except ValueError:
                pending_duration = 0.0
        elif line.startswith("#EXT-X-BYTERANGE:"):
            spec = line.split(":", 1)[1]
            length, _, offset = spec.partition("@")
            try:
                off = int(offset) if offset else next_offset
                pending_range = (off, int(length))
                next_offset = off + int(length)
            except ValueError:
                pending_range = None
        elif line.startswith("#EXT-X-STREAM-INF:"):
            pending_variant = _attrs(line)
        elif line.startswith("#EXT-X-MAP:"):
            uri = _attrs(line).get("URI")
            if uri:
                playlist.init_uri = urljoin(url, uri)
        elif line.startswith("#EXT-X-KEY:") or line.startswith("#EXT-X-SESSION-KEY:"):
            attrs = _attrs(line)
            method = attrs.get("METHOD", "NONE").upper()
            if method != "NONE":
                # AES-128 or SAMPLE-AES: the fetched bytes are ciphertext.
                keyformat = attrs.get("KEYFORMAT", "identity").lower()
                system = _HLS_KEYFORMATS.get(keyformat)
                if system is None and keyformat.startswith("urn:uuid:"):
                    system = drm_system_name(keyformat)
                _add(playlist.drm, f"{system} ({method})" if system else f"HLS {method}")
        elif line.startswith("#EXT-X-ENDLIST"):
            playlist.ended = True
        elif line.startswith("#"):
            continue
        elif pending_variant is not None:
            playlist.variants.append(HlsVariant(
                uri=urljoin(url, line), codecs=pending_variant.get("CODECS", "")))
            pending_variant = None
        elif pending_duration is not None:
            playlist.segments.append(HlsSegment(
                uri=urljoin(url, line), sequence=sequence, start=start,
                duration=pending_duration, byterange=pending_range,
            ))
            sequence += 1
            start += pending_duration
            pending_duration = None
            pending_range = None
    return playlist


def kind_from_codecs(codecs: str) -> str | None:
    """AUDIO / VIDEO / MUXED from an RFC 6381 CODECS string."""
    parts = [c.strip().lower() for c in codecs.split(",") if c.strip()]
    video = any(p.startswith(("avc", "hvc", "hev", "vp0", "vp8", "vp9", "av01", "dvh"))
                for p in parts)
    audio = any(p.startswith(("mp4a", "opus", "ac-3", "ec-3", "flac", "vorbis"))
                for p in parts)
    return _kind(video, audio)


# --------------------------------------------------------------------------
# DASH
# --------------------------------------------------------------------------

@dataclass
class DashRepresentation:
    id: str
    kind: str | None
    init_url: str | None
    # Segment URL -> (sequence, start seconds, duration seconds)
    segments: dict[str, tuple[int, float, float]]
    base_url: str | None      # SegmentBase / single-file representations
    drm: list[str]            # from ContentProtection: systems and scheme

    @property
    def encrypted(self) -> bool:
        return bool(self.drm)


@dataclass
class DashManifest:
    duration: float
    representations: list[DashRepresentation]


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _child(el, name):
    for c in el:
        if _local(c.tag) == name:
            return c
    return None


def _children(el, name):
    return [c for c in el if _local(c.tag) == name]


def _protection(el) -> list[str]:
    """The DRM an element's ContentProtection children name."""
    labels: list[str] = []
    for cp in _children(el, "ContentProtection"):
        scheme = (cp.get("schemeIdUri") or "").lower()
        if scheme.startswith("urn:uuid:"):
            _add(labels, drm_system_name(scheme))
        elif scheme == "urn:mpeg:dash:mp4protection:2011" and cp.get("value"):
            _add(labels, f"scheme {cp.get('value')}")
        else:
            _add(labels, f"ContentProtection {scheme or '(unnamed)'}")
    return labels


def _base(el, parent: str) -> str:
    """An element's BaseURL, resolved against its parent's; the parent's if none."""
    child = _child(el, "BaseURL")
    return urljoin(parent, child.text.strip()) if child is not None and child.text else parent


_ISO_DURATION = re.compile(
    r"P(?:(?P<d>\d+(?:\.\d+)?)D)?(?:T(?:(?P<h>\d+(?:\.\d+)?)H)?"
    r"(?:(?P<m>\d+(?:\.\d+)?)M)?(?:(?P<s>\d+(?:\.\d+)?)S)?)?"
)


def parse_iso_duration(text: str | None) -> float:
    if not text:
        return 0.0
    m = _ISO_DURATION.fullmatch(text.strip())
    if not m:
        return 0.0
    parts = {k: float(v) if v else 0.0 for k, v in m.groupdict().items()}
    return parts["d"] * 86400 + parts["h"] * 3600 + parts["m"] * 60 + parts["s"]


_TEMPLATE_VAR = re.compile(r"\$(RepresentationID|Number|Time|Bandwidth)(?:%0(\d+)d)?\$")


def _fill(template: str, rep_id: str, bandwidth: int, number=None, time=None) -> str:
    def sub(m):
        name, width = m.group(1), m.group(2)
        value = {"RepresentationID": rep_id, "Bandwidth": bandwidth,
                 "Number": number, "Time": time}[name]
        if value is None:
            return m.group(0)
        if width and name != "RepresentationID":
            return str(value).zfill(int(width))
        return str(value)
    return _TEMPLATE_VAR.sub(sub, template).replace("$$", "$")


def parse_dash(xml_text: str, url: str, max_segments: int = 20_000) -> DashManifest:
    """Parse an MPD's representations and enumerate their segment URLs.

    Covers SegmentTemplate (with $Number$ or a SegmentTimeline), SegmentList
    and single-file SegmentBase representations -- what on-demand sites use.
    """
    root = ET.fromstring(xml_text)
    duration = parse_iso_duration(root.get("mediaPresentationDuration"))
    reps: list[DashRepresentation] = []

    base = _base(root, url)

    for period in _children(root, "Period"):
        p_base = _base(period, base)
        p_duration = parse_iso_duration(period.get("duration")) or duration

        for aset in _children(period, "AdaptationSet"):
            a_base = _base(aset, p_base)
            a_drm = _protection(aset)
            a_template = _child(aset, "SegmentTemplate")
            content = (aset.get("contentType") or aset.get("mimeType") or "").lower()

            for rep in _children(aset, "Representation"):
                rep_id = rep.get("id", "")
                bandwidth = int(rep.get("bandwidth", "0") or 0)
                codecs = rep.get("codecs") or aset.get("codecs") or ""
                mime = (rep.get("mimeType") or content)
                kind = kind_from_codecs(codecs) or kind_from_mime(mime) or (
                    VIDEO if "video" in content else AUDIO if "audio" in content else None)
                r_base = _base(rep, a_base)
                drm = a_drm + [p for p in _protection(rep) if p not in a_drm]

                template = _child(rep, "SegmentTemplate")
                if template is None:
                    template = a_template
                seg_list = _child(rep, "SegmentList")
                segments: dict[str, tuple[int, float, float]] = {}
                init_url = None
                single = None

                if template is not None:
                    timescale = int(template.get("timescale", "1") or 1)
                    start_number = int(template.get("startNumber", "1") or 1)
                    media = template.get("media")
                    if template.get("initialization"):
                        init_url = urljoin(r_base, _fill(template.get("initialization"),
                                                          rep_id, bandwidth))
                    timeline = _child(template, "SegmentTimeline")
                    if media and timeline is not None:
                        number, t = start_number, 0
                        for s in _children(timeline, "S"):
                            if s.get("t") is not None:
                                t = int(s.get("t"))
                            d = int(s.get("d", "0"))
                            for _ in range(int(s.get("r", "0")) + 1):
                                seg_url = urljoin(r_base, _fill(media, rep_id, bandwidth,
                                                                number=number, time=t))
                                segments[seg_url] = (number, t / timescale, d / timescale)
                                number += 1
                                t += d
                                if len(segments) >= max_segments:
                                    break
                    elif media and template.get("duration"):
                        d = int(template.get("duration")) / timescale
                        count = int(-(-p_duration // d)) if d and p_duration else 0
                        for i in range(min(count, max_segments)):
                            number = start_number + i
                            seg_url = urljoin(r_base, _fill(media, rep_id, bandwidth,
                                                            number=number))
                            segments[seg_url] = (number, i * d, min(d, p_duration - i * d))
                elif seg_list is not None:
                    timescale = int(seg_list.get("timescale", "1") or 1)
                    d = int(seg_list.get("duration", "0") or 0) / timescale
                    init = _child(seg_list, "Initialization")
                    if init is not None and init.get("sourceURL"):
                        init_url = urljoin(r_base, init.get("sourceURL"))
                    for i, seg in enumerate(_children(seg_list, "SegmentURL")):
                        if seg.get("media"):
                            segments[urljoin(r_base, seg.get("media"))] = (i, i * d, d)
                else:
                    single = r_base

                reps.append(DashRepresentation(
                    id=rep_id, kind=kind, init_url=init_url, segments=segments, base_url=single,
                    drm=drm,
                ))
    return DashManifest(duration=duration, representations=reps)


# --------------------------------------------------------------------------
# Byte ranges and stream identity
# --------------------------------------------------------------------------

_CONTENT_RANGE = re.compile(r"bytes\s+(\d+)-(\d+)/(\d+|\*)", re.I)


def content_range(value: str | None) -> tuple[int, int, int | None] | None:
    """(first, last, total) from a Content-Range header; total None if unknown."""
    if not value:
        return None
    m = _CONTENT_RANGE.search(value)
    if not m:
        return None
    total = None if m.group(3) == "*" else int(m.group(3))
    return int(m.group(1)), int(m.group(2)), total


def query_range(url: str) -> tuple[int, int] | None:
    """A `range=first-last` query parameter, as googlevideo and others use."""
    for key, value in parse_qsl(urlsplit(url).query, keep_blank_values=True):
        if key == "range":
            first, _, last = value.partition("-")
            if first.isdigit() and last.isdigit():
                return int(first), int(last)
    return None


def query_value(url: str, name: str) -> str | None:
    for key, value in parse_qsl(urlsplit(url).query, keep_blank_values=True):
        if key == name:
            return value
    return None


# Parameters that differ between requests for pieces of the same file.
_VOLATILE_PARAMS = {
    "range", "rn", "rbuf", "bytes", "sq", "segment", "cpn", "_", "t", "ts", "cb",
    "alr", "ump", "srfvp", "pot", "expire", "ei", "sig", "lsig", "rqh", "c", "cver",
    "token", "hdnts", "hdnea", "policy", "signature", "key-pair-id", "x-amz-signature",
    "x-amz-date", "x-amz-credential", "x-amz-security-token", "x-amz-expires",
}


def stream_url_key(url: str) -> str:
    """Identify the file a byte-range request is a piece of.

    googlevideo URLs carry the file identity in `id` + `itag`; everything
    else about them changes between requests. Elsewhere, drop the parameters
    that vary per piece or per signature and keep the rest.
    """
    parts = urlsplit(url)
    if parts.hostname and parts.hostname.endswith("googlevideo.com"):
        return f"gv:{query_value(url, 'id')}:{query_value(url, 'itag')}"
    kept = sorted((k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True)
                  if k.lower() not in _VOLATILE_PARAMS)
    return urlunsplit(("", parts.netloc.lower(), parts.path, urlencode(kept), ""))


_TRACKING_PARAMS = {"utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
                    "fbclid", "gclid", "si", "feature", "t", "pp", "ab_channel", "list",
                    "index", "start_radio", "ref"}
_YT_ID = re.compile(r"^[A-Za-z0-9_-]{11}$")


def page_id(url: str) -> str:
    """A stable identity for the page a piece of media belongs to.

    YouTube in all its URL shapes collapses to `yt:<video id>`; any other page
    is its host and path plus non-tracking query parameters.
    """
    parts = urlsplit(url or "")
    host = (parts.hostname or "").lower()
    if host.startswith("www."):
        host = host[4:]
    if host.endswith("youtube.com") or host == "youtu.be" or host.endswith("youtube-nocookie.com"):
        candidate = None
        if host == "youtu.be":
            candidate = parts.path.strip("/").split("/")[0]
        elif parts.path.startswith(("/shorts/", "/embed/", "/live/")):
            candidate = parts.path.split("/")[2] if len(parts.path.split("/")) > 2 else None
        else:
            candidate = query_value(url, "v")
        if candidate and _YT_ID.match(candidate):
            return f"yt:{candidate}"
    kept = sorted((k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True)
                  if k.lower() not in _TRACKING_PARAMS)
    return urlunsplit(("", host, parts.path.rstrip("/") or "/", urlencode(kept), ""))


def site_name(url: str) -> str:
    """The site a page is on, for folder names and the classifier: "youtube.com"."""
    host = (urlsplit(url or "").hostname or "").lower()
    return host[4:] if host.startswith("www.") else host
