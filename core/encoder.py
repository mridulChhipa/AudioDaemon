"""Turn captured network streams into tagged library files.

Streams are the source's own encoded bytes, so they are only *remuxed*
(`-c copy`): audio into its native container (.m4a, .opus, ...), video with
its audio into .mp4 / .webm / .mkv. No second generation of loss.

Tags come from the media session, which knows the real album art.
"""
import asyncio
import json
import logging
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from core.config import FFMPEG_DIR, MUSIC_DIR, VIDEO_DIR

log = logging.getLogger(__name__)

_ILLEGAL_FILENAME = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
# Otherwise a console flashes up on every ffmpeg run under pythonw.
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def build_filename_stem(artist: str, title: str) -> str:
    """Build the library filename from the cleaned artist and title."""
    artist, title = (artist or "").strip(), (title or "").strip()
    stem = f"{artist} - {title}" if artist else title
    return safe_component(stem)


def safe_component(text: str) -> str:
    """Make any string usable as one Windows path component."""
    cleaned = _ILLEGAL_FILENAME.sub("_", (text or "").strip()).strip(" .") or "unknown"
    return cleaned[:180]  # Windows caps a path component at 255


def _tool(name: str) -> str:
    """`name` (ffmpeg, ffprobe) from PATH, else the configured static build."""
    found = shutil.which(name)
    if found:
        return found
    candidate = FFMPEG_DIR / f"{name}.exe"
    return str(candidate) if candidate.exists() else name


def _run(args: list[str], *, what: str = "ffmpeg") -> bool:
    proc = subprocess.run(args, capture_output=True, creationflags=_NO_WINDOW)
    if proc.returncode != 0:
        log.error("%s failed (%s): %s", what, proc.returncode,
                  proc.stderr.decode("utf-8", "replace")[-600:])
        return False
    return True


# --------------------------------------------------------------------------
# Probing
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class ProbeInfo:
    duration: float
    video_codec: str | None
    audio_codec: str | None
    width: int = 0
    height: int = 0

    @property
    def has_video(self) -> bool:
        return self.video_codec is not None

    @property
    def has_audio(self) -> bool:
        return self.audio_codec is not None


# Codecs ffprobe reports for pictures that aren't a video track: cover art.
_STILL_CODECS = {"mjpeg", "png", "bmp", "gif", "webp"}


def probe(path: Path) -> ProbeInfo | None:
    """What ffprobe makes of a file, or None if it can't read it."""
    proc = subprocess.run(
        [_tool("ffprobe"), "-v", "error", "-print_format", "json",
         "-show_entries", "stream=codec_type,codec_name,width,height,duration:format=duration",
         str(path)],
        capture_output=True, creationflags=_NO_WINDOW,
    )
    if proc.returncode != 0:
        return None
    try:
        data = json.loads(proc.stdout or b"{}")
    except ValueError:
        return None
    video = audio = None
    width = height = 0
    stream_duration = 0.0
    for s in data.get("streams", []):
        if s.get("codec_type") == "video" and video is None and s.get("codec_name") not in _STILL_CODECS:
            video, width, height = s.get("codec_name"), int(s.get("width") or 0), int(s.get("height") or 0)
        elif s.get("codec_type") == "audio" and audio is None:
            audio = s.get("codec_name")
        try:
            stream_duration = max(stream_duration, float(s.get("duration") or 0))
        except ValueError:
            pass
    try:
        duration = float((data.get("format") or {}).get("duration") or 0) or stream_duration
    except ValueError:
        duration = stream_duration
    if video is None and audio is None:
        return None
    return ProbeInfo(duration, video, audio, width, height)


# --------------------------------------------------------------------------
# Containers
# --------------------------------------------------------------------------

_AUDIO_EXT = {"aac": ".m4a", "alac": ".m4a", "opus": ".opus", "vorbis": ".ogg",
              "mp3": ".mp3", "flac": ".flac"}
_MP4_VIDEO = {"h264", "hevc", "mpeg4", "av1"}
_MP4_AUDIO = {"aac", "mp3", "ac3", "eac3", "alac", None}
_WEBM_VIDEO = {"vp8", "vp9", "av1"}
_WEBM_AUDIO = {"opus", "vorbis", None}


def audio_extension(codec: str | None) -> str:
    return _AUDIO_EXT.get(codec or "", ".mka")


def video_extension(video_codec: str | None, audio_codec: str | None) -> str:
    """The most widely playable container that holds both without re-encoding."""
    if video_codec in _MP4_VIDEO and audio_codec in _MP4_AUDIO:
        return ".mp4"
    if video_codec in _WEBM_VIDEO and audio_codec in _WEBM_AUDIO:
        return ".webm"
    return ".mkv"


def music_path(artist: str, title: str, ext: str) -> Path:
    return MUSIC_DIR / f"{build_filename_stem(artist, title)}{ext}"


def video_path(folder: str, title: str, ext: str) -> Path:
    """library/video/<channel or site>/<title>.<ext>, never overwriting."""
    base = VIDEO_DIR / safe_component(folder or "Unsorted")
    stem = safe_component(title)
    dest = base / f"{stem}{ext}"
    n = 2
    while dest.exists():
        dest = base / f"{stem} ({n}){ext}"
        n += 1
    return dest


def _metadata_args(meta: dict[str, str]) -> list[str]:
    args = []
    for key, value in meta.items():
        if value:
            args += ["-metadata", f"{key}={value}"]
    return args


def _remux_audio_blocking(src: Path, dest: Path) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    args = [_tool("ffmpeg"), "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(src), "-map", "0:a:0", "-vn", "-c", "copy"]
    if dest.suffix == ".m4a":
        args += ["-movflags", "+faststart"]
    return _run(args + [str(dest)], what="audio remux")


async def remux_audio(src: Path, dest: Path) -> bool:
    """Copy the first audio track of `src` into `dest`'s container, untouched."""
    return await asyncio.to_thread(_remux_audio_blocking, src, dest)


def _mux_video_blocking(video: Path, audio: Path | None, dest: Path,
                        meta: dict[str, str]) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    args = [_tool("ffmpeg"), "-hide_banner", "-loglevel", "error", "-y", "-i", str(video)]
    if audio is not None:
        args += ["-i", str(audio), "-map", "0:v:0", "-map", "1:a:0"]
    else:
        args += ["-map", "0:v:0", "-map", "0:a:0?"]
    args += ["-c", "copy"] + _metadata_args(meta)
    if dest.suffix == ".mp4":
        args += ["-movflags", "+faststart"]
    return _run(args + [str(dest)], what="video mux")


async def mux_video(video: Path, audio: Path | None, dest: Path,
                    meta: dict[str, str] | None = None) -> bool:
    """Mux a video track (and a separate audio track, if given) without re-encoding."""
    return await asyncio.to_thread(_mux_video_blocking, video, audio, dest, meta or {})


# --------------------------------------------------------------------------
# Tags
# --------------------------------------------------------------------------

def write_tags(
    path: Path,
    artist: str,
    title: str,
    album: str = "",
    album_artist: str = "",
    track_number: int = 0,
    cover: bytes | None = None,
    cover_mime: str = "image/jpeg",
) -> None:
    """Write tags and embed cover art in any library audio format. Never raises."""
    try:
        suffix = path.suffix.lower()
        if suffix == ".flac":
            _tag_flac(path, artist, title, album, album_artist, track_number, cover, cover_mime)
        elif suffix == ".m4a":
            _tag_mp4(path, artist, title, album, album_artist, track_number, cover, cover_mime)
        elif suffix in (".opus", ".ogg"):
            _tag_ogg(path, artist, title, album, album_artist, track_number, cover, cover_mime)
        elif suffix == ".mp3":
            _tag_mp3(path, artist, title, album, album_artist, track_number, cover, cover_mime)
        else:
            log.debug("No tagger for %s; leaving it untagged", suffix)
    except Exception:
        # An untagged track is still worth keeping.
        log.warning("Could not tag %s", path.name, exc_info=True)


def _picture(cover: bytes, cover_mime: str):
    from mutagen.flac import Picture

    picture = Picture()
    picture.data = cover
    picture.type = 3  # front cover
    picture.mime = cover_mime
    return picture


def _tag_flac(path, artist, title, album, album_artist, track_number, cover, cover_mime):
    from mutagen.flac import FLAC

    audio = FLAC(path)
    _vorbis_comments(audio, artist, title, album, album_artist, track_number)
    if cover:
        audio.clear_pictures()
        audio.add_picture(_picture(cover, cover_mime))
    audio.save()


def _tag_ogg(path, artist, title, album, album_artist, track_number, cover, cover_mime):
    import base64

    from mutagen.oggopus import OggOpus
    from mutagen.oggvorbis import OggVorbis

    audio = OggOpus(path) if path.suffix.lower() == ".opus" else OggVorbis(path)
    _vorbis_comments(audio, artist, title, album, album_artist, track_number)
    if cover:
        # Ogg carries FLAC's picture block, base64'd, in a comment.
        block = _picture(cover, cover_mime).write()
        audio["metadata_block_picture"] = [base64.b64encode(block).decode("ascii")]
    audio.save()


def _vorbis_comments(audio, artist, title, album, album_artist, track_number):
    audio["title"] = title
    if artist:
        audio["artist"] = artist
    if album:
        audio["album"] = album
    if album_artist:
        audio["albumartist"] = album_artist
    if track_number:
        audio["tracknumber"] = str(track_number)


def _tag_mp4(path, artist, title, album, album_artist, track_number, cover, cover_mime):
    from mutagen.mp4 import MP4, MP4Cover

    audio = MP4(path)
    audio["\xa9nam"] = title
    if artist:
        audio["\xa9ART"] = artist
    if album:
        audio["\xa9alb"] = album
    if album_artist:
        audio["aART"] = album_artist
    if track_number:
        audio["trkn"] = [(track_number, 0)]
    if cover:
        fmt = MP4Cover.FORMAT_PNG if "png" in cover_mime else MP4Cover.FORMAT_JPEG
        audio["covr"] = [MP4Cover(cover, imageformat=fmt)]
    audio.save()


def _tag_mp3(path, artist, title, album, album_artist, track_number, cover, cover_mime):
    from mutagen.id3 import APIC, ID3, TALB, TIT2, TPE1, TPE2, TRCK, ID3NoHeaderError

    try:
        tags = ID3(path)
    except ID3NoHeaderError:
        tags = ID3()
    tags.add(TIT2(encoding=3, text=title))
    if artist:
        tags.add(TPE1(encoding=3, text=artist))
    if album:
        tags.add(TALB(encoding=3, text=album))
    if album_artist:
        tags.add(TPE2(encoding=3, text=album_artist))
    if track_number:
        tags.add(TRCK(encoding=3, text=str(track_number)))
    if cover:
        tags.delall("APIC")
        tags.add(APIC(encoding=3, mime=cover_mime, type=3, desc="Cover", data=cover))
    tags.save(path)


async def read_session_cover(info) -> tuple[bytes, str] | None:
    """Pull the cover art out of an SMTC media-properties object."""
    ref = getattr(info, "thumbnail", None)
    if ref is None:
        return None
    try:
        from winrt.windows.storage.streams import DataReader

        stream = await ref.open_read_async()
        if not stream.size:
            return None
        reader = DataReader(stream.get_input_stream_at(0))
        loaded = await reader.load_async(stream.size)
        # read_bytes fills a buffer you supply; passing a count raises TypeError.
        buffer = bytearray(loaded)
        reader.read_bytes(buffer)
        data = bytes(buffer)
        mime = stream.content_type or "image/jpeg"
        return (data, mime) if data else None
    except Exception:
        # Warning, not debug: silence here looks identical to a player with no
        # artwork, which hid a real bug in this function.
        log.warning("Could not read cover art from the session", exc_info=True)
        return None

