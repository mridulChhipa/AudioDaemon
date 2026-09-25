"""YouTube's UMP framing -- the one site-specific adapter.

YouTube's player (SABR streaming) POSTs to googlevideo and gets back
`application/vnd.yt-ump`: a stream of typed parts rather than plain segments.
The media inside is ordinary fMP4 or WebM; UMP only wraps it, so unwrapping it
is all that's needed to feed it to the same generic assembly as every other
site.

    part    := varint(type) varint(size) payload
    MEDIA_HEADER (20)   protobuf: which format, which segment, its timing
    MEDIA        (21)   varint(header_id) + a slice of that segment's bytes
    MEDIA_END    (22)   varint(header_id): the segment is complete

UMP's varint is not protobuf's: the count of leading 1-bits in the first byte
gives the length, like UTF-8. Pure functions, no I/O.
"""
import gzip
import logging
from dataclasses import dataclass

log = logging.getLogger(__name__)

MEDIA_HEADER, MEDIA, MEDIA_END = 20, 21, 22
_GZIP = 2  # MediaHeader.compression_algorithm


def read_varint(data: bytes, pos: int) -> tuple[int, int] | None:
    """UMP varint at `pos` -> (value, next position), or None if truncated."""
    if pos >= len(data):
        return None
    first = data[pos]
    if first < 0x80:
        length = 1
    elif first < 0xC0:
        length = 2
    elif first < 0xE0:
        length = 3
    elif first < 0xF0:
        length = 4
    else:
        length = 5
    if pos + length > len(data):
        return None
    b = data[pos : pos + length]
    if length == 1:
        value = b[0]
    elif length == 2:
        value = (b[0] & 0x3F) + 64 * b[1]
    elif length == 3:
        value = (b[0] & 0x1F) + 32 * (b[1] + 256 * b[2])
    elif length == 4:
        value = (b[0] & 0x0F) + 16 * (b[1] + 256 * (b[2] + 256 * b[3]))
    else:
        value = int.from_bytes(b[1:5], "little")
    return value, pos + length


def iter_parts(data: bytes):
    """Yield (type, payload) for each complete part; stop at a truncated one."""
    pos = 0
    while pos < len(data):
        got = read_varint(data, pos)
        if got is None:
            return
        part_type, pos = got
        got = read_varint(data, pos)
        if got is None:
            return
        size, pos = got
        if pos + size > len(data):
            return
        yield part_type, data[pos : pos + size]
        pos += size


# --------------------------------------------------------------------------
# Minimal protobuf
# --------------------------------------------------------------------------

def _pb_varint(data: bytes, pos: int) -> tuple[int, int]:
    value, shift = 0, 0
    while True:
        if pos >= len(data):
            raise ValueError("truncated varint")
        byte = data[pos]
        pos += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return value, pos
        shift += 7
        if shift > 70:
            raise ValueError("varint too long")


def decode_protobuf(data: bytes) -> dict[int, list]:
    """Field number -> list of raw values (ints for varints, bytes otherwise)."""
    fields: dict[int, list] = {}
    pos = 0
    while pos < len(data):
        key, pos = _pb_varint(data, pos)
        number, wire = key >> 3, key & 7
        if wire == 0:
            value, pos = _pb_varint(data, pos)
        elif wire == 1:
            value, pos = int.from_bytes(data[pos : pos + 8], "little"), pos + 8
        elif wire == 2:
            length, pos = _pb_varint(data, pos)
            value, pos = data[pos : pos + length], pos + length
        elif wire == 5:
            value, pos = int.from_bytes(data[pos : pos + 4], "little"), pos + 4
        else:
            raise ValueError(f"unsupported wire type {wire}")
        if pos > len(data):
            raise ValueError("truncated field")
        fields.setdefault(number, []).append(value)
    return fields


# --------------------------------------------------------------------------
# Media
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class MediaHeader:
    header_id: int
    video_id: str
    itag: int
    xtags: str
    is_init: bool
    sequence: int
    start_ms: int
    duration_ms: int
    content_length: int
    compressed: bool


def parse_media_header(payload: bytes) -> MediaHeader | None:
    try:
        f = decode_protobuf(payload)
    except ValueError:
        return None

    def num(n, default=0):
        v = f.get(n)
        return v[0] if v and isinstance(v[0], int) else default

    def text(n):
        v = f.get(n)
        return v[0].decode("utf-8", "replace") if v and isinstance(v[0], bytes) else ""

    itag = num(3)
    if not itag and f.get(13):  # format_id { itag = 1 }
        try:
            itag = decode_protobuf(f[13][0]).get(1, [0])[0]
        except ValueError:
            pass
    return MediaHeader(
        header_id=num(1), video_id=text(2), itag=itag, xtags=text(5),
        is_init=bool(num(8)), sequence=num(9), start_ms=num(11), duration_ms=num(12),
        content_length=num(14), compressed=num(7) == _GZIP,
    )


@dataclass(frozen=True)
class UmpChunk:
    """One complete segment (or init segment) of one format."""

    video_id: str
    itag: int
    xtags: str
    is_init: bool
    sequence: int
    start: float      # seconds
    duration: float   # seconds
    data: bytes


def extract_chunks(body: bytes) -> list[UmpChunk]:
    """Every complete segment in a UMP response body.

    A segment's bytes may be spread over several MEDIA parts; they are joined
    by header id. A segment whose MEDIA_END never arrived is dropped -- it's
    incomplete, and a later response will carry it whole.
    """
    headers: dict[int, MediaHeader] = {}
    buffers: dict[int, bytearray] = {}
    chunks: list[UmpChunk] = []

    for part_type, payload in iter_parts(body):
        if part_type == MEDIA_HEADER:
            header = parse_media_header(payload)
            if header is not None:
                headers[header.header_id] = header
                buffers[header.header_id] = bytearray()
        elif part_type == MEDIA:
            got = read_varint(payload, 0)
            if got is None:
                continue
            header_id, start = got
            if header_id in buffers:
                buffers[header_id] += payload[start:]
        elif part_type == MEDIA_END:
            got = read_varint(payload, 0)
            if got is None:
                continue
            header_id = got[0]
            header = headers.pop(header_id, None)
            data = buffers.pop(header_id, None)
            if header is None or data is None:
                continue
            raw = bytes(data)
            if header.compressed:
                try:
                    raw = gzip.decompress(raw)
                except OSError:
                    log.debug("Undecompressable UMP segment for itag %s", header.itag)
                    continue
            # Whether content_length counts compressed bytes isn't documented;
            # accept either, reject anything else as a torn segment.
            if header.content_length and header.content_length not in (len(raw), len(data)):
                log.debug("UMP segment length mismatch (%d != %d); dropping",
                          len(raw), header.content_length)
                continue
            chunks.append(UmpChunk(
                video_id=header.video_id, itag=header.itag, xtags=header.xtags,
                is_init=header.is_init, sequence=header.sequence,
                start=header.start_ms / 1000, duration=header.duration_ms / 1000,
                data=raw,
            ))
    return chunks
