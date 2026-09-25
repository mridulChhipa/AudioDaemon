"""YouTube's UMP framing: varints, parts, and segments split across parts."""
import gzip

import pytest

from core.sniffer import ump as U


def encode_varint(value: int) -> bytes:
    """The inverse of U.read_varint, to build UMP streams."""
    if value < 0x80:
        return bytes([value])
    if value < 0x4000:
        return bytes([0x80 | (value & 0x3F), value >> 6])
    if value < 0x200000:
        return bytes([0xC0 | (value & 0x1F), (value >> 5) & 0xFF, value >> 13])
    if value < 0x10000000:
        return bytes([0xE0 | (value & 0x0F), (value >> 4) & 0xFF, (value >> 12) & 0xFF,
                      value >> 20])
    return bytes([0xF0]) + value.to_bytes(4, "little")


def encode_protobuf(fields: dict[int, object]) -> bytes:
    """A protobuf message from {field: int | str | bytes}."""
    def varint(v: int) -> bytes:
        out = bytearray()
        while True:
            byte = v & 0x7F
            v >>= 7
            out.append(byte | (0x80 if v else 0))
            if not v:
                return bytes(out)

    out = bytearray()
    for number, value in fields.items():
        if isinstance(value, int):  # bools included
            out += varint(number << 3) + varint(int(value))
        else:
            raw = value.encode() if isinstance(value, str) else bytes(value)
            out += varint((number << 3) | 2) + varint(len(raw)) + raw
    return bytes(out)


def part(part_type: int, payload: bytes) -> bytes:
    return encode_varint(part_type) + encode_varint(len(payload)) + payload


def header(header_id, *, video_id="abcdefghijk", itag=251, seq=1, start_ms=0,
           duration_ms=5000, is_init=False, length=None, compressed=False) -> bytes:
    fields = {1: header_id, 2: video_id, 3: itag, 5: "", 8: is_init, 9: seq,
              11: start_ms, 12: duration_ms}
    if length is not None:
        fields[14] = length
    if compressed:
        fields[7] = 2
    return part(U.MEDIA_HEADER, encode_protobuf(fields))


def media(header_id, data: bytes) -> bytes:
    return part(U.MEDIA, encode_varint(header_id) + data)


def end(header_id) -> bytes:
    return part(U.MEDIA_END, encode_varint(header_id))


class TestVarint:
    @pytest.mark.parametrize("value", [0, 1, 127, 128, 16383, 16384, 2_097_151,
                                       2_097_152, 268_435_455, 268_435_456, 2**32 - 1])
    def test_round_trip(self, value):
        encoded = encode_varint(value)
        assert U.read_varint(encoded, 0) == (value, len(encoded))

    def test_lengths_follow_the_leading_bits(self):
        assert len(encode_varint(100)) == 1
        assert len(encode_varint(1000)) == 2
        assert len(encode_varint(10**9)) == 5

    def test_truncated(self):
        assert U.read_varint(encode_varint(1000)[:1], 0) is None


class TestProtobuf:
    def test_round_trip(self):
        f = U.decode_protobuf(encode_protobuf({1: 7, 2: "vid", 9: 300}))
        assert f[1] == [7] and f[2] == [b"vid"] and f[9] == [300]

    def test_header_fields(self):
        h = U.parse_media_header(encode_protobuf(
            {1: 3, 2: "abcdefghijk", 3: 399, 8: True, 9: 12, 11: 60000, 12: 5005, 14: 999}))
        assert (h.header_id, h.video_id, h.itag, h.is_init, h.sequence) == (3, "abcdefghijk", 399, True, 12)
        assert (h.start_ms, h.duration_ms, h.content_length) == (60000, 5005, 999)

    def test_itag_from_format_id(self):
        h = U.parse_media_header(encode_protobuf({1: 0, 13: encode_protobuf({1: 251})}))
        assert h.itag == 251


class TestExtract:
    def test_segment_split_across_media_parts(self):
        body = header(0, seq=4, start_ms=15000, length=10) + media(0, b"01234") + media(0, b"56789") + end(0)
        (chunk,) = U.extract_chunks(body)
        assert chunk.data == b"0123456789"
        assert (chunk.itag, chunk.sequence, chunk.start, chunk.duration) == (251, 4, 15.0, 5.0)
        assert chunk.video_id == "abcdefghijk"

    def test_interleaved_formats(self):
        body = (header(0, itag=251, seq=1) + header(1, itag=399, seq=1)
                + media(1, b"video") + media(0, b"audio") + end(0) + end(1))
        chunks = {c.itag: c.data for c in U.extract_chunks(body)}
        assert chunks == {251: b"audio", 399: b"video"}

    def test_init_segment_flag(self):
        (chunk,) = U.extract_chunks(header(0, is_init=True) + media(0, b"init") + end(0))
        assert chunk.is_init

    def test_unfinished_segment_is_dropped(self):
        assert U.extract_chunks(header(0) + media(0, b"half")) == []

    def test_length_mismatch_is_dropped(self):
        assert U.extract_chunks(header(0, length=100) + media(0, b"short") + end(0)) == []

    def test_other_parts_are_ignored(self):
        body = part(35, b"policy") + header(0) + media(0, b"x") + end(0) + part(58, b"\x08\x01")
        assert len(U.extract_chunks(body)) == 1

    def test_truncated_body(self):
        body = header(0) + media(0, b"abc") + end(0)
        assert U.extract_chunks(body + header(1)[:3]) != []

    def test_gzip(self):
        raw = b"compressed segment" * 10
        packed = gzip.compress(raw)
        body = header(0, compressed=True, length=len(packed)) + media(0, packed) + end(0)
        (chunk,) = U.extract_chunks(body)
        assert chunk.data == raw
