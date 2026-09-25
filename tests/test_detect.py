"""Recognising media from bytes: containers, manifests, DRM, page identity.

Everything is built synthetically -- MP4 boxes and EBML elements by hand -- so
the tests say exactly which structure each rule keys on.
"""
import struct

import pytest

from core.sniffer import detect as D


def box(kind: bytes, payload: bytes = b"") -> bytes:
    return struct.pack(">I4s", 8 + len(payload), kind) + payload


def full(kind: bytes, payload: bytes, version: int = 0) -> bytes:
    return box(kind, bytes([version, 0, 0, 0]) + payload)


def hdlr(handler: bytes) -> bytes:
    return full(b"hdlr", b"\0\0\0\0" + handler + b"\0" * 12)


def stsd(*entries: bytes) -> bytes:
    return full(b"stsd", struct.pack(">I", len(entries)) + b"".join(box(e, b"\0" * 8) for e in entries))


def trak(handler: bytes, entry: bytes) -> bytes:
    mdhd = full(b"mdhd", b"\0" * 8 + struct.pack(">II", 48000, 0) + b"\0" * 4)
    return box(b"trak", box(b"mdia", mdhd + hdlr(handler) +
                            box(b"minf", box(b"stbl", stsd(entry)))))


def init_segment(*traks: bytes, extra: bytes = b"", duration_ms: int = 216_000) -> bytes:
    mvhd = full(b"mvhd", b"\0" * 8 + struct.pack(">II", 1000, duration_ms) + b"\0" * 80)
    return box(b"ftyp", b"iso6\0\0\0\0") + box(b"moov", mvhd + b"".join(traks) + extra)


def media_segment(sequence: int, decode_time: int, *, encrypted=False) -> bytes:
    traf = full(b"tfdt", struct.pack(">I", decode_time))
    if encrypted:
        traf += full(b"senc", b"\0\0\0\0")
    return box(b"moof", full(b"mfhd", struct.pack(">I", sequence)) + box(b"traf", traf)) + box(b"mdat", b"\0" * 32)


class TestCandidates:
    def test_media_mime(self):
        assert D.is_candidate("https://x/y", "video/mp4", "Media")
        assert D.is_candidate("https://x/y", "audio/webm; codecs=opus", "XHR")

    def test_manifests_by_mime(self):
        assert D.is_candidate("https://x/y", "application/vnd.apple.mpegurl", "XHR")
        assert D.is_candidate("https://x/y", "application/dash+xml", "Fetch")

    def test_by_extension(self):
        assert D.is_candidate("https://x/seg-12.m4s?token=1", "text/plain", "XHR")

    def test_opaque_xhr_is_sniffed(self):
        assert D.is_candidate("https://x/chunk", "application/octet-stream", "XHR")

    def test_documents_and_scripts_never(self):
        assert not D.is_candidate("https://x/a.mp4", "video/mp4", "Document")
        assert not D.is_candidate("https://x/a.js", "application/javascript", "Script")
        assert not D.is_candidate("https://x/api", "application/json", "XHR")


class TestSniffContainer:
    @pytest.mark.parametrize("body,expected", [
        (b"#EXTM3U\n#EXT-X-VERSION:3", D.HLS),
        (b'<?xml version="1.0"?><MPD xmlns="urn:mpeg:dash:schema:mpd:2011">', D.DASH),
        (box(b"ftyp", b"isom"), D.MP4),
        (box(b"styp", b"msdh"), D.MP4),
        (box(b"moof"), D.MP4),
        (b"\x1a\x45\xdf\xa3" + b"\0" * 20, D.WEBM),
        (b"\x1f\x43\xb6\x75" + b"\0" * 20, D.WEBM),
        (b"G" + b"\0" * 187 + b"G" + b"\0" * 187, D.MPEGTS),
        (b"ID3\x04\0\0", D.MP3),
        (b"\xff\xf1\x50\x80", D.ADTS),
        (b"<html>", D.UNKNOWN),
        (b"", D.UNKNOWN),
    ])
    def test_magic(self, body, expected):
        assert D.sniff_container(body) == expected

    def test_ump_is_known_by_mime(self):
        assert D.sniff_container(b"\x14\x10", "application/vnd.yt-ump") == D.UMP


class TestMp4:
    def test_init_segment_kind_codecs_duration(self):
        data = init_segment(trak(b"vide", b"avc1"), trak(b"soun", b"mp4a"))
        facts = D.media_facts(data)
        assert facts.kind == D.MUXED
        assert D.mp4_info(data).codecs == ["h264", "aac"]
        assert facts.is_init and not facts.has_media
        assert facts.duration == pytest.approx(216.0)
        assert not facts.encrypted

    def test_audio_only(self):
        facts = D.media_facts(init_segment(trak(b"soun", b"Opus")))
        assert facts.kind == D.AUDIO

    def test_encv_sample_entry_is_drm(self):
        facts = D.media_facts(init_segment(trak(b"vide", b"encv")))
        assert facts.encrypted
        assert facts.kind == D.VIDEO  # still says what it carries

    def test_pssh_is_drm(self):
        facts = D.media_facts(init_segment(trak(b"soun", b"mp4a"), extra=full(b"pssh", b"\0" * 20)))
        assert facts.encrypted

    def test_media_segment(self):
        info = D.mp4_info(media_segment(7, 96000))
        assert info.has_media and not info.has_init

    def test_encrypted_media_segment(self):
        assert D.mp4_info(media_segment(1, 0, encrypted=True)).encrypted

    def test_truncated_box_does_not_raise(self):
        data = init_segment(trak(b"vide", b"avc1"))
        D.mp4_info(data[: len(data) // 2])


def ebml(el_id: int, payload: bytes) -> bytes:
    """One EBML element: its ID, a 1- or 2-byte size vint, the payload."""
    id_bytes = el_id.to_bytes((el_id.bit_length() + 7) // 8, "big")
    if len(payload) < 127:
        size = bytes([0x80 | len(payload)])
    else:
        size = (0x4000 | len(payload)).to_bytes(2, "big")
    return id_bytes + size + payload


def webm(*tracks: tuple[int, str], encrypted=False, cluster=False) -> bytes:
    entries = b""
    for track_type, codec in tracks:
        body = ebml(0x83, bytes([track_type])) + ebml(0x86, codec.encode())
        if encrypted:
            body += ebml(0x6D80, ebml(0x6240, ebml(0x5035, b"\x47\x81\x05")))
        entries += ebml(0xAE, body)
    segment = ebml(0x1654AE6B, entries)
    if cluster:
        segment += ebml(0x1F43B675, b"\xe7\x81\x00")
    return ebml(0x1A45DFA3, ebml(0x4282, b"webm")) + b"\x18\x53\x80\x67\x01\xff\xff\xff\xff\xff\xff\xff" + segment


class TestWebm:
    def test_tracks(self):
        facts = D.media_facts(webm((1, "V_VP9"), (2, "A_OPUS")))
        assert facts.kind == D.MUXED
        assert facts.is_init and not facts.has_media

    def test_audio_only(self):
        assert D.media_facts(webm((2, "A_OPUS"))).kind == D.AUDIO

    def test_content_encryption_is_drm(self):
        assert D.media_facts(webm((1, "V_VP9"), encrypted=True)).encrypted

    def test_cluster_means_media(self):
        assert D.media_facts(webm((2, "A_OPUS"), cluster=True)).has_media


HLS_MASTER = """#EXTM3U
#EXT-X-STREAM-INF:BANDWIDTH=800000,RESOLUTION=640x360,CODECS="avc1.4d401e,mp4a.40.2"
low/index.m3u8
#EXT-X-STREAM-INF:BANDWIDTH=128000,CODECS="mp4a.40.2"
audio/index.m3u8
"""

HLS_MEDIA = """#EXTM3U
#EXT-X-MEDIA-SEQUENCE:10
#EXT-X-MAP:URI="init.mp4"
#EXTINF:6.0,
seg10.m4s
#EXTINF:6.0,
seg11.m4s
#EXTINF:3.5,
https://cdn.other/seg12.m4s
#EXT-X-ENDLIST
"""


class TestHls:
    def test_master(self):
        p = D.parse_hls(HLS_MASTER, "https://x.com/v/master.m3u8")
        assert p.is_master
        assert p.variants[0].uri == "https://x.com/v/low/index.m3u8"
        assert D.kind_from_codecs(p.variants[0].codecs) == D.MUXED
        assert D.kind_from_codecs(p.variants[1].codecs) == D.AUDIO

    def test_media(self):
        p = D.parse_hls(HLS_MEDIA, "https://x.com/v/low/index.m3u8")
        assert [s.sequence for s in p.segments] == [10, 11, 12]
        assert [s.start for s in p.segments] == [0.0, 6.0, 12.0]
        assert p.segments[0].uri == "https://x.com/v/low/seg10.m4s"
        assert p.segments[2].uri == "https://cdn.other/seg12.m4s"
        assert p.init_uri == "https://x.com/v/low/init.mp4"
        assert p.ended and p.duration == pytest.approx(15.5)
        assert not p.encrypted

    def test_byteranges(self):
        text = "#EXTM3U\n#EXTINF:4,\n#EXT-X-BYTERANGE:1000@0\nall.ts\n#EXTINF:4,\n#EXT-X-BYTERANGE:500\nall.ts\n"
        p = D.parse_hls(text, "https://x/a.m3u8")
        assert [s.byterange for s in p.segments] == [(0, 1000), (1000, 500)]

    @pytest.mark.parametrize("line", [
        '#EXT-X-KEY:METHOD=AES-128,URI="key.bin"',
        '#EXT-X-KEY:METHOD=SAMPLE-AES,URI="skd://x",KEYFORMAT="com.apple.streamingkeydelivery"',
    ])
    def test_encryption(self, line):
        p = D.parse_hls(HLS_MEDIA.replace("#EXT-X-MAP", line + "\n#EXT-X-MAP"), "https://x/a.m3u8")
        assert p.encrypted

    def test_method_none_is_clear(self):
        p = D.parse_hls(HLS_MEDIA.replace("#EXT-X-MAP", "#EXT-X-KEY:METHOD=NONE\n#EXT-X-MAP"), "https://x/a.m3u8")
        assert not p.encrypted


DASH_TEMPLATE = """<?xml version="1.0"?>
<MPD xmlns="urn:mpeg:dash:schema:mpd:2011" mediaPresentationDuration="PT1M0.0S">
 <Period>
  <AdaptationSet contentType="video" mimeType="video/mp4">
   <SegmentTemplate timescale="1000" duration="4000" startNumber="1"
      initialization="$RepresentationID$/init.mp4" media="$RepresentationID$/seg-$Number%05d$.m4s"/>
   <Representation id="v720" bandwidth="3000000" codecs="avc1.64001f"/>
   <Representation id="v360" bandwidth="800000" codecs="avc1.4d401e"/>
  </AdaptationSet>
  <AdaptationSet contentType="audio" mimeType="audio/mp4">
   <ContentProtection schemeIdUri="urn:uuid:edef8ba9-79d6-4ace-a3c8-27dcd51d21ed"/>
   <Representation id="a" bandwidth="128000" codecs="mp4a.40.2">
    <SegmentTemplate timescale="48000" initialization="a/init.mp4" media="a/$Time$.m4s">
     <SegmentTimeline><S t="0" d="192000" r="2"/><S d="96000"/></SegmentTimeline>
    </SegmentTemplate>
   </Representation>
  </AdaptationSet>
 </Period>
</MPD>"""


class TestDash:
    def test_number_template(self):
        m = D.parse_dash(DASH_TEMPLATE, "https://x.com/v/manifest.mpd")
        v720 = m.representations[0]
        assert m.duration == 60.0
        assert v720.kind == D.VIDEO
        assert v720.init_url == "https://x.com/v/v720/init.mp4"
        assert len(v720.segments) == 15
        assert v720.segments["https://x.com/v/v720/seg-00001.m4s"] == (1, 0.0, 4.0)
        assert "https://x.com/v/v720/seg-00015.m4s" in v720.segments
        assert not v720.encrypted

    def test_timeline_and_protection(self):
        audio = D.parse_dash(DASH_TEMPLATE, "https://x.com/v/manifest.mpd").representations[2]
        assert audio.kind == D.AUDIO and audio.encrypted
        assert audio.segments["https://x.com/v/a/384000.m4s"] == (3, 8.0, 4.0)
        assert audio.segments["https://x.com/v/a/576000.m4s"] == (4, 12.0, 2.0)

    def test_iso_durations(self):
        assert D.parse_iso_duration("PT1H2M3.5S") == pytest.approx(3723.5)
        assert D.parse_iso_duration("P1DT1S") == 86401
        assert D.parse_iso_duration("") == 0.0


class TestRangesAndIdentity:
    def test_content_range(self):
        assert D.content_range("bytes 100-199/1000") == (100, 199, 1000)
        assert D.content_range("bytes 0-99/*") == (0, 99, None)
        assert D.content_range(None) is None

    def test_query_range(self):
        assert D.query_range("https://r1.googlevideo.com/videoplayback?range=0-1023&x=1") == (0, 1023)
        assert D.query_range("https://x/a.mp4") is None

    def test_googlevideo_identity_ignores_everything_but_id_and_itag(self):
        a = "https://rr1.googlevideo.com/videoplayback?id=o-ABC&itag=251&range=0-99&sig=1&rn=1"
        b = "https://rr5.googlevideo.com/videoplayback?itag=251&id=o-ABC&range=100-199&sig=2&rn=2"
        assert D.stream_url_key(a) == D.stream_url_key(b) == "gv:o-ABC:251"

    def test_generic_identity_drops_volatile_params(self):
        a = "https://cdn.x.com/v/file.mp4?token=abc&quality=hd&_=123"
        b = "https://cdn.x.com/v/file.mp4?quality=hd&token=xyz&_=456"
        assert D.stream_url_key(a) == D.stream_url_key(b)
        assert D.stream_url_key(a) != D.stream_url_key("https://cdn.x.com/v/file.mp4?quality=sd")

    @pytest.mark.parametrize("url", [
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=42s&list=PL1",
        "https://youtu.be/dQw4w9WgXcQ?si=abc",
        "https://www.youtube.com/shorts/dQw4w9WgXcQ",
        "https://music.youtube.com/watch?v=dQw4w9WgXcQ&feature=share",
        "https://www.youtube-nocookie.com/embed/dQw4w9WgXcQ",
    ])
    def test_youtube_pages_collapse(self, url):
        assert D.page_id(url) == "yt:dQw4w9WgXcQ"

    def test_other_pages_keep_meaningful_query(self):
        assert D.page_id("https://www.vimeo.com/123?utm_source=x") == "//vimeo.com/123"
        assert D.page_id("https://site.tv/watch?id=9&fbclid=z") == "//site.tv/watch?id=9"

    def test_site_name(self):
        assert D.site_name("https://www.youtube.com/watch?v=x") == "youtube.com"


class TestWhichDrm:
    """Which DRM protects a stream, from whatever names it: detection only."""

    WV, PR = bytes.fromhex("edef8ba979d64acea3c827dcd51d21ed"), bytes.fromhex("9a04f07998404286ab92e65be0885f95")

    def test_pssh_names_the_systems(self):
        extra = full(b"pssh", self.WV + b"\0" * 4) + full(b"pssh", self.PR + b"\0" * 4)
        facts = D.media_facts(init_segment(trak(b"vide", b"encv"), extra=extra))
        assert facts.encrypted and facts.drm == ("Widevine", "PlayReady")

    def test_an_unknown_system_is_shown_by_id(self):
        facts = D.media_facts(init_segment(trak(b"vide", b"encv"),
                                           extra=full(b"pssh", b"\x11" * 16)))
        assert facts.drm == ("unknown DRM " + "11" * 16,)

    def test_scheme_from_the_encrypted_sample_entry(self):
        """encv -> sinf -> schm, after the video sample entry's 78 bytes."""
        sinf = box(b"sinf", box(b"frma", b"avc1") + full(b"schm", b"cbcs" + b"\0\0\0\1"))
        entry = box(b"encv", b"\0" * 78 + sinf)
        stsd_box = full(b"stsd", struct.pack(">I", 1) + entry)
        moov = box(b"moov", box(b"trak", box(b"mdia", box(b"minf", box(b"stbl", stsd_box)))))
        info = D.mp4_info(moov)
        assert info.encrypted and info.drm == ["scheme cbcs"]

    def test_webm_names_only_the_cipher(self):
        facts = D.media_facts(webm((1, "V_VP9"), encrypted=True))
        assert facts.drm == ("Matroska AES encryption",)

    @pytest.mark.parametrize("key_system,name", [
        ("com.widevine.alpha", "Widevine"),
        ("com.microsoft.playready.recommendation", "PlayReady"),
        ("com.apple.fps.1_0", "FairPlay"),
        ("org.w3.clearkey", "ClearKey"),
        ("com.example.drm", "com.example.drm"),
        ("", None),
    ])
    def test_eme_key_systems(self, key_system, name):
        assert D.key_system_name(key_system) == name

    @pytest.mark.parametrize("key_line,label", [
        ('#EXT-X-KEY:METHOD=SAMPLE-AES,URI="skd://k",KEYFORMAT="com.apple.streamingkeydelivery"',
         "FairPlay (SAMPLE-AES)"),
        ('#EXT-X-KEY:METHOD=SAMPLE-AES-CTR,URI="data:x",'
         'KEYFORMAT="urn:uuid:edef8ba9-79d6-4ace-a3c8-27dcd51d21ed"', "Widevine (SAMPLE-AES-CTR)"),
        ('#EXT-X-KEY:METHOD=AES-128,URI="https://x/key"', "HLS AES-128"),
    ])
    def test_hls_keys(self, key_line, label):
        p = D.parse_hls(HLS_MEDIA.replace("#EXT-X-MAP", key_line + "\n#EXT-X-MAP"),
                        "https://x/a.m3u8")
        assert p.encrypted and p.drm == [label]

    def test_dash_content_protection(self):
        mpd = DASH_TEMPLATE.replace(
            '<AdaptationSet contentType="video" mimeType="video/mp4">',
            '<AdaptationSet contentType="video" mimeType="video/mp4">'
            '<ContentProtection schemeIdUri="urn:mpeg:dash:mp4protection:2011" value="cenc"/>'
            '<ContentProtection schemeIdUri="urn:uuid:EDEF8BA9-79D6-4ACE-A3C8-27DCD51D21ED"/>')
        video = D.parse_dash(mpd, "https://x.com/v/manifest.mpd").representations[0]
        assert video.encrypted and video.drm == ["scheme cenc", "Widevine"]


class TestSecurityLevel:
    """The level comes from the robustness the browser granted, not the stream."""

    @pytest.mark.parametrize("video,label", [
        (["HW_SECURE_ALL"], "Widevine L1 (HW_SECURE_ALL)"),
        (["HW_SECURE_DECODE", "SW_SECURE_CRYPTO"], "Widevine L1 (HW_SECURE_DECODE)"),
        (["HW_SECURE_CRYPTO"], "Widevine L2 (HW_SECURE_CRYPTO)"),
        (["SW_SECURE_DECODE"], "Widevine L3 (SW_SECURE_DECODE)"),
        (["sw_secure_crypto"], "Widevine L3 (SW_SECURE_CRYPTO)"),
        ([""], "Widevine L3 (no robustness requested)"),
        ([], "Widevine L3 (no robustness requested)"),
    ])
    def test_widevine(self, video, label):
        assert D.eme_drm_label("com.widevine.alpha", video, []) == label

    def test_audio_only_quotes_the_audio(self):
        assert D.eme_drm_label("com.widevine.alpha", [], ["HW_SECURE_CRYPTO"]) == (
            "Widevine L2 (HW_SECURE_CRYPTO)")

    @pytest.mark.parametrize("key_system,video,label", [
        ("com.microsoft.playready.recommendation.3000", [], "PlayReady SL3000"),
        ("com.microsoft.playready.recommendation", ["2000"], "PlayReady SL2000"),
        ("com.microsoft.playready.hardware", [], "PlayReady SL3000 (hardware)"),
        ("com.microsoft.playready", [], "PlayReady"),
        ("com.apple.fps", [], "FairPlay"),
    ])
    def test_other_systems(self, key_system, video, label):
        assert D.eme_drm_label(key_system, video, []) == label



class TestPlaybackSecurity:
    """The playback's own level, from its player's media-log properties."""

    def test_media_foundation_is_hardware_secure(self):
        assert D.playback_security({"kRendererName": "MediaFoundationRenderer"}) == (
            "hardware", "MediaFoundationRenderer, the hardware-secure path")

    def test_decrypting_demuxer_stream_is_the_software_cdm(self):
        props = {"kRendererName": "RendererImpl", "kIsAudioDecryptingDemuxerStream": "true",
                 "kAudioDecoderName": "FFmpegAudioDecoder"}
        assert D.playback_security(props) == (
            "software", "audio decrypted by the software CDM (DecryptingDemuxerStream)")

    def test_decrypting_decoder_is_the_software_cdm(self):
        props = {"kRendererName": "RendererImpl", "kVideoDecoderName": '"DecryptingVideoDecoder"'}
        assert D.playback_security(props) == (
            "software", "video decrypted and decoded by the software CDM (DecryptingVideoDecoder)")

    def test_clear_playback_says_nothing(self):
        props = {"kRendererName": "RendererImpl", "kIsVideoDecryptingDemuxerStream": "false",
                 "kVideoDecoderName": "D3D11VideoDecoder"}
        assert D.playback_security(props) is None

    def test_level_names(self):
        assert D.level_name("Widevine", "software") == "L3"
        assert D.level_name("Widevine", "hardware") == "L1"
        assert D.level_name("PlayReady", "hardware") == "SL3000"
        assert D.level_name("scheme cenc", "software") is None
