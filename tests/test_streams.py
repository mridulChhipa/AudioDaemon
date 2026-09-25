"""The stream store: placing pieces, completeness, renditions, assembly, DRM,
and planning what to fetch to complete a stream."""
import json
import os
import time

import pytest

from core.sniffer import detect, streams
from core.sniffer.streams import (
    IGNORED,
    MANIFEST,
    STORED,
    UNPLACED,
    FetchRequest,
    Response,
    StreamStore,
    coverage_gaps,
    digest,
    duration_matches,
    merge_intervals,
)
from tests.test_detect import init_segment, media_segment, trak
from tests.test_ump import end, header, media

PAGE = "https://www.example.tv/watch/42"
ITEM = detect.page_id(PAGE)


@pytest.fixture
def store(tmp_path):
    return StreamStore(tmp_path / "streams")


def resp(url, body, *, mime="video/mp4", status=200, headers=None, page=PAGE):
    return Response(page, url, status, mime, headers or {}, body)


def ranged(url, body, first, total, mime="video/mp4", page=PAGE):
    last = first + len(body) - 1
    return resp(url, body, mime=mime, status=206, page=page,
                headers={"content-range": f"bytes {first}-{last}/{total if total else '*'}"})


class TestMergeIntervals:
    def test_empty(self):
        assert merge_intervals([]) == []

    def test_disjoint_are_kept_apart(self):
        assert merge_intervals([(0, 10), (20, 30)]) == [(0, 10), (20, 30)]

    def test_overlapping_are_merged(self):
        assert merge_intervals([(0, 12), (10, 30)]) == [(0, 30)]

    def test_unordered_input(self):
        assert merge_intervals([(20, 30), (0, 25)]) == [(0, 30)]

    def test_contained_interval_does_not_shrink_the_union(self):
        assert merge_intervals([(0, 30), (5, 10)]) == [(0, 30)]

    def test_tolerance_bridges_a_small_hole(self):
        assert merge_intervals([(0, 10), (10.5, 20)], tolerance=1.0) == [(0, 20)]


class TestCoverageGaps:
    def test_full_coverage(self):
        assert coverage_gaps([(0, 200)], 200) == []

    def test_missing_tail(self):
        assert coverage_gaps([(0, 100)], 200) == [(100, 200)]

    def test_missing_head(self):
        assert coverage_gaps([(40, 200)], 200) == [(0, 40)]

    def test_hole_in_the_middle(self):
        assert coverage_gaps([(0, 80), (120, 200)], 200) == [(80, 120)]

    def test_small_shortfall_is_within_tolerance(self):
        # Segment timings are rounded; the reported length with them.
        assert coverage_gaps([(0.4, 199.7)], 200, tolerance=1.0) == []

    def test_zero_duration_has_no_gaps(self):
        assert coverage_gaps([(0, 10)], 0) == []


class TestDurationMatches:
    def test_close_enough(self):
        assert duration_matches(215.8, 216.9)

    def test_an_ad_is_not_the_video(self):
        assert not duration_matches(15.0, 216.0)

    def test_unknown_lengths_pass(self):
        assert duration_matches(0.0, 216.0)
        assert duration_matches(15.0, 0.0)

    def test_loose_allows_a_padded_stream(self):
        assert not duration_matches(208.0, 216.0)
        assert duration_matches(208.0, 216.0, loose=True)
        assert not duration_matches(15.0, 216.0, loose=True)


class TestByteRanges:
    def test_out_of_order_ranges_complete_and_assemble(self, store, tmp_path):
        data = init_segment(trak(b"vide", b"avc1")) + os.urandom(5000)
        a, b, c = data[:2000], data[2000:4000], data[4000:]
        url = "https://cdn.example.tv/v/file.mp4?token=1"
        assert store.ingest(ranged(url, c, 4000, len(data))) == STORED
        assert store.ingest(ranged(url.replace("token=1", "token=2"), a, 0, len(data))) == STORED
        assert store.candidates(ITEM) == []  # a hole at 2000-4000
        store.ingest(ranged(url, b, 2000, len(data)))

        (cand,) = store.candidates(ITEM)
        out = tmp_path / "out.mp4"
        assert store.assemble(ITEM, cand.key, out)
        assert out.read_bytes() == data

    def test_overlapping_ranges_assemble_once(self, store, tmp_path):
        data = os.urandom(3000)
        url = "https://cdn.example.tv/a.mp4"
        store.ingest(ranged(url, data[:2000], 0, 3000))
        store.ingest(ranged(url, data[1500:], 1500, 3000))
        (cand,) = store.candidates(ITEM)
        out = tmp_path / "o"
        store.assemble(ITEM, cand.key, out)
        assert out.read_bytes() == data

    def test_mid_file_range_is_not_sniffed(self, store):
        """A range starting with 0x47 is not an MPEG-TS file."""
        url = "https://cdn.example.tv/a.mp4"
        store.ingest(ranged(url, b"G" + os.urandom(999), 5000, 10000))
        (state,) = store.streams(ITEM)
        assert state.kind == detect.VIDEO  # from the MIME, not the bytes

    def test_googlevideo_query_ranges(self, store):
        base = "https://rr3.googlevideo.com/videoplayback?id=o-X&itag=140&clen=3000&range="
        data = os.urandom(3000)
        store.ingest(resp(base + "0-1499", data[:1500], mime="audio/mp4"))
        store.ingest(resp(base + "1500-2999&rn=2", data[1500:], mime="audio/mp4"))
        (cand,) = store.candidates(ITEM)
        assert cand.key == "gv:o-X:140" and cand.kind == detect.AUDIO

    def test_whole_file_without_range(self, store):
        data = init_segment(trak(b"soun", b"mp4a")) + b"\0\0\0\x10mdat" + os.urandom(8)
        assert store.ingest(resp("https://x/a.m4a", data, mime="audio/mp4")) == STORED
        assert len(store.candidates(ITEM)) == 1

    def test_init_segment_alone_is_not_a_whole_file(self, store):
        """ftyp+moov with no media: held until its manifest explains it."""
        assert store.ingest(resp("https://x/init.mp4", init_segment(trak(b"vide", b"avc1")))) == UNPLACED
        assert store.candidates(ITEM) == []

    def test_replayed_range_is_stored_once(self, store):
        url = "https://x/a.mp4"
        store.ingest(ranged(url, b"abc", 0, 10))
        store.ingest(ranged(url, b"abc", 0, 10))
        (state,) = store.streams(ITEM)
        assert len(state.chunks) == 1

    def test_non_media_is_ignored(self, store):
        assert store.ingest(resp("https://x/api", b"{}", mime="application/json")) == IGNORED
        assert store.ingest(resp("https://x/a.mp4", b"", status=204)) == IGNORED


HLS_MASTER = """#EXTM3U
#EXT-X-STREAM-INF:BANDWIDTH=900000,CODECS="avc1.4d401e,mp4a.40.2"
hi/index.m3u8
#EXT-X-STREAM-INF:BANDWIDTH=300000,CODECS="avc1.4d400d,mp4a.40.2"
lo/index.m3u8
"""


def hls_media(n=3):
    lines = ["#EXTM3U", "#EXT-X-MEDIA-SEQUENCE:0"]
    for i in range(n):
        lines += ["#EXTINF:4.0,", f"seg{i}.ts"]
    return "\n".join(lines + ["#EXT-X-ENDLIST"])


def ts(tag: bytes) -> bytes:
    return (b"G" + tag + b"\0" * (187 - len(tag))) * 2


class TestHls:
    def test_segments_complete_per_variant(self, store, tmp_path):
        base = "https://cdn.example.tv/v/"
        assert store.ingest(resp(base + "master.m3u8", HLS_MASTER.encode(),
                                 mime="application/vnd.apple.mpegurl")) == MANIFEST
        store.ingest(resp(base + "hi/index.m3u8", hls_media().encode(), mime="audio/mpegurl"))
        store.ingest(resp(base + "lo/index.m3u8", hls_media().encode(), mime="audio/mpegurl"))

        for i in (2, 0, 1):
            store.ingest(resp(base + f"hi/seg{i}.ts", ts(b"hi%d" % i), mime="video/mp2t"))
        store.ingest(resp(base + "lo/seg0.ts", ts(b"lo0"), mime="video/mp2t"))

        (cand,) = store.candidates(ITEM)
        assert cand.kind == detect.MUXED  # from the master's CODECS
        out = tmp_path / "hi.ts"
        store.assemble(ITEM, cand.key, out)
        assert out.read_bytes() == ts(b"hi0") + ts(b"hi1") + ts(b"hi2")

        lo = next(s for s in store.streams(ITEM) if "lo" in s.key)
        assert lo.describe_coverage() == "1/3 segments"

    def test_segments_before_their_manifest_are_held_and_placed(self, store):
        base = "https://cdn.example.tv/v/hi/"
        assert store.ingest(resp(base + "seg0.ts", ts(b"a"), mime="video/mp2t")) == UNPLACED
        store.ingest(resp(base + "index.m3u8", hls_media(1).encode(), mime="audio/mpegurl"))
        assert len(store.candidates(ITEM)) == 1

    def test_signed_segment_urls_still_match(self, store):
        base = "https://cdn.example.tv/v/hi/"
        store.ingest(resp(base + "index.m3u8", hls_media(1).encode(), mime="audio/mpegurl"))
        assert store.ingest(resp(base + "seg0.ts?hdnts=exp=1~hmac=2", ts(b"a"), mime="video/mp2t")) == STORED

    def test_encrypted_playlist_marks_drm_and_keeps_nothing(self, store):
        base = "https://cdn.example.tv/v/hi/"
        text = hls_media(1).replace("#EXT-X-MEDIA-SEQUENCE:0",
                                    '#EXT-X-MEDIA-SEQUENCE:0\n#EXT-X-KEY:METHOD=AES-128,URI="k"')
        store.ingest(resp(base + "index.m3u8", text.encode(), mime="audio/mpegurl"))
        store.ingest(resp(base + "seg0.ts", ts(b"cipher"), mime="video/mp2t"))
        assert store.is_drm(ITEM)
        (state,) = store.streams(ITEM)
        assert state.encrypted and state.chunks == []
        assert store.candidates(ITEM) == []


DASH = """<?xml version="1.0"?>
<MPD xmlns="urn:mpeg:dash:schema:mpd:2011" mediaPresentationDuration="PT8S">
 <Period><AdaptationSet contentType="video">
  <SegmentTemplate timescale="1" duration="4" startNumber="1"
     initialization="$RepresentationID$/init.mp4" media="$RepresentationID$/$Number$.m4s"/>
  <Representation id="v1" bandwidth="1" codecs="avc1.4d401e"/>
 </AdaptationSet></Period>
</MPD>"""


class TestDash:
    def test_init_plus_numbered_segments(self, store, tmp_path):
        base = "https://cdn.example.tv/d/"
        store.ingest(resp(base + "m.mpd", DASH.encode(), mime="application/dash+xml"))
        init = init_segment(trak(b"vide", b"avc1"))
        s1, s2 = media_segment(1, 0), media_segment(2, 4)
        store.ingest(resp(base + "v1/2.m4s", s2))
        assert store.candidates(ITEM) == []
        store.ingest(resp(base + "v1/1.m4s", s1))
        assert store.candidates(ITEM) == []  # no init yet
        store.ingest(resp(base + "v1/init.mp4", init))

        (cand,) = store.candidates(ITEM)
        assert cand.kind == detect.VIDEO
        out = tmp_path / "d.mp4"
        store.assemble(ITEM, cand.key, out)
        assert out.read_bytes() == init + s1 + s2

    def test_encrypted_init_drops_the_stream(self, store):
        base = "https://cdn.example.tv/d/"
        store.ingest(resp(base + "m.mpd", DASH.encode(), mime="application/dash+xml"))
        store.ingest(resp(base + "v1/1.m4s", media_segment(1, 0)))
        store.ingest(resp(base + "v1/init.mp4", init_segment(trak(b"vide", b"encv"))))
        (state,) = store.streams(ITEM)
        assert state.encrypted and state.chunks == [] and state.kind == detect.VIDEO
        summary = store.summary(ITEM)
        assert summary.drm and summary.has_video and summary.clear == 0


class TestUmp:
    def test_itags_become_streams_under_the_video_id(self, store, tmp_path):
        # Fragmented init segments often leave mvhd's duration at zero; then
        # only the page's reported length can say when coverage is complete.
        init = init_segment(trak(b"soun", b"Opus"), duration_ms=0)
        body = (header(0, itag=251, is_init=True) + media(0, init) + end(0)
                + header(1, itag=251, seq=1, start_ms=0, duration_ms=5000)
                + media(1, b"one") + end(1)
                + header(2, itag=251, seq=2, start_ms=5000, duration_ms=5000)
                + media(2, b"two") + end(2))
        store.ingest(resp("https://rr1.googlevideo.com/videoplayback?x", body,
                          mime="application/vnd.yt-ump",
                          page="https://www.youtube.com/watch?v=abcdefghijk"))

        item = "yt:abcdefghijk"
        assert store.candidates(item, expected_duration=11.0)
        assert store.candidates(item, expected_duration=30.0) == []
        (cand,) = store.candidates(item, expected_duration=10.0)
        assert cand.kind == detect.AUDIO
        out = tmp_path / "a.webm"
        store.assemble(item, cand.key, out)
        assert out.read_bytes() == init + b"one" + b"two"

    def test_ad_video_ids_are_their_own_items(self, store):
        body = header(0, video_id="ADVERTISEM1", seq=1) + media(0, b"ad") + end(0)
        store.ingest(resp("https://rr1.googlevideo.com/videoplayback", body,
                          mime="application/vnd.yt-ump",
                          page="https://www.youtube.com/watch?v=abcdefghijk"))
        assert store.streams("yt:abcdefghijk") == []
        assert store.streams("yt:ADVERTISEM1")


class TestEmbeddedYouTube:
    """A YouTube player on another site's page is filed under yt:<id>, and
    linked to the page so the play in that tab still finds it."""

    BLOG = "https://blog.example/post/7"

    def ingest_embed(self, store):
        body = (header(1, seq=1, start_ms=0, duration_ms=5000) + media(1, b"one") + end(1))
        store.ingest(resp("https://rr1.googlevideo.com/videoplayback", body,
                          mime="application/vnd.yt-ump", page=self.BLOG))

    def test_page_item_reaches_the_embedded_video(self, store):
        self.ingest_embed(store)
        page = detect.page_id(self.BLOG)
        assert store.members(page) == [page, "yt:abcdefghijk"]
        assert store.summary(page).streams == 1
        (cand,) = store.candidates(page, expected_duration=5.0)
        assert cand.item == "yt:abcdefghijk"

    def test_claim_and_discard_cover_the_embed(self, store):
        self.ingest_embed(store)
        page = detect.page_id(self.BLOG)
        store.claim(page, "wanted")
        meta = json.loads((store.root / digest("yt:abcdefghijk") / "item.json").read_text())
        assert meta["claimed"] is True
        store.discard(page)
        assert store.streams("yt:abcdefghijk") == []

    def test_on_youtube_other_ids_stay_apart(self, store):
        """There, another video id is an ad."""
        body = header(0, video_id="ADVERTISEM1", seq=1) + media(0, b"ad") + end(0)
        store.ingest(resp("https://rr1.googlevideo.com/videoplayback", body,
                          mime="application/vnd.yt-ump",
                          page="https://www.youtube.com/watch?v=abcdefghijk"))
        assert store.members("yt:abcdefghijk") == ["yt:abcdefghijk"]


class TestMissing:
    """What the page is asked to fetch when a play ends incomplete."""

    URL = "https://cdn.example.tv/v/file.mp4"

    def test_holes_in_a_file_become_ranges(self, store, monkeypatch):
        monkeypatch.setattr(streams, "FETCH_CHUNK_BYTES", 1500)
        store.ingest(ranged(self.URL, os.urandom(1000), 0, 5000))
        store.ingest(ranged(self.URL, os.urandom(500), 2000, 5000))
        assert store.missing(ITEM) == [
            FetchRequest(self.URL, (1000, 1999)),
            FetchRequest(self.URL, (2500, 3999)),
            FetchRequest(self.URL, (4000, 4999)),
        ]

    def test_unknown_size_asks_for_one_piece_past_what_is_held(self, store, monkeypatch):
        monkeypatch.setattr(streams, "FETCH_CHUNK_BYTES", 1000)
        store.ingest(ranged(self.URL, os.urandom(300), 0, None))
        assert store.missing(ITEM) == [FetchRequest(self.URL, (300, 1299))]

    def test_query_ranges_are_asked_for_in_the_query(self, store):
        base = "https://rr3.googlevideo.com/videoplayback?id=o-X&itag=140&clen=3000&range="
        store.ingest(resp(base + "0-1499", os.urandom(1500), mime="audio/mp4"))
        (request,) = store.missing(ITEM)
        assert request.range is None
        assert detect.query_range(request.url) == (1500, 2999)
        assert detect.query_value(request.url, "itag") == "140"

    def test_a_fetched_piece_lands_in_its_item_after_the_tab_moved_on(self, store):
        store.ingest(ranged(self.URL, b"a" * 10, 0, 20))
        (request,) = store.missing(ITEM)
        store.ingest(ranged(request.url, b"b" * 10, 10, 20, page="https://www.example.tv/next"))
        assert len(store.candidates(ITEM)) == 1
        assert store.missing(ITEM) == []

    def test_listed_but_unfetched_segments(self, store):
        base = "https://cdn.example.tv/v/"
        store.ingest(resp(base + "hi/index.m3u8", hls_media().encode(), mime="audio/mpegurl"))
        for i in (0, 2):
            store.ingest(resp(base + f"hi/seg{i}.ts", ts(b"x"), mime="video/mp2t"))
        assert store.missing(ITEM) == [FetchRequest(base + "hi/seg1.ts")]

    def test_hls_byte_ranges_keep_their_range(self, store):
        base = "https://cdn.example.tv/v/"
        text = ("#EXTM3U\n#EXT-X-MEDIA-SEQUENCE:0\n"
                "#EXTINF:4.0,\n#EXT-X-BYTERANGE:376@0\nall.ts\n"
                "#EXTINF:4.0,\n#EXT-X-BYTERANGE:376@376\nall.ts\n#EXT-X-ENDLIST")
        store.ingest(resp(base + "index.m3u8", text.encode(), mime="audio/mpegurl"))
        store.ingest(resp(base + "all.ts", ts(b"x"), mime="video/mp2t", status=206,
                          headers={"content-range": "bytes 0-375/752"}))
        assert store.missing(ITEM) == [FetchRequest(base + "all.ts", (376, 751))]

    def test_a_missed_init_segment_is_fetched(self, store):
        base = "https://cdn.example.tv/d/"
        store.ingest(resp(base + "m.mpd", DASH.encode(), mime="application/dash+xml"))
        store.ingest(resp(base + "v1/1.m4s", media_segment(1, 0)))
        store.ingest(resp(base + "v1/2.m4s", media_segment(2, 4)))
        assert store.missing(ITEM, 8.0) == [FetchRequest(base + "v1/init.mp4")]

    def test_the_best_rendition_is_the_one_completed(self, store):
        """A quality switch mid-play: finish the sharper rendition, not the other."""
        base = "https://cdn.example.tv/v/"
        store.ingest(resp(base + "master.m3u8", HLS_MASTER.encode(),
                          mime="application/vnd.apple.mpegurl"))
        store.ingest(resp(base + "hi/index.m3u8", hls_media().encode(), mime="audio/mpegurl"))
        store.ingest(resp(base + "lo/index.m3u8", hls_media().encode(), mime="audio/mpegurl"))
        store.ingest(resp(base + "hi/seg0.ts", ts(b"hi") * 4, mime="video/mp2t"))
        for i in (1, 2):
            store.ingest(resp(base + f"lo/seg{i}.ts", ts(b"lo"), mime="video/mp2t"))
        assert store.missing(ITEM) == [FetchRequest(base + "hi/seg1.ts"),
                                       FetchRequest(base + "hi/seg2.ts")]

    def test_complete_streams_need_nothing(self, store):
        store.ingest(ranged(self.URL, b"abc", 0, 3))
        assert store.missing(ITEM) == []

    def test_an_ad_length_stream_is_not_completed(self, store):
        data = init_segment(trak(b"vide", b"avc1"), duration_ms=15_000) + os.urandom(100)
        store.ingest(ranged(self.URL, data, 0, len(data) * 4))
        assert store.missing(ITEM, 216.0) == []
        assert store.missing(ITEM, 15.0)

    def test_encrypted_streams_are_not_completed(self, store):
        base = "https://cdn.example.tv/d/"
        store.ingest(resp(base + "m.mpd", DASH.encode(), mime="application/dash+xml"))
        store.ingest(resp(base + "v1/init.mp4", init_segment(trak(b"vide", b"encv"))))
        assert store.missing(ITEM, 8.0) == []

    def test_ump_can_not_be_replayed(self, store):
        body = header(1, seq=1, start_ms=0, duration_ms=5000) + media(1, b"one") + end(1)
        store.ingest(resp("https://rr1.googlevideo.com/videoplayback", body,
                          mime="application/vnd.yt-ump",
                          page="https://www.youtube.com/watch?v=abcdefghijk"))
        assert store.missing("yt:abcdefghijk", 30.0) == []


class TestLifecycle:
    def test_state_survives_a_restart(self, store, tmp_path):
        store.ingest(ranged("https://x/a.mp4", b"abc", 0, 6))
        again = StreamStore(store.root)
        again.ingest(ranged("https://x/a.mp4", b"def", 3, 6))
        assert len(again.candidates(ITEM)) == 1

    def test_unclaimed_items_are_purged_first(self, store):
        store.ingest(ranged("https://x/a.mp4", b"abc", 0, 6))
        store.ingest(resp("https://y/b.mp4", b"abc", status=206,
                          headers={"content-range": "bytes 0-2/6"}, page="https://other.tv/p"))
        store.claim(ITEM, "wanted")
        old = time.time() - 7200
        for idir in store.root.iterdir():
            meta = json.loads((idir / "item.json").read_text())
            meta["updated"] = old
            (idir / "item.json").write_text(json.dumps(meta))

        assert store.purge() == 1
        assert store.streams(ITEM)
        assert store.streams(detect.page_id("https://other.tv/p")) == []

    def test_discard(self, store):
        store.ingest(ranged("https://x/a.mp4", b"abc", 0, 6))
        store.discard(ITEM)
        assert store.streams(ITEM) == []
        assert not (store.root / digest(ITEM)).exists()

    def test_a_bad_body_never_raises(self, store):
        assert store.ingest(resp("https://x/m.mpd", b"<MPD <<<", mime="application/dash+xml")) == MANIFEST


class TestWhichDrm:
    def test_first_mark_and_every_new_system_are_logged(self, store, caplog):
        with caplog.at_level("INFO"):
            store.mark_drm(ITEM, "the page attached MediaKeys")
            store.mark_drm(ITEM, "again")                       # nothing new
            store.mark_drm(ITEM, "a pssh", ("Widevine", "PlayReady"))
            store.mark_drm(ITEM, "same pssh", ("Widevine",))     # nothing new
        lines = [r.message for r in caplog.records if "DRM-protected" in r.message]
        assert len(lines) == 2
        assert "DRM system not named yet" in lines[0]
        assert "Widevine, PlayReady" in lines[1]
        assert store.summary(ITEM).drm_systems == ("Widevine", "PlayReady")

    def test_encrypted_segments_report_their_pssh(self, store):
        base = "https://cdn.example.tv/d/"
        store.ingest(resp(base + "m.mpd", DASH.encode(), mime="application/dash+xml"))
        pssh = b"\0\0\0\x20pssh\0\0\0\0" + bytes.fromhex("9a04f07998404286ab92e65be0885f95") + b"\0\0\0\0"
        store.ingest(resp(base + "v1/init.mp4",
                          init_segment(trak(b"vide", b"encv"), extra=pssh)))
        assert store.summary(ITEM).drm_systems == ("PlayReady",)


class TestDrmLabels:
    def test_a_level_replaces_the_bare_system(self, store):
        store.mark_drm(ITEM, "pssh", ("Widevine", "PlayReady"))
        store.mark_drm(ITEM, "EME", ("Widevine L3 (SW_SECURE_DECODE)",))
        store.mark_drm(ITEM, "another pssh", ("Widevine",))   # adds nothing now
        assert store.summary(ITEM).drm_systems == ("PlayReady", "Widevine L3 (SW_SECURE_DECODE)")


class TestPlaybackLevel:
    HOW = "audio decrypted by the software CDM (DecryptingDemuxerStream)"

    def test_the_one_system_gets_the_playback_level(self, store, caplog):
        store.mark_drm(ITEM, "pssh", ("scheme cenc", "Widevine"))
        with caplog.at_level("INFO"):
            store.mark_playback(ITEM, "software", self.HOW)
            store.mark_playback(ITEM, "software", self.HOW)   # unchanged: not logged again
        assert store.summary(ITEM).drm_systems == (
            "scheme cenc", f"Widevine L3 (playback: {self.HOW})")
        assert sum("(the playback)" in r.message for r in caplog.records) == 1

    def test_several_candidates_get_the_level_in_each_ones_terms(self, store):
        store.mark_drm(ITEM, "pssh", ("Widevine", "PlayReady"))
        store.mark_playback(ITEM, "hardware", "MediaFoundationRenderer, the hardware-secure path")
        assert store.drm_systems(ITEM) == [
            "Widevine", "PlayReady",
            "playback hardware-secure: Widevine L1 / PlayReady SL3000 "
            "(MediaFoundationRenderer, the hardware-secure path)"]

    def test_the_pages_own_level_stands(self, store):
        store.mark_drm(ITEM, "EME", ("Widevine L3 (SW_SECURE_CRYPTO)",))
        store.mark_playback(ITEM, "software", self.HOW)
        assert store.drm_systems(ITEM) == ["Widevine L3 (SW_SECURE_CRYPTO)"]

    def test_the_playback_alone_still_says_its_level(self, store):
        store.mark_playback(ITEM, "software", self.HOW)
        assert store.is_drm(ITEM)
        assert store.drm_systems(ITEM) == [f"playback software-secure ({self.HOW})"]
