"""Whole plays through the pipeline: completing, choosing and routing streams.

A real StreamStore and database in a temp directory; ffmpeg is never run --
probing, remuxing and muxing are stubbed, keyed by a marker at the start of
each synthetic stream.
"""
import sqlite3
from pathlib import Path

import pytest

from core import database, encoder, pipeline
from core.agent import TrackVerdict
from core.encoder import ProbeInfo
from core.pipeline import MediaJob, Pipeline, _Assembled, choose
from core.sniffer.streams import Response, StreamStore

ITEM = "yt:abcdefghijk"
PAGE = "https://www.youtube.com/watch?v=abcdefghijk"

PROBES = {
    b"AUD": ProbeInfo(100.0, None, "opus"),
    b"VID": ProbeInfo(100.0, "vp9", None, 1920, 1080),
    b"LOW": ProbeInfo(100.0, "vp9", None, 640, 360),
    b"MUX": ProbeInfo(100.0, "h264", "aac", 1280, 720),
    b"PAD": ProbeInfo(108.0, "h264", "aac", 1280, 720),
    b"ADV": ProbeInfo(15.0, "h264", "aac", 1280, 720),
}


def library(db, kind=None):
    """What the library records, as (kind, title) rows."""
    with sqlite3.connect(db) as conn:
        rows = conn.execute("SELECT kind, title FROM acquired_media").fetchall()
    return [r for r in rows if kind is None or r[0] == kind]


def put_file(store, url, marker, size=4000, complete=True):
    """A progressive file, fetched whole or only its first half."""
    body = marker + bytes(size - len(marker))
    end = size if complete else size // 2
    store.ingest(Response(PAGE, url, 206, "video/mp4",
                          {"content-range": f"bytes 0-{end - 1}/{size}"}, body[:end]))
    return body


class FakeSniffer:
    """Completes an item the way the page would: by fetching what's missing."""

    def __init__(self, store):
        self.store = store
        self.served: dict[str, bytes] = {}   # url -> whole file the "server" has
        self.calls = []

    async def complete(self, item, duration):
        self.calls.append((item, duration))
        requests = self.store.missing(item, duration)
        for r in requests:
            body = self.served.get(r.url)
            if body is None or r.range is None:
                continue
            first, last = r.range
            self.store.ingest(Response(PAGE, r.url, 206, "video/mp4",
                {"content-range": f"bytes {first}-{last}/{len(body)}"}, body[first:last + 1]))
        return len(requests)


@pytest.fixture
def media(tmp_path, monkeypatch):
    db = tmp_path / "memory.db"
    database.init_db(db)
    monkeypatch.setattr(database, "DB_PATH", db)
    monkeypatch.setattr(encoder, "MUSIC_DIR", tmp_path / "library" / "music")
    monkeypatch.setattr(encoder, "VIDEO_DIR", tmp_path / "library" / "video")
    monkeypatch.setattr(pipeline, "STAGING_DIR", tmp_path / "staging")
    monkeypatch.setattr(pipeline, "STREAM_SETTLE_SECONDS", 0)

    made = []

    def fake_probe(path):
        return PROBES.get(path.read_bytes()[:3])

    async def fake_remux(src, dest):
        made.append(("audio", src.read_bytes()[:3], dest))
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"x")
        return True

    async def fake_mux(video, audio, dest, meta=None):
        made.append(("video", video.read_bytes()[:3],
                     audio.read_bytes()[:3] if audio else None, dest))
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"x")
        return True

    monkeypatch.setattr(encoder, "probe", fake_probe)
    monkeypatch.setattr(encoder, "remux_audio", fake_remux)
    monkeypatch.setattr(encoder, "mux_video", fake_mux)
    monkeypatch.setattr(encoder, "write_tags", lambda *a, **k: None)

    store = StreamStore(tmp_path / "streams")
    sniffer = FakeSniffer(store)
    return Pipeline(store, sniffer), tmp_path, store, made, db, sniffer


def media_job(verdict_type="MUSIC", *, item=ITEM, duration=100.0, has_video=None):
    return MediaJob(
        verdict=TrackVerdict(verdict_type, "Artist", "Title"), album="", album_artist="",
        track_number=0, duration=duration, item=item, site="youtube.com",
        has_video=has_video, cover=None, source="Comet", closed_at=0.0,
    )


@pytest.mark.asyncio
async def test_music_video_lands_in_both_libraries(media):
    pipe, root, store, made, db, _ = media
    put_file(store, "https://cdn.example/a.webm", b"AUD")
    put_file(store, "https://cdn.example/v.webm", b"VID")
    put_file(store, "https://cdn.example/low.webm", b"LOW")

    await pipe.process_media(media_job("MUSIC"))

    audio = [m for m in made if m[0] == "audio"]
    video = [m for m in made if m[0] == "video"]
    assert audio[0][1] == b"AUD" and audio[0][2].name == "Artist - Title.opus"
    # The sharpest picture, paired with the separate audio track.
    assert video[0][1:3] == (b"VID", b"AUD")
    assert video[0][3].suffix == ".webm" and video[0][3].parent.name == "Artist"
    assert database.get_file_path(database.make_track_hash("Artist", "Title"), db)
    assert database.get_file_path(database.make_video_hash(ITEM), db, kind="video")
    assert store.streams(ITEM) == []          # pieces gone once both are made
    assert list((root / "staging").glob("*")) == []   # assemblies cleaned up


@pytest.mark.asyncio
async def test_plain_music_is_not_filed_as_video(media):
    pipe, _, store, made, db, _ = media
    put_file(store, "https://cdn.example/a.m4a", b"AUD")

    await pipe.process_media(media_job("MUSIC"))

    assert [m[0] for m in made] == ["audio"]
    assert library(db, "video") == []


@pytest.mark.asyncio
async def test_video_goes_only_to_the_video_library(media):
    pipe, _, store, made, db, _ = media
    put_file(store, "https://cdn.example/m.mp4", b"MUX")

    await pipe.process_media(media_job("VIDEO"))

    assert [m[0] for m in made] == ["video"]
    assert made[0][1:3] == (b"MUX", None) and made[0][3].suffix == ".mp4"
    assert library(db, "music") == []


@pytest.mark.asyncio
async def test_an_ad_on_the_page_is_not_mistaken_for_the_video(media):
    pipe, _, store, made, db, _ = media
    put_file(store, "https://ads.example/ad.mp4", b"ADV")

    await pipe.process_media(media_job("VIDEO"))

    assert made == []
    assert library(db) == []
    assert store.streams(ITEM)  # kept: the real video may still complete


@pytest.mark.asyncio
async def test_the_only_stream_may_be_a_little_padded(media):
    """108s against a 100s play: too far for the strict check, but it's the
    page's only stream, so not an ad beside the real one."""
    pipe, _, store, made, _, _ = media
    put_file(store, "https://cdn.example/m.mp4", b"PAD")

    await pipe.process_media(media_job("VIDEO"))

    assert [m[1] for m in made] == [b"PAD"]


@pytest.mark.asyncio
async def test_padding_is_not_forgiven_beside_a_better_match(media):
    pipe, _, store, made, _, _ = media
    put_file(store, "https://cdn.example/m.mp4", b"MUX", size=3000)
    put_file(store, "https://cdn.example/p.mp4", b"PAD", size=5000)

    await pipe.process_media(media_job("VIDEO"))

    assert [m[1] for m in made] == [b"MUX"]


@pytest.mark.asyncio
async def test_a_stream_stopped_halfway_is_completed_from_the_page(media):
    pipe, _, store, made, db, sniffer = media
    url = "https://cdn.example/a.webm"
    sniffer.served[url] = put_file(store, url, b"AUD", complete=False)

    await pipe.process_media(media_job("MUSIC"))

    assert sniffer.calls == [(ITEM, 100.0)]
    assert [m[:2] for m in made] == [("audio", b"AUD")]
    assert database.get_file_path(database.make_track_hash("Artist", "Title"), db)


@pytest.mark.asyncio
async def test_what_can_not_be_completed_waits_for_a_later_play(media, caplog):
    pipe, _, store, made, _, _ = media
    put_file(store, "https://cdn.example/a.webm", b"AUD", complete=False)  # server gone

    with caplog.at_level("INFO"):
        await pipe.process_media(media_job("MUSIC"))

    assert made == []
    assert store.streams(ITEM)  # a later play may fill the gap
    assert any("not complete yet" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_drm_is_reported_and_nothing_is_made(media, caplog):
    pipe, _, store, made, db, _ = media
    store.mark_drm(ITEM, "test", ("PlayReady", "scheme cenc"))

    with caplog.at_level("INFO"):
        await pipe.process_media(media_job("VIDEO", has_video=True))

    assert made == []
    assert library(db) == []
    assert any("DRM-protected (PlayReady, scheme cenc); not archived" in r.message
               for r in caplog.records)


@pytest.mark.asyncio
async def test_already_owned_music_is_not_made_twice(media):
    pipe, root, store, made, db, _ = media
    owned = root / "owned.opus"
    owned.write_bytes(b"x")
    database.add_media(database.make_track_hash("Artist", "Title"), "Artist", "Title",
                       "x", str(owned), db)
    put_file(store, "https://cdn.example/a.webm", b"AUD")

    await pipe.process_media(media_job("MUSIC"))
    assert made == []


@pytest.mark.asyncio
async def test_queue_is_drained_in_order(media):
    pipe = media[0]
    order = []

    async def record(job):
        order.append(job.verdict.clean_title)

    pipe.process_media = record
    pipe.start()
    for name in ("first", "second", "third"):
        job = media_job()
        job.verdict = TrackVerdict("MUSIC", "Artist", name)
        pipe.submit(job)
    await pipe.stop()

    assert order == ["first", "second", "third"]


def _a(name, info, size):
    return _Assembled(Path(name), info, size)


class TestChoose:
    def test_best_audio_is_the_largest_audio_only_stream(self):
        sel = choose([_a("a1", ProbeInfo(1, None, "opus"), 100),
                      _a("a2", ProbeInfo(1, None, "aac"), 300),
                      _a("m", ProbeInfo(1, "h264", "aac", 640, 360), 900)])
        assert sel.audio.path.name == "a2"

    def test_muxed_audio_when_there_is_no_audio_only_stream(self):
        sel = choose([_a("m", ProbeInfo(1, "h264", "aac", 640, 360), 900)])
        assert sel.audio.path.name == "m"
        assert sel.video.path.name == "m" and sel.video_audio is None

    def test_sharper_video_only_beats_muxed(self):
        sel = choose([_a("m", ProbeInfo(1, "h264", "aac", 640, 360), 900),
                      _a("v", ProbeInfo(1, "vp9", None, 1920, 1080), 500)])
        assert sel.video.path.name == "v"
        assert sel.video_audio.path.name == "m"

    def test_nothing(self):
        sel = choose([])
        assert sel.audio is None and sel.video is None


class TestAlreadyOwned:
    @pytest.mark.asyncio
    async def test_missing_file_is_forgotten(self, tmp_path, monkeypatch):
        db = tmp_path / "memory.db"
        database.init_db(db)
        monkeypatch.setattr(database, "DB_PATH", db)
        h = database.make_track_hash("A", "B")
        database.add_media(h, "A", "B", "Comet", str(tmp_path / "gone.opus"), db)

        assert await pipeline.already_owned(h) is False
        assert database.get_file_path(h, db) is None

    @pytest.mark.asyncio
    async def test_present_file_is_owned(self, tmp_path, monkeypatch):
        db = tmp_path / "memory.db"
        database.init_db(db)
        monkeypatch.setattr(database, "DB_PATH", db)
        real = tmp_path / "there.opus"
        real.write_bytes(b"x")
        h = database.make_track_hash("A", "B")
        database.add_media(h, "A", "B", "Comet", str(real), db)

        assert await pipeline.already_owned(h) is True
