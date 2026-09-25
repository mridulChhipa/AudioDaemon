"""Listener state machine, driven by scripted media-session readings.

No browser, no LLM: the sniffer and the classifier are replaced, and the
stream store is real in a temp directory, so what's under test is purely the
decision logic -- which plays are taken, which are skipped and why, and what a
finished play is handed to the pipeline as.
"""
import asyncio
import json

import pytest

from core import listener as L
from core.agent import TrackVerdict
from core.sniffer import detect
from core.sniffer.cdp import TabInfo
from core.sniffer.streams import Response, StreamStore, digest

PAGE = "https://www.youtube.com/watch?v=abcdefghijk"
ITEM = "yt:abcdefghijk"


def snapshot(title="In the End", artist="Linkin Park", *, playing=True, duration=216.0,
             app_id="Comet"):
    return L.Snapshot(
        artist=artist, title=title, album="Hybrid Theory", album_artist=artist,
        track_number=1, playing=playing, duration=duration, app_id=app_id, info=None,
    )


class FakeSniffer:
    def __init__(self, store, tab=None):
        self.store = store
        self.tab = tab
        self.asked = []
        self.connecting = False

    async def match_tab(self, title):
        self.asked.append(title)
        return self.tab



@pytest.fixture
async def wired(tmp_path, monkeypatch):
    """A Listener with a matched tab, a real StreamStore, and stubbed classifier
    and pipeline.

    Async so its teardown runs while the event loop is still open -- cancelling
    a pending decision task needs a live loop.
    """
    from core import database

    monkeypatch.setattr(database, "DB_PATH", tmp_path / "memory.db")
    database.init_db(tmp_path / "memory.db")
    monkeypatch.setattr(L, "STREAM_PROBE_SECONDS", 0)

    async def no_cover(info):
        return None

    monkeypatch.setattr(L.encoder, "read_session_cover", no_cover)

    submitted = []

    class FakePipeline:
        def start(self): pass
        def submit(self, job): submitted.append(job)

    class FakeAgent:
        result = TrackVerdict("MUSIC", "Linkin Park", "In the End")
        calls = []

        async def classify(self, artist, title, site="", has_video=None):
            self.calls.append((artist, title, site, has_video))
            return self.result

    store = StreamStore(tmp_path / "streams")
    tab = TabInfo("T1", url=PAGE, title="In the End - YouTube", session_id="S1")
    sniffer = FakeSniffer(store, tab)
    agent = FakeAgent()
    lis = L.Listener(sniffer, FakePipeline(), agent)
    yield lis, agent, submitted, store, sniffer

    # Tests that end mid-play leave a background decision task pending,
    # which asyncio complains about at interpreter shutdown.
    lis._reset()
    await asyncio.sleep(0)  # let the cancellation actually be delivered


async def drive(lis, snaps, monkeypatch):
    """Feed a sequence of snapshots through _tick, one poll each."""
    for snap in snaps:
        async def read(_s=snap):
            return _s
        monkeypatch.setattr(L, "read_session", read)
        await lis._tick()
        # let the background decision task finish
        await asyncio.sleep(0)
        await asyncio.sleep(0)


async def settle(lis, snaps, monkeypatch):
    await drive(lis, snaps, monkeypatch)
    if lis._deciding is not None:
        await asyncio.wait_for(asyncio.shield(lis._deciding), 1)
        await drive(lis, [snaps[-1]], monkeypatch)


@pytest.mark.parametrize("app_id,name", [
    ("Comet.OOYZE3HLEZSNWAR6L5FEVE3ZKE", "Comet"),
    ("SpotifyAB.SpotifyMusic_zpdnekdrzrea0!Spotify", "Spotify"),
    ("Spotify.exe", "Spotify"),
    ("", ""),
])
def test_app_name(app_id, name):
    assert L.app_name(app_id) == name


def test_is_browser():
    assert L.is_browser("Comet")
    assert L.is_browser("Chrome")
    assert L.is_browser("MSEdge")
    assert not L.is_browser("Spotify.exe")
    assert not L.is_browser("")


@pytest.mark.asyncio
async def test_nothing_playing_starts_nothing(wired, monkeypatch):
    lis, *_ = wired
    await drive(lis, [snapshot(playing=False)], monkeypatch)
    assert lis._snapshot is None


@pytest.mark.asyncio
async def test_no_duration_is_not_taken(wired, monkeypatch):
    """Without a length, neither completeness nor the ad filter can be judged."""
    lis, *_ = wired
    await drive(lis, [snapshot(duration=0.0)], monkeypatch)
    assert lis._snapshot is None


@pytest.mark.asyncio
async def test_browser_play_claims_its_page(wired, monkeypatch):
    lis, agent, submitted, store, _ = wired
    await settle(lis, [snapshot(), snapshot()], monkeypatch)
    await drive(lis, [snapshot(playing=False)], monkeypatch)

    (job,) = submitted
    assert (job.artist, job.title) == ("Linkin Park", "In the End")
    assert job.item == ITEM and job.site == "youtube.com"
    assert job.duration == 216.0
    assert agent.calls[-1][2] == "youtube.com"
    meta = json.loads((store.root / digest(ITEM) / "item.json").read_text())
    assert meta["claimed"] is True  # the unclaimed-media purge leaves it alone


@pytest.mark.asyncio
async def test_the_classifier_hears_what_the_streams_carry(wired, monkeypatch):
    lis, agent, _, store, _ = wired
    from tests.test_detect import init_segment, trak

    store.ingest(Response(PAGE, "https://cdn/v.mp4", 206, "video/mp4",
                          {"content-range": "bytes 0-99/1000"},
                          (init_segment(trak(b"vide", b"avc1")) + bytes(100))[:100]))
    await settle(lis, [snapshot()], monkeypatch)
    assert agent.calls[-1][3] is True


@pytest.mark.asyncio
async def test_track_change_closes_and_starts_the_next(wired, monkeypatch):
    lis, _, submitted, _, _ = wired
    await settle(lis, [snapshot()], monkeypatch)
    await drive(lis, [snapshot(title="Numb")], monkeypatch)

    assert [j.title for j in submitted] == ["In the End"]
    assert lis._snapshot is not None and lis._snapshot.title == "Numb"


@pytest.mark.asyncio
async def test_decision_still_running_at_the_end_is_waited_for(wired, monkeypatch):
    lis, _, submitted, _, _ = wired
    await drive(lis, [snapshot()], monkeypatch)
    await drive(lis, [snapshot(playing=False)], monkeypatch)
    assert len(submitted) == 1


@pytest.mark.asyncio
async def test_not_wanted_is_dropped_and_not_asked_again(wired, monkeypatch):
    lis, agent, submitted, _, _ = wired
    agent.result = TrackVerdict("OTHER", "Some Podcast", "Episode 12")

    await settle(lis, [snapshot(), snapshot()], monkeypatch)
    await drive(lis, [snapshot(playing=False), snapshot(), snapshot()], monkeypatch)

    assert submitted == []
    assert len(agent.calls) == 1


@pytest.mark.asyncio
async def test_no_title_is_not_kept(wired, monkeypatch):
    lis, agent, submitted, _, _ = wired
    await settle(lis, [snapshot(title=""), snapshot(title="")], monkeypatch)
    await drive(lis, [snapshot(title="", playing=False)], monkeypatch)
    assert submitted == [] and agent.calls == []


@pytest.mark.asyncio
async def test_already_owned_is_not_taken(wired, monkeypatch, tmp_path):
    lis, _, submitted, _, _ = wired
    from core import database

    owned = tmp_path / "owned.opus"
    owned.write_bytes(b"x")
    database.add_media(database.make_track_hash("Linkin Park", "In the End"),
                       "Linkin Park", "In the End", "x", str(owned))
    await settle(lis, [snapshot(), snapshot()], monkeypatch)
    await drive(lis, [snapshot(playing=False)], monkeypatch)
    assert submitted == []


@pytest.mark.asyncio
async def test_a_desktop_player_is_not_archived(wired, monkeypatch, caplog):
    """Spotify's app: no browser tab, so no stream to take."""
    lis, agent, submitted, _, sniffer = wired
    spotify = "SpotifyAB.SpotifyMusic_zpdnekdrzrea0!Spotify"
    with caplog.at_level("INFO"):
        await drive(lis, [snapshot(app_id=spotify)] * 3, monkeypatch)

    assert lis._snapshot is None and agent.calls == [] and sniffer.asked == []
    assert sum("not playing in a captured browser tab (Spotify)" in r.message
               for r in caplog.records) == 1


@pytest.mark.asyncio
async def test_a_browser_play_in_no_captured_tab_is_not_archived(wired, monkeypatch, caplog):
    lis, agent, _, _, sniffer = wired
    sniffer.tab = None
    with caplog.at_level("INFO"):
        await drive(lis, [snapshot()] * 3, monkeypatch)

    assert lis._snapshot is None and agent.calls == []
    assert len(sniffer.asked) == 1  # not looked for on every poll
    assert sum("not playing in a captured browser tab" in r.message
               for r in caplog.records) == 1


@pytest.mark.asyncio
async def test_the_same_play_is_found_once_the_debugging_browser_has_it(
        wired, monkeypatch, caplog):
    """Started in a plain browser, then replayed in the captured one: taken,
    without a second "not archived" line in between."""
    lis, _, submitted, _, sniffer = wired
    tab, sniffer.tab = sniffer.tab, None
    with caplog.at_level("INFO"):
        await drive(lis, [snapshot()], monkeypatch)
        sniffer.tab = tab
        lis._unmatched_at -= L._REMATCH_SECONDS  # time passes
        await settle(lis, [snapshot(), snapshot()], monkeypatch)
        await drive(lis, [snapshot(playing=False)], monkeypatch)

    assert len(submitted) == 1
    assert sum("not playing in a captured browser tab" in r.message
               for r in caplog.records) == 1


@pytest.mark.asyncio
async def test_an_unmatched_track_is_looked_for_again_after_another(wired, monkeypatch):
    lis, _, _, _, sniffer = wired
    sniffer.tab = None
    await drive(lis, [snapshot(), snapshot(title="Numb"), snapshot()], monkeypatch)
    assert sniffer.asked == ["In the End", "Numb", "In the End"]


def _encrypted_stream(store):
    store.mark_drm(ITEM, "test", ("Widevine",))
    state = store._state(ITEM, "k", "seq")
    state.kind = detect.VIDEO
    state.encrypted = True
    store._save(ITEM, state)


@pytest.mark.asyncio
async def test_drm_protected_page_is_not_archived(wired, monkeypatch, caplog):
    lis, agent, submitted, store, _ = wired
    _encrypted_stream(store)
    with caplog.at_level("INFO"):
        await settle(lis, [snapshot(), snapshot()], monkeypatch)
        await drive(lis, [snapshot(playing=False)], monkeypatch)

    assert submitted == [] and agent.calls == []
    assert any("Not archived (DRM-protected: Widevine)" in r.message
               for r in caplog.records)


@pytest.mark.asyncio
async def test_drm_flag_beside_a_clear_stream_is_still_taken(wired, monkeypatch):
    """A page can attach MediaKeys for an ad while the video itself is clear."""
    lis, _, submitted, store, _ = wired
    store.mark_drm(ITEM, "the page attached MediaKeys")
    store.ingest(Response(PAGE, "https://cdn/a.webm", 206, "audio/webm",
                          {"content-range": "bytes 0-9/100"}, bytes(10)))
    await settle(lis, [snapshot(), snapshot()], monkeypatch)
    await drive(lis, [snapshot(playing=False)], monkeypatch)
    assert len(submitted) == 1


@pytest.mark.asyncio
async def test_the_playback_level_is_in_the_log(wired, monkeypatch, caplog):
    """Spotify set up its keys before the daemon attached: its player's own
    pipeline says the level. And no generic "dropping" line follows."""
    lis, _, submitted, store, _ = wired
    store.mark_drm(ITEM, "pssh", ("scheme cenc", "Widevine"))
    store.mark_playback(ITEM, "software",
                        "audio decrypted by the software CDM (DecryptingDemuxerStream)")
    with caplog.at_level("INFO"):
        await settle(lis, [snapshot(), snapshot()], monkeypatch)

    assert submitted == []
    assert any("Not archived (DRM-protected: scheme cenc, Widevine L3 (playback: audio "
               "decrypted by the software CDM (DecryptingDemuxerStream))): Linkin Park - "
               "In the End" in r.message for r in caplog.records)
    assert not any("Dropping play" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_a_browser_play_waits_while_the_browser_is_being_asked(wired, monkeypatch, caplog):
    """While the user hasn't answered the Allow prompt, a play isn't declared
    "not in a captured tab" -- it is taken once the browser answers."""
    lis, _, submitted, _, sniffer = wired
    sniffer.connecting = True
    with caplog.at_level("INFO"):
        await drive(lis, [snapshot()] * 3, monkeypatch)
        assert sniffer.asked == [] and lis._snapshot is None
        sniffer.connecting = False
        await settle(lis, [snapshot(), snapshot()], monkeypatch)
        await drive(lis, [snapshot(playing=False)], monkeypatch)

    assert len(submitted) == 1
    assert not any("not playing in a captured browser tab" in r.message for r in caplog.records)
