"""The CDP sniffer's event handling, without a browser.

Events are fed to `_on_event` exactly as the reader loop would, and `send` is
replaced by a script, so what's under test is the bookkeeping: which responses
are kept, in what order streamed bytes are reassembled, which tab a play
belongs to, what is let go of, and how missing pieces are asked of the page.
"""
import asyncio
import base64
import json
import os
import time

import pytest

from core.sniffer.cdp import CdpError, Sniffer, TabInfo, find_browser, read_active_port
from core.sniffer.streams import STORED, UNPLACED, FetchRequest

URL = "https://cdn.example.tv/movie.mp4"


class RecordingStore:
    def __init__(self, outcome=STORED):
        self.responses = []
        self.drm = []
        self.outcome = outcome
        self.plans = []          # what missing() answers, one list per call
        self.playback = []

    def missing(self, item, duration=0.0):
        return self.plans.pop(0) if self.plans else []

    def ingest(self, response):
        self.responses.append(response)
        return self.outcome

    def mark_drm(self, item, reason="", systems=()):
        self.drm.append(item)
        self.systems = tuple(systems)

    def mark_playback(self, item, secure, how):
        self.playback.append((item, secure, how))


class ScriptedSniffer(Sniffer):
    """send() answers from a table; everything sent is recorded."""

    def __post_init__(self):
        super().__post_init__()
        self.sent = []
        self.answers = {}
        self.connected = True

    async def send(self, method, params=None, session_id=None, timeout=30.0):
        self.sent.append((method, params or {}, session_id))
        answer = self.answers.get(method, {})
        if callable(answer):
            answer = answer(params or {}, session_id)
        if isinstance(answer, Exception):
            raise answer
        return answer


class TestFindingTheBrowser:
    """The browser you use says where it listens, in its own profile folder,
    once remote debugging is turned on at chrome://inspect."""

    def profile(self, root, name, port=None, path="/devtools/browser/abc", mtime=None):
        data = root / name
        data.mkdir()
        if port is not None:
            f = data / "DevToolsActivePort"
            f.write_text(f"{port}\n{path}\n", encoding="utf-8")
            if mtime is not None:
                os.utime(f, (mtime, mtime))
        return data

    def test_reads_port_and_websocket_path(self, tmp_path):
        data = self.profile(tmp_path, "Comet", 9222, "/devtools/browser/1f2e")
        url, _ = read_active_port(data)
        assert url == "ws://127.0.0.1:9222/devtools/browser/1f2e"

    def test_debugging_off_means_no_file(self, tmp_path):
        assert read_active_port(self.profile(tmp_path, "Comet")) is None

    def test_a_malformed_file_is_ignored(self, tmp_path):
        assert read_active_port(self.profile(tmp_path, "Comet", "x")) is None

    def test_the_most_recently_started_browser_wins(self, tmp_path):
        old = self.profile(tmp_path, "Chrome", 9333, mtime=1_000)
        new = self.profile(tmp_path, "Comet", 9222, mtime=2_000)
        off = self.profile(tmp_path, "Edge")
        assert find_browser([old, off, new]) == "ws://127.0.0.1:9222/devtools/browser/abc"

    def test_none_enabled(self, tmp_path):
        assert find_browser([self.profile(tmp_path, "Comet"), tmp_path / "missing"]) is None


@pytest.fixture
def sniffer():
    s = ScriptedSniffer(RecordingStore())
    tab = TabInfo("T1", url="https://site.tv/watch/1", title="A Film - Site", session_id="S1")
    s.tabs["T1"] = tab
    s._session_root["S1"] = "T1"
    return s


async def drain(s):
    for _ in range(5):
        if s._tasks:
            await asyncio.gather(*list(s._tasks), return_exceptions=True)
        await asyncio.sleep(0)


def response_event(request_id="R1", url=URL, status=206, mime="video/mp4",
                   rtype="Media", headers=None):
    return {"requestId": request_id, "type": rtype, "response": {
        "url": url, "status": status, "mimeType": mime,
        "headers": headers if headers is not None else {"Content-Range": "bytes 1000-1999/5000"},
    }}


def b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


@pytest.mark.asyncio
async def test_finished_xhr_body_is_fetched_and_ingested(sniffer):
    sniffer.answers["Network.getResponseBody"] = {"body": b64(b"segment"), "base64Encoded": True}
    sniffer._on_event("Network.responseReceived",
                      response_event(rtype="XHR", status=200, headers={}), "S1")
    sniffer._on_event("Network.loadingFinished", {"requestId": "R1"}, "S1")
    await drain(sniffer)

    (resp,) = sniffer.store.responses
    assert resp.body == b"segment"
    assert resp.page_url == "https://site.tv/watch/1"
    assert sniffer.tabs["T1"].last_media > 0


@pytest.mark.asyncio
async def test_text_bodies_are_encoded(sniffer):
    sniffer.answers["Network.getResponseBody"] = {"body": "#EXTM3U", "base64Encoded": False}
    sniffer._on_event("Network.responseReceived",
                      response_event(url="https://x/a.m3u8", rtype="XHR", status=200,
                                     mime="application/vnd.apple.mpegurl", headers={}), "S1")
    sniffer._on_event("Network.loadingFinished", {"requestId": "R1"}, "S1")
    await drain(sniffer)
    assert sniffer.store.responses[0].body == b"#EXTM3U"


@pytest.mark.asyncio
async def test_non_media_is_not_tracked(sniffer):
    sniffer._on_event("Network.responseReceived",
                      response_event(url="https://x/app.js", mime="application/javascript",
                                     rtype="Script"), "S1")
    assert sniffer._pending == {}


@pytest.mark.asyncio
async def test_streamed_media_keeps_order_and_offsets(sniffer):
    """Bytes that arrived before streaming began come first, and a cancelled
    request still yields what it received, at the right place in the file."""
    gate = asyncio.Event()

    async def slow_stream(params):
        await gate.wait()
        return {"bufferedData": b64(b"AAAA")}

    answers = {"Network.streamResourceContent": None}

    async def send(method, params=None, session_id=None, timeout=30.0):
        if method == "Network.streamResourceContent":
            return await slow_stream(params)
        return answers.get(method, {})

    sniffer.send = send
    sniffer._on_event("Network.responseReceived", response_event(), "S1")
    await asyncio.sleep(0)
    # Data arriving while the stream request is in flight...
    sniffer._on_event("Network.dataReceived", {"requestId": "R1", "data": b64(b"BBBB")}, "S1")
    gate.set()
    await drain(sniffer)
    sniffer._on_event("Network.dataReceived", {"requestId": "R1", "data": b64(b"CC")}, "S1")
    # ...and the player cancelling the rest.
    sniffer._on_event("Network.loadingFailed", {"requestId": "R1", "canceled": True}, "S1")
    await drain(sniffer)

    (resp,) = sniffer.store.responses
    assert resp.body == b"AAAABBBBCC"
    assert resp.headers["content-range"] == "bytes 1000-1009/5000"


@pytest.mark.asyncio
async def test_large_streams_flush_in_pieces(sniffer, monkeypatch):
    from core.sniffer import cdp

    monkeypatch.setattr(cdp, "_STREAM_FLUSH_BYTES", 4)
    sniffer.answers["Network.streamResourceContent"] = {"bufferedData": ""}
    sniffer._on_event("Network.responseReceived", response_event(), "S1")
    await drain(sniffer)
    for chunk in (b"0123", b"4567", b"89"):
        sniffer._on_event("Network.dataReceived", {"requestId": "R1", "data": b64(chunk)}, "S1")
    sniffer._on_event("Network.loadingFinished", {"requestId": "R1"}, "S1")
    await drain(sniffer)

    ranges = sorted(r.headers["content-range"] for r in sniffer.store.responses)
    assert ranges == ["bytes 1000-1003/5000", "bytes 1004-1007/5000", "bytes 1008-1009/5000"]


@pytest.mark.asyncio
async def test_unstreamable_media_falls_back_to_the_body(sniffer, monkeypatch):
    monkeypatch.setattr(asyncio, "sleep", _instant_sleep)
    sniffer.answers["Network.streamResourceContent"] = CdpError("does not exists")
    sniffer.answers["Network.getResponseBody"] = {"body": b64(b"whole"), "base64Encoded": True}
    sniffer._on_event("Network.responseReceived", response_event(), "S1")
    await drain(sniffer)
    sniffer._on_event("Network.loadingFinished", {"requestId": "R1"}, "S1")
    await drain(sniffer)
    assert [r.body for r in sniffer.store.responses] == [b"whole"]


_real_sleep = asyncio.sleep


async def _instant_sleep(delay, *a, **k):
    await _real_sleep(0)


@pytest.mark.asyncio
async def test_unplaced_segment_asks_the_page_for_its_manifests(sniffer):
    sniffer.store.outcome = UNPLACED
    sniffer.answers["Network.getResponseBody"] = {"body": b64(b"ts"), "base64Encoded": True}
    sniffer.answers["Runtime.evaluate"] = {"result": {"value": 1}}
    for rid in ("R1", "R2"):
        sniffer._on_event("Network.responseReceived",
                          response_event(request_id=rid, rtype="XHR", status=200, headers={}), "S1")
        sniffer._on_event("Network.loadingFinished", {"requestId": rid}, "S1")
    await drain(sniffer)

    evaluations = [p for m, p, _ in sniffer.sent if m == "Runtime.evaluate"]
    assert len(evaluations) == 1  # rate-limited
    assert "performance.getEntriesByType" in evaluations[0]["expression"]


@pytest.mark.asyncio
async def test_manifest_recovery_asks_every_frame_of_the_tab(sniffer):
    """An embedded player's manifest is only in its own frame's resource timing."""
    sniffer._session_root["F1"] = "T1"
    sniffer._session_root["OTHER"] = "T9"
    sniffer.answers["Runtime.evaluate"] = {"result": {"value": 1}}
    await sniffer._recover_manifests(sniffer.tabs["T1"])
    asked = {sid for m, _, sid in sniffer.sent if m == "Runtime.evaluate"}
    assert asked == {"S1", "F1"}


@pytest.mark.asyncio
async def test_large_xhr_body_is_streamed_and_kept_whole(sniffer, monkeypatch):
    """Too big for the browser to keep, but a manifest segment is one piece."""
    from core.sniffer import cdp

    monkeypatch.setattr(cdp, "_LARGE_BODY_BYTES", 3)
    monkeypatch.setattr(cdp, "_STREAM_FLUSH_BYTES", 2)
    sniffer.answers["Network.streamResourceContent"] = {"bufferedData": b64(b"AB")}
    sniffer._on_event("Network.responseReceived",
                      response_event(rtype="XHR", status=200,
                                     headers={"Content-Length": "6"}), "S1")
    await drain(sniffer)
    for chunk in (b"CD", b"EF"):
        sniffer._on_event("Network.dataReceived", {"requestId": "R1", "data": b64(chunk)}, "S1")
    assert sniffer.store.responses == []  # nothing handed over in pieces
    sniffer._on_event("Network.loadingFinished", {"requestId": "R1"}, "S1")
    await drain(sniffer)

    (resp,) = sniffer.store.responses
    assert resp.body == b"ABCDEF" and resp.status == 200
    assert "content-range" not in resp.headers


@pytest.mark.asyncio
async def test_failed_large_xhr_is_dropped(sniffer, monkeypatch):
    from core.sniffer import cdp

    monkeypatch.setattr(cdp, "_LARGE_BODY_BYTES", 3)
    sniffer.answers["Network.streamResourceContent"] = {"bufferedData": b64(b"AB")}
    sniffer._on_event("Network.responseReceived",
                      response_event(rtype="XHR", status=200,
                                     headers={"Content-Length": "6"}), "S1")
    await drain(sniffer)
    sniffer._on_event("Network.loadingFailed", {"requestId": "R1"}, "S1")
    await drain(sniffer)
    assert sniffer.store.responses == []


class TestPlaybackLevel:
    @pytest.mark.asyncio
    async def test_player_properties_give_the_playback_level(self, sniffer):
        """Properties arrive a few at a time; the level is known once they show it."""
        sniffer._on_event("Media.playerPropertiesChanged", {"playerId": "p1", "properties": [
            {"name": "kRendererName", "value": "RendererImpl"}]}, "S1")
        await drain(sniffer)
        assert sniffer.store.playback == []
        sniffer._on_event("Media.playerPropertiesChanged", {"playerId": "p1", "properties": [
            {"name": "kIsAudioDecryptingDemuxerStream", "value": "true"}]}, "S1")
        await drain(sniffer)
        assert sniffer.store.playback == [(sniffer.tabs["T1"].item, "software",
            "audio decrypted by the software CDM (DecryptingDemuxerStream)")]

    @pytest.mark.asyncio
    async def test_frames_enable_the_media_domain(self, sniffer):
        await sniffer._configure("F9", frame=True, resume=False)
        assert ("Media.enable", "F9") in [(m, sid) for m, _, sid in sniffer.sent]


class TestComplete:
    """Missing pieces are fetched by the page, in a frame that played the item."""

    def played_from(self, sniffer, session):
        sniffer._session_root[session] = "T1"
        sniffer._on_event("Network.responseReceived",
                          response_event(request_id=session, rtype="XHR", status=200,
                                         headers={}), session)

    @pytest.mark.asyncio
    async def test_fetches_in_the_frame_that_played_it(self, sniffer):
        self.played_from(sniffer, "F1")
        item = sniffer.tabs["T1"].item
        sniffer.store.plans = [[FetchRequest("https://cdn/a.mp4", (0, 99)),
                                FetchRequest("https://cdn/seg7.ts")]]
        sniffer.answers["Runtime.evaluate"] = {"result": {"value": 2}}

        assert await sniffer.complete(item, 100.0) == 2

        (sent,) = [(p, sid) for m, p, sid in sniffer.sent if m == "Runtime.evaluate"]
        params, sid = sent
        assert sid == "F1" and params["awaitPromise"] is True
        assert '"u": "https://cdn/a.mp4", "r": "bytes=0-99"' in params["expression"]
        assert '"u": "https://cdn/seg7.ts", "r": null' in params["expression"]

    @pytest.mark.asyncio
    async def test_rounds_continue_while_pieces_arrive(self, sniffer):
        """A file of unknown size: the first round learns it, the second fetches the rest."""
        self.played_from(sniffer, "S1")
        sniffer.store.plans = [[FetchRequest("https://cdn/a.mp4", (0, 9))],
                               [FetchRequest("https://cdn/a.mp4", (10, 19))]]
        sniffer.answers["Runtime.evaluate"] = {"result": {"value": 1}}
        assert await sniffer.complete(sniffer.tabs["T1"].item, 100.0) == 2

    @pytest.mark.asyncio
    async def test_a_frame_that_went_away_is_skipped(self, sniffer):
        self.played_from(sniffer, "F1")
        self.played_from(sniffer, "F2")
        del sniffer._session_root["F2"]  # detached since
        sniffer.store.plans = [[FetchRequest("https://cdn/seg1.ts")]]
        sniffer.answers["Runtime.evaluate"] = {"result": {"value": 1}}
        await sniffer.complete(sniffer.tabs["T1"].item, 100.0)
        assert [sid for m, _, sid in sniffer.sent if m == "Runtime.evaluate"] == ["F1"]

    @pytest.mark.asyncio
    async def test_no_page_left_to_ask(self, sniffer):
        sniffer.store.plans = [[FetchRequest("https://cdn/seg1.ts")]]
        assert await sniffer.complete("//gone.tv/page", 100.0) == 0
        assert not any(m == "Runtime.evaluate" for m, _, _ in sniffer.sent)

    @pytest.mark.asyncio
    async def test_nothing_missing_asks_nothing(self, sniffer):
        assert await sniffer.complete(sniffer.tabs["T1"].item, 100.0) == 0
        assert sniffer.sent == []


@pytest.mark.asyncio
async def test_media_keys_mark_the_page_drm(sniffer):
    sniffer._on_event("Runtime.bindingCalled", {"name": "__audiodaemonEme", "payload": ""}, "S1")
    assert sniffer.store.drm == [sniffer.tabs["T1"].item]


@pytest.mark.asyncio
async def test_media_keys_say_which_drm_and_level(sniffer):
    """The hook reports the key system and the robustness the browser granted."""
    payload = json.dumps({"keySystem": "com.widevine.alpha",
                          "video": ["SW_SECURE_DECODE"], "audio": ["SW_SECURE_CRYPTO"]})
    sniffer._on_event("Runtime.bindingCalled",
                      {"name": "__audiodaemonEme", "payload": payload}, "S1")
    assert sniffer.store.systems == ("Widevine L3 (SW_SECURE_DECODE)",)


@pytest.mark.asyncio
async def test_an_unreadable_eme_report_still_marks_drm(sniffer):
    sniffer._on_event("Runtime.bindingCalled",
                      {"name": "__audiodaemonEme", "payload": "{not json"}, "S1")
    assert sniffer.store.drm and sniffer.store.systems == ()


@pytest.mark.asyncio
async def test_unwanted_auto_attached_targets_are_released(sniffer):
    sniffer._on_event("Target.attachedToTarget", {
        "sessionId": "UI1", "waitingForDebugger": True,
        "targetInfo": {"targetId": "B1", "type": "browser_ui", "url": "chrome://x"}}, None)
    await drain(sniffer)
    methods = [(m, sid) for m, _, sid in sniffer.sent]
    assert ("Runtime.runIfWaitingForDebugger", "UI1") in methods
    assert ("Target.detachFromTarget", None) in methods


@pytest.mark.asyncio
async def test_new_page_is_configured_before_it_runs(sniffer):
    sniffer._on_event("Target.attachedToTarget", {
        "sessionId": "S2", "waitingForDebugger": True,
        "targetInfo": {"targetId": "T2", "type": "page", "url": "https://new.tv/", "title": ""}}, None)
    await drain(sniffer)
    methods = [m for m, _, sid in sniffer.sent if sid == "S2"]
    assert methods.index("Network.enable") < methods.index("Runtime.runIfWaitingForDebugger")
    assert sniffer.tabs["T2"].session_id == "S2"


@pytest.mark.asyncio
async def test_iframe_traffic_belongs_to_its_page(sniffer):
    sniffer._on_event("Target.attachedToTarget", {
        "sessionId": "F1", "waitingForDebugger": True,
        "targetInfo": {"targetId": "X", "type": "iframe", "url": "https://player.cdn/"}}, "S1")
    await drain(sniffer)
    sniffer.answers["Network.getResponseBody"] = {"body": b64(b"x"), "base64Encoded": True}
    sniffer._on_event("Network.responseReceived",
                      response_event(rtype="XHR", status=200, headers={}), "F1")
    sniffer._on_event("Network.loadingFinished", {"requestId": "R1"}, "F1")
    await drain(sniffer)
    assert sniffer.store.responses[0].page_url == "https://site.tv/watch/1"


def playing_in(*sessions):
    """Answer the audible-media probe: true only in these sessions."""
    def answer(params, session_id):
        return {"result": {"value": session_id in sessions}}
    return answer


class TestMatchTab:
    @pytest.mark.asyncio
    async def test_title_match_wins(self, sniffer):
        other = TabInfo("T9", url="https://b.tv/", title="Something else", last_media=time.time())
        sniffer.tabs["T9"] = other
        assert (await sniffer.match_tab("A Film")).target_id == "T1"

    @pytest.mark.asyncio
    async def test_the_playing_one_of_two_titled_tabs(self, sniffer):
        """The same video open twice: the one making sound is the play."""
        sniffer.tabs["T9"] = TabInfo("T9", title="A Film - Site", session_id="S9",
                                     last_media=time.time())
        sniffer._session_root["S9"] = "T9"
        sniffer.answers["Runtime.evaluate"] = playing_in("S1")
        assert (await sniffer.match_tab("A Film")).target_id == "T1"

    @pytest.mark.asyncio
    async def test_media_playing_in_an_iframe_counts_for_its_tab(self, sniffer):
        sniffer._session_root["F1"] = "T1"
        sniffer.tabs["T9"] = TabInfo("T9", title="x", last_media=time.time())
        sniffer.answers["Runtime.evaluate"] = playing_in("F1")
        assert (await sniffer.match_tab("Unrelated title")).target_id == "T1"

    @pytest.mark.asyncio
    async def test_most_recent_media_otherwise(self, sniffer):
        sniffer.tabs["T9"] = TabInfo("T9", title="x", last_media=time.time())
        assert (await sniffer.match_tab("Unrelated title")).target_id == "T9"

    @pytest.mark.asyncio
    async def test_stale_media_is_not_a_match(self, sniffer):
        sniffer.tabs["T1"].last_media = time.time() - 600
        assert await sniffer.match_tab("Unrelated title") is None

    @pytest.mark.asyncio
    async def test_disconnected(self, sniffer):
        sniffer.connected = False
        assert await sniffer.match_tab("A Film") is None


@pytest.mark.asyncio
async def test_request_finishing_before_streaming_starts(sniffer):
    """The media request completed before we could ask to stream it: its
    body is fetched instead, and nothing trips over the missed stream."""
    sniffer.answers["Network.getResponseBody"] = {"body": b64(b"all"), "base64Encoded": True}
    sniffer._on_event("Network.responseReceived", response_event(), "S1")
    sniffer._on_event("Network.loadingFinished", {"requestId": "R1"}, "S1")
    await drain(sniffer)
    assert [r.body for r in sniffer.store.responses] == [b"all"]
