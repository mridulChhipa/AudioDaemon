"""The application-layer tap: response bodies from the browser, over CDP.

Connects to the Chromium browser you already use, on your own profile, once
remote debugging is turned on at chrome://inspect/#remote-debugging (the
browser asks you to allow the connection), attaches to every page -- and, through auto-attach,
to every iframe and worker inside it, so an embedded player on any site is
seen too -- and hands each finished media response to the StreamStore.

Bodies are fetched with `Network.getResponseBody` once `loadingFinished`
fires. That is decrypted, de-chunked, decompressed HTTP payload: TLS, HTTP/2
and QUIC are already behind us.

When a play ends with its streams incomplete, `complete()` has the page itself
fetch the missing pieces -- with its own cookies, through the same Network
domain -- so they arrive here like any other response.

Without the browser nothing is captured; the daemon waits for it to appear.
"""
import asyncio
import base64
import itertools
import json
import logging
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from core.config import (
    BROWSER_DATA_DIRS,
    CDP_APPROVAL_SECONDS,
    CDP_MAX_RESOURCE_BUFFER,
    CDP_MAX_TOTAL_BUFFER,
    COMPLETION_ROUNDS,
)
from core.sniffer import detect
from core.sniffer.streams import UNPLACED, FetchRequest, Response, StreamStore

log = logging.getLogger(__name__)

_EME_BINDING = "__audiodaemonEme"

_EME_HOOK = f"""
(() => {{
  if (window.__audiodaemonEmeHooked || !window.HTMLMediaElement) return;
  window.__audiodaemonEmeHooked = true;
  const systems = new WeakMap();
  const Access = window.MediaKeySystemAccess;
  if (Access && Access.prototype.createMediaKeys) {{
    const create = Access.prototype.createMediaKeys;
    Access.prototype.createMediaKeys = function () {{
      const granted = {{ keySystem: this.keySystem, video: [], audio: [] }};
      try {{
        const config = this.getConfiguration();
        granted.video = (config.videoCapabilities || []).map(c => c.robustness || '');
        granted.audio = (config.audioCapabilities || []).map(c => c.robustness || '');
      }} catch (e) {{}}
      return create.call(this).then(keys => {{
        try {{ systems.set(keys, JSON.stringify(granted)); }} catch (e) {{}}
        return keys;
      }});
    }};
  }}
  const original = HTMLMediaElement.prototype.setMediaKeys;
  if (!original) return;
  HTMLMediaElement.prototype.setMediaKeys = function (keys) {{
    try {{
      if (keys && window.{_EME_BINDING}) window.{_EME_BINDING}(systems.get(keys) || '');
    }} catch (e) {{}}
    return original.call(this, keys);
  }};
}})();
"""

# Manifests this page loaded, fetched again by the page (so with its own
# cookies and headers). Frames can't be reached from here; top-level players
# and same-origin frames are the common case.
_REFETCH_MANIFESTS = r"""
(() => {
  const urls = [...new Set(performance.getEntriesByType('resource')
    .map(e => e.name).filter(n => /\.(m3u8|mpd)(\?|#|$)/i.test(n)))];
  urls.forEach(u => fetch(u, {credentials: 'include'}).catch(() => {}));
  return urls.length;
})()
"""
_RECOVERY_INTERVAL = 20.0

# Whether this frame has a media element audibly playing. Workers have no
# document, and answer false.
_PLAYING = """
(() => typeof document !== 'undefined' &&
  [...document.querySelectorAll('video, audio')]
    .some(m => !m.paused && !m.muted && m.volume > 0 && m.readyState > 2))()
"""

# Fetch the listed pieces from the page, a few at a time. Same-origin
# credentials first (what a CDN with `Access-Control-Allow-Origin: *` accepts),
# then with cookies for servers that want them. Returns how many succeeded.
_FETCH_PIECES = """
(async (jobs, width) => {
  let next = 0, ok = 0;
  const one = async (job, credentials) => {
    const r = await fetch(job.u, {credentials, headers: job.r ? {Range: job.r} : {}});
    await r.arrayBuffer();
    return r.ok;
  };
  const worker = async () => {
    while (next < jobs.length) {
      const job = jobs[next++];
      try { if (await one(job, 'same-origin')) { ok++; continue; } } catch (e) {}
      try { if (await one(job, 'include')) ok++; } catch (e) {}
    }
  };
  await Promise.all(Array.from({length: width}, worker));
  return ok;
})(%s, %d)
"""
_FETCH_WIDTH = 4


class CdpError(Exception):
    pass


def _eme_label(payload: str) -> str | None:
    try:
        granted = json.loads(payload) if payload else {}
    except ValueError:
        return None
    if not isinstance(granted, dict):
        return None
    return detect.eme_drm_label(str(granted.get("keySystem") or ""),
                                list(granted.get("video") or []),
                                list(granted.get("audio") or []))


_ACTIVE_PORT_FILE = "DevToolsActivePort"
# How often to look for the browser while it isn't reachable, and the longest
# wait before asking again at an endpoint that already said no.
_LOOK_SECONDS = 5.0
_MAX_BACKOFF_SECONDS = 300.0


def read_active_port(data_dir: Path) -> tuple[str, float] | None:
    """The browser endpoint a User Data directory advertises, and when.

    `DevToolsActivePort` holds the port on its first line and the browser
    target's websocket path on its second. Returns (ws URL, file mtime), or
    None if remote debugging isn't on for that browser.
    """
    path = Path(data_dir) / _ACTIVE_PORT_FILE
    try:
        lines = path.read_text("utf-8").split()
        mtime = path.stat().st_mtime
    except OSError:
        return None
    if len(lines) < 2 or not lines[0].isdigit() or not lines[1].startswith("/"):
        return None
    return f"ws://127.0.0.1:{int(lines[0])}{lines[1]}", mtime


def find_browser(data_dirs: list[Path]) -> str | None:
    """The websocket URL of the most recently started debuggable browser."""
    found = [e for e in (read_active_port(d) for d in data_dirs) if e is not None]
    return max(found, key=lambda e: e[1])[0] if found else None


@dataclass
class TabInfo:
    target_id: str
    url: str = ""
    title: str = ""
    session_id: str | None = None
    last_media: float = 0.0     # when a media response last arrived for it

    @property
    def item(self) -> str:
        return detect.page_id(self.url)


@dataclass
class _Pending:
    url: str
    status: int
    mime: str
    headers: dict[str, str]
    resource_type: str
    # Streaming (progressive <video>/<audio> loads, and bodies too large for
    # the browser to keep): bytes as they arrive.
    streaming: bool = False
    whole: bool = False            # hand over only once finished, as one body
    ready: bool = False            # streamResourceContent has answered
    parts: list[bytes] = field(default_factory=list)
    buffered: int = 0
    offset: int = 0                # file offset of parts[0]
    total: int | None = None
    page_url: str = ""


# Media elements fetch progressive files with open-ended ranges and cancel
# them whenever their buffer is full, so those responses never "finish" and
# getResponseBody has nothing to give. Their bytes are streamed instead, and
# flushed to the store in pieces of this size.
_STREAM_FLUSH_BYTES = 8 * 1024 * 1024
# Other responses larger than this would be evicted before getResponseBody
# could ask, so they are streamed too -- but kept whole, since a segment
# placed by its manifest can't be stored in pieces.
_LARGE_BODY_BYTES = CDP_MAX_RESOURCE_BUFFER // 2


@dataclass
class Sniffer:
    store: StreamStore
    data_dirs: list[Path] = field(default_factory=lambda: list(BROWSER_DATA_DIRS))
    tabs: dict[str, TabInfo] = field(default_factory=dict)
    stats: Counter = field(default_factory=Counter)
    connected: bool = False
    connecting: bool = False   # waiting on the handshake -- often on the user's Allow

    def __post_init__(self) -> None:
        self._ws = None
        self._ids = itertools.count(1)
        self._waiters: dict[int, asyncio.Future] = {}
        self._session_root: dict[str, str] = {}   # session -> top page target id
        self._pending: dict[tuple[str, str], _Pending] = {}
        self._tasks: set[asyncio.Task] = set()
        self._recovered: dict[str, float] = {}      # tab -> last manifest recovery
        # item -> the sessions (frames) that fetched its media, latest last:
        # where a later fetch of its missing pieces will be allowed.
        self._item_sessions: dict[str, list[str]] = {}
        # (session, player id) -> the media player's properties so far.
        self._players: dict[tuple[str | None, str], dict[str, str]] = {}

    # -- lifecycle ---------------------------------------------------------

    async def run(self) -> None:
        """Stay connected to the browser for as long as the daemon runs.

        The browser is looked for every few seconds. Each connection makes it
        ask you for permission, so an endpoint that refused (or went
        unanswered) is asked again only after a growing wait -- at once if the
        browser restarts and advertises a new one.
        """
        said_off = False
        failed: str | None = None
        backoff, retry_at = _LOOK_SECONDS, 0.0
        while True:
            ws_url = await asyncio.to_thread(find_browser, self.data_dirs)
            if ws_url is None:
                if not said_off:
                    log.info("Remote debugging isn't on in the browser; network capture is "
                             "paused. Turn it on at chrome://inspect/#remote-debugging "
                             "(comet://inspect/#remote-debugging in Comet)")
                    said_off = True
            elif ws_url != failed or time.monotonic() >= retry_at:
                said_off = False
                try:
                    await self._connect_once(ws_url)
                    failed, backoff = None, _LOOK_SECONDS
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    if ws_url != failed:
                        backoff = _LOOK_SECONDS
                        log.info("The browser didn't let the daemon connect (%s); allow it "
                                 "in the browser's prompt -- it will ask again",
                                 type(exc).__name__)
                    failed = ws_url
                    retry_at = time.monotonic() + backoff
                    backoff = min(backoff * 2, _MAX_BACKOFF_SECONDS)
            await asyncio.sleep(_LOOK_SECONDS)

    async def _connect_once(self, ws_url: str) -> None:
        from websockets.asyncio.client import connect

        log.info("Connecting to the browser; allow it in the browser's prompt if asked")
        # The handshake waits while the browser asks you, so give it time.
        self.connecting = True
        try:
            ws = await connect(ws_url, max_size=None, ping_interval=None,
                               open_timeout=CDP_APPROVAL_SECONDS)
        finally:
            self.connecting = False
        async with ws:  # closes the connection on the way out
            self._ws = ws
            self.connected = True
            log.info("Connected to the browser (%s)", ws_url.split("/devtools")[0])
            try:
                reader = asyncio.create_task(self._read_loop(), name="cdp-reader")
                # New tabs are held paused until capture is configured in them,
                # so their first requests -- often the manifest -- aren't missed.
                await self.send("Target.setAutoAttach", {
                    "autoAttach": True, "waitForDebuggerOnStart": True, "flatten": True})
                # Tabs that already exist are announced here and attached to.
                await self.send("Target.setDiscoverTargets", {"discover": True})
                await reader
            finally:
                self._teardown()
        log.info("Browser connection closed")

    def _teardown(self) -> None:
        self.connected = False
        self._ws = None
        for fut in self._waiters.values():
            if not fut.done():
                fut.set_exception(CdpError("connection closed"))
        self._waiters.clear()
        self._session_root.clear()
        self._pending.clear()
        self._players.clear()
        self.tabs.clear()

    # -- transport ---------------------------------------------------------

    async def send(self, method: str, params: dict | None = None,
                   session_id: str | None = None, timeout: float = 30.0) -> dict:
        if self._ws is None:
            raise CdpError("not connected")
        msg_id = next(self._ids)
        message = {"id": msg_id, "method": method, "params": params or {}}
        if session_id:
            message["sessionId"] = session_id
        fut = asyncio.get_running_loop().create_future()
        self._waiters[msg_id] = fut
        await self._ws.send(json.dumps(message))
        try:
            return await asyncio.wait_for(fut, timeout)
        finally:
            self._waiters.pop(msg_id, None)

    async def _read_loop(self) -> None:
        async for raw in self._ws:
            msg = json.loads(raw)
            if "id" in msg:
                fut = self._waiters.get(msg["id"])
                if fut is not None and not fut.done():
                    if "error" in msg:
                        fut.set_exception(CdpError(msg["error"].get("message", "error")))
                    else:
                        fut.set_result(msg.get("result", {}))
                continue
            try:
                self._on_event(msg.get("method", ""), msg.get("params", {}),
                               msg.get("sessionId"))
            except Exception:
                log.warning("CDP event handler failed for %s", msg.get("method"),
                            exc_info=True)

    def _spawn(self, coro) -> None:
        task = asyncio.create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    # -- events ------------------------------------------------------------

    def _on_event(self, method: str, params: dict, session_id: str | None) -> None:
        """Bookkeeping stays synchronous so responseReceived is always recorded
        before its loadingFinished is handled; only I/O is spawned."""
        if method in ("Target.targetCreated", "Target.targetInfoChanged"):
            info = params["targetInfo"]
            if info.get("type") != "page":
                return
            tab = self.tabs.get(info["targetId"])
            if tab is None:
                tab = self.tabs[info["targetId"]] = TabInfo(info["targetId"])
                self._spawn(self._attach_page(tab))
            tab.url, tab.title = info.get("url", tab.url), info.get("title", tab.title)

        elif method == "Target.targetDestroyed":
            self.tabs.pop(params.get("targetId"), None)

        elif method == "Target.attachedToTarget":
            child = params["sessionId"]
            info = params["targetInfo"]
            if session_id is None:
                # Browser-level: a page auto-attached (new tab, paused) or the
                # result of our own attachToTarget on an existing one.
                if child in self._session_root:
                    return
                if info.get("type") != "page":
                    # browser_ui, service workers of other origins...: auto-attach
                    # paused them, and a paused browser_ui is a frozen browser.
                    self._spawn(self._release(child, params.get("waitingForDebugger", False)))
                    return
                tab = self.tabs.setdefault(info["targetId"], TabInfo(info["targetId"]))
                tab.url, tab.title = info.get("url", tab.url), info.get("title", tab.title)
                if tab.session_id is None:
                    tab.session_id = child
                self._session_root[child] = tab.target_id
                self._spawn(self._configure(child, frame=True,
                                            resume=params.get("waitingForDebugger", False)))
                return
            root = self._session_root.get(session_id)
            if root is None:
                return
            self._session_root[child] = root
            self._spawn(self._configure(child, frame=info.get("type") == "iframe",
                                        resume=True))

        elif method == "Target.detachedFromTarget":
            self._session_root.pop(params.get("sessionId"), None)

        elif method == "Network.responseReceived":
            resp = params.get("response", {})
            url = resp.get("url", "")
            if not url.startswith("http"):
                return
            if not detect.is_candidate(url, resp.get("mimeType"), params.get("type")):
                return
            tab = self._tab_for(session_id)
            pending = _Pending(
                url=url,
                status=int(resp.get("status", 0)),
                mime=resp.get("mimeType", ""),
                headers={k.lower(): v for k, v in (resp.get("headers") or {}).items()},
                resource_type=params.get("type", ""),
                page_url=tab.url if tab else "",
            )
            key = (session_id, params["requestId"])
            self._pending[key] = pending
            if tab is not None:
                tab.last_media = time.time()
                self._note_session(tab.item, session_id)
            if not 200 <= pending.status < 300:
                return
            cr = detect.content_range(pending.headers.get("content-range"))
            length = pending.headers.get("content-length", "")
            if pending.resource_type == "Media" and (cr or pending.status == 200):
                pending.streaming = True
                pending.offset = cr[0] if cr else 0
                pending.total = cr[2] if cr else (int(length) if length.isdigit() else None)
                self._spawn(self._start_streaming(session_id, params["requestId"], pending))
            elif length.isdigit() and int(length) > _LARGE_BODY_BYTES:
                pending.streaming = pending.whole = True
                self._spawn(self._start_streaming(session_id, params["requestId"], pending))

        elif method == "Network.dataReceived":
            pending = self._pending.get((session_id, params.get("requestId")))
            data = params.get("data")
            if pending is not None and pending.streaming and data:
                chunk = base64.b64decode(data)
                pending.parts.append(chunk)
                pending.buffered += len(chunk)
                if (pending.ready and not pending.whole
                        and pending.buffered >= _STREAM_FLUSH_BYTES):
                    self._flush(pending)

        elif method in ("Network.loadingFinished", "Network.loadingFailed"):
            pending = self._pending.pop((session_id, params.get("requestId")), None)
            if pending is None:
                return
            if pending.streaming and pending.ready and pending.whole:
                # Half a segment is no segment; a failed one is refetched by
                # complete() if it's needed.
                if method == "Network.loadingFinished":
                    body = b"".join(pending.parts)
                    self._spawn(self._ingest(session_id, pending, body))
            elif pending.streaming and pending.ready:
                # Everything that arrived is valid at its offset, even when the
                # player cancelled the rest.
                self._flush(pending)
            elif method == "Network.loadingFinished":
                self._spawn(self._fetch_body(session_id, params["requestId"], pending))

        elif method == "Media.playerPropertiesChanged":
            self._on_player_properties(session_id, params)

        elif method == "Runtime.bindingCalled" and params.get("name") == _EME_BINDING:
            tab = self._tab_for(session_id)
            if tab is not None:
                label = _eme_label(params.get("payload", ""))
                self.store.mark_drm(tab.item, "the page attached MediaKeys",
                                    (label,) if label else ())

    def _on_player_properties(self, session_id: str | None, params: dict) -> None:
        tab = self._tab_for(session_id)
        if tab is None:
            return
        props = self._players.setdefault((session_id, str(params.get("playerId"))), {})
        for prop in params.get("properties") or []:
            props[str(prop.get("name"))] = str(prop.get("value"))
        found = detect.playback_security(props)
        if found:
            self._spawn(asyncio.to_thread(self.store.mark_playback, tab.item, *found))

    def _tab_for(self, session_id: str | None) -> TabInfo | None:
        root = self._session_root.get(session_id) if session_id else None
        return self.tabs.get(root) if root else None

    def _note_session(self, item: str, session_id: str | None) -> None:
        if session_id is None:
            return
        sessions = self._item_sessions.setdefault(item, [])
        if sessions and sessions[-1] == session_id:
            return
        if session_id in sessions:
            sessions.remove(session_id)
        sessions.append(session_id)
        del sessions[:-8]

    def _frames(self, tab: TabInfo) -> list[str]:
        """Every attached session belonging to a tab: the page, its frames, workers."""
        return [sid for sid, root in self._session_root.items() if root == tab.target_id]

    async def _attach_page(self, tab: TabInfo) -> None:
        """Attach to a tab that existed before we connected.

        New tabs arrive through auto-attach instead; the pause lets that
        announcement land first so a tab isn't attached twice.
        """
        await asyncio.sleep(0.5)
        if tab.session_id is not None or tab.target_id not in self.tabs:
            return
        try:
            # Its attachedToTarget event registers and configures the session.
            await self.send("Target.attachToTarget",
                            {"targetId": tab.target_id, "flatten": True})
        except (CdpError, asyncio.TimeoutError) as exc:
            log.debug("Could not attach to %s: %s", tab.url[:80], exc)

    async def _release(self, session_id: str, waiting: bool) -> None:
        """Let go of a target we have no use for, resuming it if it's paused."""
        steps = [("Target.detachFromTarget", {"sessionId": session_id}, None)]
        if waiting:
            steps.insert(0, ("Runtime.runIfWaitingForDebugger", {}, session_id))
        for method, params, sid in steps:
            try:
                await self.send(method, params, sid, timeout=10)
            except (CdpError, asyncio.TimeoutError):
                pass

    async def _configure(self, session_id: str, *, frame: bool, resume: bool) -> None:
        """Turn on what we need in one target; each step may be unsupported."""
        steps = [
            ("Target.setAutoAttach", {"autoAttach": True, "waitForDebuggerOnStart": True,
                                      "flatten": True}),
            ("Network.enable", {"maxTotalBufferSize": CDP_MAX_TOTAL_BUFFER,
                                "maxResourceBufferSize": CDP_MAX_RESOURCE_BUFFER}),
        ]
        if frame:
            steps += [
                ("Runtime.addBinding", {"name": _EME_BINDING}),
                ("Page.addScriptToEvaluateOnNewDocument", {"source": _EME_HOOK}),
                ("Runtime.evaluate", {"expression": _EME_HOOK}),
                # Each media player's pipeline -- how it decrypts, so the
                # playback's DRM security level. Players that already exist
                # are reported too.
                ("Media.enable", {}),
            ]
        if resume:
            steps.append(("Runtime.runIfWaitingForDebugger", {}))
        for method, params in steps:
            try:
                await self.send(method, params, session_id, timeout=10)
            except (CdpError, asyncio.TimeoutError):
                pass  # workers have no Page domain, etc.

    async def _start_streaming(self, session_id: str, request_id: str, p: _Pending) -> None:
        """Switch a media request to streaming; fall back to the body if we can't.

        The browser sometimes doesn't know the request yet when its response
        event is already out ("Request ... does not exists"), so a few quick
        retries.
        """
        result, error = None, "it finished first"
        for delay in (0.0, 0.05, 0.2, 0.5, 1.0):
            if delay:
                await asyncio.sleep(delay)
            if (session_id, request_id) not in self._pending:
                break  # finished meanwhile; its body is what's left
            try:
                result = await self.send("Network.streamResourceContent",
                                         {"requestId": request_id}, session_id, timeout=10)
                break
            except (CdpError, asyncio.TimeoutError) as exc:
                error = exc
        if result is None:
            log.debug("Can't stream %s (%s); will ask for the body", p.url[:80], error)
            p.streaming = False
            p.parts.clear()
            p.buffered = 0
            return
        # What arrived before streaming began comes first; dataReceived events
        # carrying data can only have been sent after it.
        early = base64.b64decode(result.get("bufferedData") or "")
        if early:
            p.parts.insert(0, early)
            p.buffered += len(early)
        p.ready = True
        if p.whole:
            if (session_id, request_id) not in self._pending:
                # Finished while we were asking; its loadingFinished found it
                # not ready and asked for the body instead.
                p.parts.clear()
            return
        if (session_id, request_id) not in self._pending or p.buffered >= _STREAM_FLUSH_BYTES:
            self._flush(p)  # it already finished while we were asking

    def _flush(self, p: _Pending) -> None:
        """Hand what has streamed so far to the store, as a byte range."""
        if not p.parts:
            return
        data = b"".join(p.parts)
        p.parts.clear()
        p.buffered = 0
        first, last = p.offset, p.offset + len(data) - 1
        p.offset += len(data)
        headers = dict(p.headers)
        headers["content-range"] = f"bytes {first}-{last}/{p.total if p.total else '*'}"
        self.stats["bodies"] += 1
        self.stats["bytes"] += len(data)
        response = Response(page_url=p.page_url, url=p.url, status=206,
                            mime=p.mime, headers=headers, body=data)
        self._spawn(asyncio.to_thread(self.store.ingest, response))

    async def _fetch_body(self, session_id: str, request_id: str, p: _Pending) -> None:
        try:
            result = await self.send("Network.getResponseBody", {"requestId": request_id},
                                     session_id, timeout=60)
        except (CdpError, asyncio.TimeoutError) as exc:
            # The browser didn't keep it: evicted, streamed, or a detached frame.
            self.stats["missed"] += 1
            log.debug("No body for %s (%s)", p.url[:100], exc)
            return
        body = result.get("body", "")
        data = base64.b64decode(body) if result.get("base64Encoded") else body.encode("utf-8")
        await self._ingest(session_id, p, data)

    async def _ingest(self, session_id: str, p: _Pending, data: bytes) -> None:
        """Hand one whole response body to the store."""
        self.stats["bodies"] += 1
        self.stats["bytes"] += len(data)
        tab = self._tab_for(session_id)
        response = Response(
            page_url=tab.url if tab else "",
            url=p.url, status=p.status, mime=p.mime, headers=p.headers, body=data,
        )
        outcome = await asyncio.to_thread(self.store.ingest, response)
        if outcome == UNPLACED and tab is not None:
            await self._recover_manifests(tab)

    async def _recover_manifests(self, tab: TabInfo) -> None:
        """Have the page fetch its manifests again.

        A segment arrived with no manifest to place it: the page loaded its
        manifest before we attached. Its resource-timing list still names it,
        and a fetch from the page itself goes through the Network domain like
        any other response. Rate-limited, since every unplaced segment asks.
        """
        now = time.time()
        if now - self._recovered.get(tab.target_id, 0.0) < _RECOVERY_INTERVAL:
            return
        self._recovered[tab.target_id] = now
        # An embedded player loaded its manifest in its own frame, and only
        # that frame's resource timing names it.
        count = 0
        for sid in self._frames(tab):
            try:
                result = await self.send("Runtime.evaluate", {
                    "expression": _REFETCH_MANIFESTS, "returnByValue": True,
                }, sid, timeout=10)
            except (CdpError, asyncio.TimeoutError):
                continue
            value = (result.get("result") or {}).get("value")
            count += value if isinstance(value, int) else 0
        if count:
            log.info("Re-fetching %d manifest(s) the page loaded before capture began",
                     count)

    # -- what the listener and pipeline ask ---------------------------------

    async def match_tab(self, title: str, window: float = 30.0) -> TabInfo | None:
        """The tab a media session belongs to.

        Tabs whose title contains the session's title are preferred, and among
        them one with media audibly playing; failing a title match, the tab
        that is playing, then the one that most recently fetched media.
        """
        if not self.connected or not self.tabs:
            return None
        now = time.time()
        needle = (title or "").strip().casefold()
        titled = [t for t in self.tabs.values() if needle and (
            needle in t.title.casefold() or (t.title and t.title.casefold() in needle))]
        playing = await self._playing_tabs()
        recent = [t for t in self.tabs.values() if now - t.last_media <= window]
        for pool in ([t for t in titled if t.target_id in playing], titled,
                     [t for t in self.tabs.values() if t.target_id in playing], recent):
            if pool:
                return max(pool, key=lambda t: t.last_media)
        return None

    async def _playing_tabs(self) -> set[str]:
        """Target ids of the tabs with media audibly playing in any frame."""
        async def playing(sid: str) -> bool:
            try:
                result = await self.send("Runtime.evaluate",
                                         {"expression": _PLAYING, "returnByValue": True},
                                         sid, timeout=2)
            except (CdpError, asyncio.TimeoutError):
                return False
            return (result.get("result") or {}).get("value") is True

        sessions = list(self._session_root.items())
        answers = await asyncio.gather(*(playing(sid) for sid, _ in sessions))
        return {root for (_, root), yes in zip(sessions, answers) if yes}

    async def complete(self, item: str, duration: float) -> int:
        """Have the page fetch what the store still lacks for an item.

        Returns how many requests were made. The responses go through the
        Network domain like the player's own, so they land in the store by
        the usual path; this waits until they have.
        """
        made = 0
        for _ in range(COMPLETION_ROUNDS):
            requests = await asyncio.to_thread(self.store.missing, item, duration)
            if not requests:
                break
            log.info("Fetching %d missing piece(s) of %s from the page", len(requests), item)
            fetched = await self._fetch_from_page(item, requests)
            if fetched is None:
                log.info("No open page can fetch the rest of %s", item)
                break
            made += len(requests)
            await self._settle()
            if fetched == 0:
                break
        return made

    async def _fetch_from_page(self, item: str, requests: list[FetchRequest]) -> int | None:
        """Run the fetches in a frame that fetched this item's media before.

        Returns how many succeeded, or None if no such frame is attached.
        """
        jobs = json.dumps([{"u": r.url, "r": r.header} for r in requests])
        sessions = [sid for sid in reversed(self._item_sessions.get(item, []))
                    if sid in self._session_root]
        sessions += [t.session_id for t in self.tabs.values()
                     if t.item == item and t.session_id and t.session_id not in sessions]
        for sid in sessions:
            try:
                result = await self.send("Runtime.evaluate", {
                    "expression": _FETCH_PIECES % (jobs, _FETCH_WIDTH),
                    "awaitPromise": True, "returnByValue": True,
                }, sid, timeout=60 + 5 * len(requests))
            except (CdpError, asyncio.TimeoutError) as exc:
                log.debug("Could not fetch from session %s: %s", sid, exc)
                continue
            value = (result.get("result") or {}).get("value")
            return value if isinstance(value, int) else 0
        return None

    async def _settle(self, timeout: float = 60.0) -> None:
        """Wait for the bodies of just-finished requests to reach the store.

        Only what is in flight now is waited on: the next track may already
        be streaming into the same tab.
        """
        await asyncio.sleep(0.2)  # the last loadingFinished events
        deadline = time.monotonic() + timeout
        for _ in range(2):  # a body fetch may spawn its ingest
            tasks = [t for t in self._tasks if not t.done()]
            remaining = deadline - time.monotonic()
            if not tasks or remaining <= 0:
                return
            await asyncio.wait(tasks, timeout=remaining)
