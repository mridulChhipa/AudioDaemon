import asyncio
import logging
import time
from dataclasses import dataclass

from winrt.windows.media.control import (
    GlobalSystemMediaTransportControlsSessionManager as SessionManager,
)
from winrt.windows.media.control import (
    GlobalSystemMediaTransportControlsSessionPlaybackStatus as PlaybackStatus,
)

from core import database, encoder
from core.agent import CurationAgent, TrackVerdict, route
from core.config import POLL_SECONDS
from core.database import MUSIC, VIDEO
from core.pipeline import MediaJob, Pipeline, already_owned
from core.sniffer import detect
from core.sniffer.cdp import Sniffer, TabInfo

log = logging.getLogger(__name__)

STREAM_PROBE_SECONDS = 1.5
_PURGE_EVERY_SECONDS = 600

_REMATCH_SECONDS = 5.0

_BROWSER_APPS = ("chrome", "comet", "msedge", "edge", "brave", "opera", "vivaldi",
                 "chromium", "arc", "perplexity", "yandex", "thorium")


def is_browser(app_id: str) -> bool:
    app = (app_id or "").lower()
    return any(name in app for name in _BROWSER_APPS)


def app_name(app_id: str) -> str:
    """A readable name for the classifier, from an SMTC app id.

    SpotifyAB.SpotifyMusic_zpdnekdrzrea0!Spotify -> Spotify
    Comet.OOYZE3HLEZSNWAR6L5FEVE3ZKE              -> Comet
    """
    app = (app_id or "").split("!")[-1]
    return app.split(".")[0] if app else ""


@dataclass
class Snapshot:
    """One reading of the current media session."""

    artist: str
    title: str
    album: str
    album_artist: str
    track_number: int
    playing: bool
    duration: float
    app_id: str
    info: object  # raw media properties, for cover art

    @property
    def key(self) -> tuple[str, str]:
        return (self.artist, self.title)


async def read_session() -> Snapshot | None:
    """The current media session, or None if nothing is going on."""
    manager = await SessionManager.request_async()
    session = manager.get_current_session()
    if session is None:
        return None

    try:
        playing = session.get_playback_info().playback_status == PlaybackStatus.PLAYING
        timeline = session.get_timeline_properties()
        info = await session.try_get_media_properties_async()
    except OSError:
        # Sessions can vanish between any two of the calls above.
        return None

    return Snapshot(
        artist=(info.artist or "").strip(),
        title=(info.title or "").strip(),
        album=(info.album_title or "").strip(),
        album_artist=(info.album_artist or "").strip(),
        track_number=int(info.track_number or 0),
        playing=playing,
        duration=timeline.end_time.total_seconds(),
        app_id=session.source_app_user_model_id or "",
        info=info,
    )


@dataclass
class Decision:
    verdict: TrackVerdict
    has_video: bool | None


class Listener:
    """idle -> playing -> idle, driven by the SMTC poll.

    A play is in progress while `_snapshot` is set: its latest reading, the
    tab it plays in, the decision task, and the decision once it lands.
    """

    def __init__(self, sniffer: Sniffer, pipeline: Pipeline,
                 agent: CurationAgent | None = None):
        self._sniffer = sniffer
        self._store = sniffer.store
        self._pipeline = pipeline
        self._agent = agent or CurationAgent()

        self._snapshot: Snapshot | None = None
        self._tab: TabInfo | None = None
        self._deciding: asyncio.Task | None = None
        self._decision: Decision | None = None
        # Ruled out already, so each poll doesn't re-ask the model.
        self._rejected: set[tuple[str, str]] = set()
        # The play last found in no captured tab, and when it was looked for.
        # It is looked for again now and then -- remote debugging may be
        # turned on, or the play moved into the browser -- but logged once.
        self._unmatched: tuple[str, str] | None = None
        self._unmatched_at = 0.0

    async def run(self) -> None:
        self._pipeline.start()
        log.info("Listening for media sessions (every %ss)", POLL_SECONDS)
        last_purge = 0.0
        while True:
            try:
                await self._tick()
            except Exception:
                log.exception("Poll failed; continuing")
            if time.time() - last_purge > _PURGE_EVERY_SECONDS:
                last_purge = time.time()
                purged = await asyncio.to_thread(self._store.purge)
                if purged:
                    log.info("Purged %d unclaimed or stale stream item(s)", purged)
            await asyncio.sleep(POLL_SECONDS)

    # -- the state machine -------------------------------------------------

    async def _tick(self) -> None:
        snap = await read_session()

        if self._snapshot is None:
            if snap and snap.playing:
                await self._maybe_start(snap)
            return

        if snap is None or not snap.playing:
            await self._close("playback stopped")
            return

        if snap.key != self._snapshot.key:
            await self._close("track changed")
            await self._maybe_start(snap)
            return

        self._snapshot = snap
        self._take_decision()

    def _take_decision(self) -> None:
        """Consume the background decision if it has landed."""
        if self._deciding is None or not self._deciding.done():
            return

        task, self._deciding = self._deciding, None
        try:
            decision = task.result()
        except Exception:
            log.exception("Deciding what this is failed")
            self._reset()
            return

        if decision is None:
            self._reset()  # _decide logged why
            return

        self._decision = decision
        verdict = decision.verdict
        kind = "music video" if verdict.is_music and decision.has_video else (
            "music" if verdict.is_music else "video")
        log.info("Capturing %s: %s - %s", kind, verdict.clean_artist, verdict.clean_title)

    # -- transitions -------------------------------------------------------

    async def _maybe_start(self, snap: Snapshot) -> None:
        if snap.duration <= 0:
            return  # no length: neither completeness nor the ad filter can be judged
        if snap.key in self._rejected:
            return
        browser = is_browser(snap.app_id)
        if browser and self._sniffer.connecting:
            return  # the browser is being asked; its tabs are known once it answers
        if snap.key == self._unmatched and (
                not browser or time.monotonic() - self._unmatched_at < _REMATCH_SECONDS):
            return

        tab = await self._sniffer.match_tab(snap.title) if browser else None
        if tab is None:
            if snap.key != self._unmatched:
                log.info("Not archived: %s - %s is not playing in a captured browser tab (%s)",
                         snap.artist, snap.title, app_name(snap.app_id) or "unknown player")
            self._unmatched, self._unmatched_at = snap.key, time.monotonic()
            return

        self._unmatched = None
        self._snapshot = snap
        self._tab = tab
        self._decision = None
        self._deciding = asyncio.create_task(self._decide(snap, tab))

    async def _decide(self, snap: Snapshot, tab: TabInfo) -> Decision | None:
        """Classify and dedup, while the play goes on."""
        artist, title = snap.artist, snap.title

        if not title:
            log.info("Not kept: the player published no title")
            return None

        await asyncio.sleep(STREAM_PROBE_SECONDS)
        item = tab.item
        summary = await asyncio.to_thread(self._store.summary, item)
        if summary.drm and not summary.clear:
            log.info("Not archived (DRM-protected: %s): %s - %s",
                     ", ".join(summary.drm_systems) or "DRM system not named", artist, title)
            self._rejected.add(snap.key)
            return None
        has_video = summary.has_video if summary.streams else None

        verdict = await self._agent.classify(artist, title, detect.site_name(tab.url), has_video)
        if verdict is None or not verdict.is_wanted:
            log.info("Not kept (%s): %s - %s",
                     verdict.type if verdict else "no verdict", artist, title)
            self._rejected.add(snap.key)
            return None

        want_music, want_video = route(verdict, bool(has_video))
        music_owned = want_music and await already_owned(
            database.make_track_hash(verdict.clean_artist, verdict.clean_title), MUSIC)
        video_owned = want_video and await already_owned(database.make_video_hash(item), VIDEO)
        if (not want_music or music_owned) and (not want_video or video_owned):
            log.info("Already in the library: %s - %s", verdict.clean_artist,
                     verdict.clean_title)
            self._rejected.add(snap.key)
            return None

        await asyncio.to_thread(self._store.claim, item, verdict.clean_title)
        return Decision(verdict, has_video)

    async def _close(self, reason: str) -> None:
        """Hand the finished play to the pipeline."""
        # The play is over, so a decision still in flight can be waited on.
        if self._deciding is not None and not self._deciding.done():
            try:
                await asyncio.wait_for(asyncio.shield(self._deciding), timeout=20)
            except Exception:
                pass
        self._take_decision()

        snap, tab, decision = self._snapshot, self._tab, self._decision
        self._reset()
        if snap is None:
            return  # the decision said no

        log.info("Play ended (%s): %s", reason, snap.title)
        if decision is None:
            return

        self._pipeline.submit(MediaJob(
            verdict=decision.verdict,
            album=snap.album,
            album_artist=snap.album_artist,
            track_number=snap.track_number,
            duration=snap.duration,
            item=tab.item,
            site=detect.site_name(tab.url),
            has_video=decision.has_video,
            cover=await encoder.read_session_cover(snap.info),
            source=snap.app_id,
        ))

    def _reset(self) -> None:
        self._snapshot = None
        self._tab = None
        self._decision = None
        if self._deciding is not None:
            self._deciding.cancel()
            self._deciding = None
