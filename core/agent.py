"""The brain: decide what's playing -- music, a video, or neither -- and clean
up its labels.

The only LLM-driven step. Everything touching the filesystem or database is
ordinary Python elsewhere, so a bad verdict costs at most one wrongly kept or
discarded item. Media sessions publish plenty that is neither -- adverts, a
site's UI sounds, an Instagram reel titled "Instagram" -- and this keeps them
out.

JSON mode at temperature 0. The few-shot examples are load-bearing: without
them the model misses podcasts and leaves "| Artist" suffixes in place.
"""
import asyncio
import json
import logging
import re
from dataclasses import dataclass

from ollama import AsyncClient

from core.config import LLM_TIMEOUT_SECONDS, MODEL_ID, OLLAMA_HOST

log = logging.getLogger(__name__)

MUSIC, VIDEO, OTHER = "MUSIC", "VIDEO", "OTHER"
# Small models sometimes call music "SONG".
_ALIASES = {"SONG": MUSIC, "MUSIC": MUSIC, "VIDEO": VIDEO, "OTHER": OTHER}

_PROMPT = """Classify this media item and clean its title.

Rules:
1. "type" is "MUSIC" for songs and music tracks, including a song's music
   video. "VIDEO" for other videos: films, episodes, vlogs, tutorials,
   documentaries. "OTHER" for podcasts, interviews, ads.
   A song title by a singer or band is always "MUSIC".
2. "clean_title" MUST have every promotional tag removed. Delete any
   parenthesised or bracketed segment such as (Official Music Video),
   (Official Audio), (Lyrics), [4K], [HD], and any trailing "| Artist Name".
   Keep genuine title parts like "(feat. X)" or "(Remix)".
3. "clean_artist" is the artist name only (for a VIDEO, the channel), with no
   channel suffixes such as "VEVO" or "- Topic".

Examples:
Input: artist="Dua Lipa", title="Levitating (Official Music Video) [HD]", site="youtube.com", has_video=yes
Output: {{"type": "MUSIC", "clean_artist": "Dua Lipa", "clean_title": "Levitating"}}
Input: artist="Arijit Singh, Pritam", title="Kesariya", site="unknown", has_video=unknown
Output: {{"type": "MUSIC", "clean_artist": "Arijit Singh, Pritam", "clean_title": "Kesariya"}}
Input: artist="Joe Rogan", title="JRE #1500 - Elon Musk", site="spotify.com", has_video=no
Output: {{"type": "OTHER", "clean_artist": "Joe Rogan", "clean_title": "JRE #1500 - Elon Musk"}}
Input: artist="Veritasium", title="The Surprising Secret of Synchronization", site="youtube.com", has_video=yes
Output: {{"type": "VIDEO", "clean_artist": "Veritasium", "clean_title": "The Surprising Secret of Synchronization"}}

Now do the same for:
artist="{artist}"
title="{title}"
site="{site}"
has_video={has_video}

Respond ONLY with a JSON object of the form:
{{"type": "MUSIC" or "VIDEO" or "OTHER", "clean_artist": "...", "clean_title": "..."}}"""


@dataclass(frozen=True)
class TrackVerdict:
    type: str  # MUSIC, VIDEO or OTHER
    clean_artist: str
    clean_title: str

    @property
    def is_music(self) -> bool:
        return self.type == MUSIC

    @property
    def is_video(self) -> bool:
        return self.type == VIDEO

    @property
    def is_wanted(self) -> bool:
        return self.type in (MUSIC, VIDEO)


def route(verdict: TrackVerdict, has_video: bool) -> tuple[bool, bool]:
    """(to the music library, to the video library) for a verdict.

    A music video goes to both: its audio as a track, the whole thing as a
    video. Music with no picture is a track only; a video is a video only.
    """
    return verdict.is_music, verdict.is_video or (verdict.is_music and has_video)


def _load_json(raw: str) -> dict | None:
    """Parse a JSON object from a model response.

    Falls back to the first {...} span: the retry path runs without Ollama's
    grammar, and small models like to wrap the object in prose or a fence.
    """
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        pass

    if not isinstance(raw, str):
        return None
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        return json.loads(raw[start : end + 1])
    except ValueError:
        return None


_BRACKETED = re.compile(r"\s*[\(\[]([^\)\]]*)[\)\]]")
_PROMO = re.compile(
    r"\b(official|music video|video|audio|lyrics?|visuali[sz]er|hd|hq|4k|8k|"
    r"\d{3,4}p|mv|m/v)\b", re.I)
_KEEP = re.compile(r"\b(feat|ft|remix|mix|edit|version|live|acoustic|cover)\b\.?", re.I)


def strip_promo(title: str) -> str:
    """Remove promotional tags the model left behind.

    A 1.2B model follows the cleaning rule most of the time, not all of it --
    it has left "(Official Music Video)" in place even for its own few-shot
    example. Tags are a fixed vocabulary, so a regex finishes the job.
    Brackets naming a genuine version -- "(feat. X)", "(Remix)", "(Live)" --
    are kept even when they also say "official".
    """
    def drop(m: re.Match) -> str:
        inner = m.group(1)
        return "" if _PROMO.search(inner) and not _KEEP.search(inner) else m.group(0)

    cleaned = _BRACKETED.sub(drop, title)
    if " | " in cleaned:  # "Song | Artist | Label"
        cleaned = cleaned.split(" | ", 1)[0]
    return cleaned.strip() or title.strip()


def parse_verdict(raw: str, fallback_artist: str, fallback_title: str) -> TrackVerdict | None:
    """Validate a raw LLM response into a verdict, or None if unusable.

    Small models drift: wrong enum casing, missing fields, blanked titles.
    Missing fields fall back to the original strings rather than to "".
    """
    data = _load_json(raw)
    if data is None:
        log.warning("LLM returned non-JSON: %r", raw[:200] if raw else raw)
        return None

    if not isinstance(data, dict):
        return None

    verdict_type = _ALIASES.get(str(data.get("type", "")).strip().upper())
    if verdict_type is None:
        log.warning("LLM returned unknown type %r", data.get("type"))
        return None

    artist = str(data.get("clean_artist") or "").strip() or fallback_artist
    title = strip_promo(str(data.get("clean_title") or "").strip() or fallback_title)
    if not title.strip():
        log.warning("No usable title after cleaning; refusing verdict")
        return None

    return TrackVerdict(type=verdict_type, clean_artist=artist, clean_title=title)


class CurationAgent:
    """Owns the LLM client."""

    def __init__(self, client: AsyncClient | None = None):
        self._client = client or AsyncClient(host=OLLAMA_HOST)

    async def _chat(self, prompt: str, *, json_mode: bool) -> str | None:
        kwargs = {
            "model": MODEL_ID,
            "messages": [{"role": "user", "content": prompt}],
            "options": {"temperature": 0},  # deterministic output for a classifier
        }
        if json_mode:
            kwargs["format"] = "json"

        response = await asyncio.wait_for(
            self._client.chat(**kwargs), timeout=LLM_TIMEOUT_SECONDS
        )
        return response.message.content

    async def classify(self, artist: str, title: str, site: str = "",
                       has_video: bool | None = None) -> TrackVerdict | None:
        prompt = _PROMPT.format(
            artist=artist, title=title, site=site or "unknown",
            has_video={True: "yes", False: "no", None: "unknown"}[has_video],
        )

        # Ollama's JSON grammar intermittently 500s on this model. The prompt
        # already demands bare JSON, so an unconstrained retry recovers it.
        for json_mode in (True, False):
            try:
                raw = await self._chat(prompt, json_mode=json_mode)
            except asyncio.TimeoutError:
                log.error("LLM timed out after %ss", LLM_TIMEOUT_SECONDS)
                return None
            except Exception as exc:
                if json_mode:
                    log.warning("JSON-mode call failed (%s); retrying unconstrained", exc)
                    continue
                log.exception("LLM call failed (is Ollama running?)")
                return None

            verdict = parse_verdict(raw, artist, title)
            if verdict is not None:
                return verdict
            if json_mode:
                log.warning("Unusable JSON-mode verdict; retrying unconstrained")

        return None
