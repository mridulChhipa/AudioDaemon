"""Single source of truth for paths and settings.

Paths are resolved from this file's location rather than the working directory:
the daemon is launched from a Startup shortcut, where CWD is not the project.

Per-machine overrides live in `.env` (see `.env.example`). Variables already
set in the real environment win over the file, so a one-off
`$env:AUDIODAEMON_X = ...` still does what it says.
"""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = Path(os.environ.get("AUDIODAEMON_ENV_FILE") or PROJECT_ROOT / ".env")


def _load_env(path: Path = ENV_PATH) -> None:
    """Load `.env` without overriding the environment. Missing file is fine."""
    try:
        from dotenv import load_dotenv
    except ImportError:  # optional: the defaults below still apply
        return
    load_dotenv(path, override=False)


_load_env()

DATA_DIR = PROJECT_ROOT / "data"
LIBRARY_DIR = DATA_DIR / "library"
MUSIC_DIR = LIBRARY_DIR / "music"
VIDEO_DIR = LIBRARY_DIR / "video"
# Network-captured media, per stream, until a stream is complete.
STREAM_PARTIALS_DIR = DATA_DIR / "streams"
# Streams are assembled here before being remuxed into the libraries.
STAGING_DIR = DATA_DIR / "staging"
DB_PATH = DATA_DIR / "agent_memory.db"
LOG_PATH = DATA_DIR / "daemon.log"

# --- The brain -------------------------------------------------------------
MODEL_ID = os.environ.get("AUDIODAEMON_MODEL", "LiquidAI/lfm2.5-1.2b-instruct:latest")
OLLAMA_HOST = os.environ.get("AUDIODAEMON_OLLAMA_HOST", "http://localhost:11434")
LLM_TIMEOUT_SECONDS = 60.0  # a healthy classification takes ~1s

# --- The listener ----------------------------------------------------------
# How quickly a play's start, track changes and stop are noticed.
POLL_SECONDS = 0.5

# --- Application-layer capture (Chrome DevTools Protocol) ------------------
# The browser you use every day, on its own profile. Once remote debugging is
# turned on at chrome://inspect/#remote-debugging (comet://inspect/... in
# Comet), the browser writes `DevToolsActivePort` into its User Data directory;
# that file says where to connect. AUDIODAEMON_BROWSER_DATA names the directory;
# unset, the most recently written file among these browsers' wins.
_LOCAL_APP_DATA = Path(os.environ.get("LOCALAPPDATA", ""))
BROWSER_DATA_DIRS = (
    [Path(os.environ["AUDIODAEMON_BROWSER_DATA"])]
    if os.environ.get("AUDIODAEMON_BROWSER_DATA") else [
        _LOCAL_APP_DATA / "Perplexity" / "Comet" / "User Data",
        _LOCAL_APP_DATA / "Google" / "Chrome" / "User Data",
        _LOCAL_APP_DATA / "Microsoft" / "Edge" / "User Data",
        _LOCAL_APP_DATA / "BraveSoftware" / "Brave-Browser" / "User Data",
    ]
)
# How long to wait for you to answer the browser's "allow remote debugging?"
# prompt before trying again.
CDP_APPROVAL_SECONDS = 120.0
# Chromium evicts response bodies past these; media segments are large.
CDP_MAX_TOTAL_BUFFER = 512 * 1024 * 1024
CDP_MAX_RESOURCE_BUFFER = 64 * 1024 * 1024
# After a play ends, how long to let in-flight responses land before judging
# whether the stream path produced the item.
STREAM_SETTLE_SECONDS = 3.0
# A crude guard against bumpers and stingers; adverts are mostly caught by the
# classifier and by a stream's length not matching the page's video.
MIN_VIDEO_SECONDS = 10.0

# --- Completion ------------------------------------------------------------
# How much of a stream's head or tail may be missing and still count as
# complete: segment timings are rounded, and the reported length with them.
COVERAGE_TOLERANCE_SECONDS = 2.0
# Claimed items that never completed are deleted after this long.
PARTIAL_MAX_AGE_DAYS = 30
# Missing pieces are fetched from the page in requests of at most this size,
# well under CDP_MAX_RESOURCE_BUFFER so the browser keeps every body.
FETCH_CHUNK_BYTES = 8 * 1024 * 1024
# Fetch-and-recheck rounds per play: a file of unknown size needs one round to
# learn its size and another to fetch the rest.
COMPLETION_ROUNDS = 3

# --- ffmpeg ----------------------------------------------------------------
FFMPEG_DIR = Path(
    os.environ.get("AUDIODAEMON_FFMPEG_DIR", r"D:\envs\vanta\Library\bin")
)


def ensure_dirs() -> None:
    """Create the data directories if they don't exist yet."""
    for path in (MUSIC_DIR, VIDEO_DIR, STREAM_PARTIALS_DIR, STAGING_DIR):
        path.mkdir(parents=True, exist_ok=True)
