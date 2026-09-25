"""AudioDaemon entry point.

Run `python daemon.py` for a visible session with console logging, or launch
daemon.pyw (pythonw) for the invisible background daemon.
"""
import asyncio
import ctypes
import logging
import sys
from logging.handlers import RotatingFileHandler

from core import database
from core.config import LOG_PATH, STAGING_DIR, ensure_dirs
from core.listener import Listener
from core.pipeline import Pipeline
from core.sniffer.cdp import Sniffer
from core.sniffer.streams import StreamStore

# One instance only: two daemons would each ask the browser for a connection
# and fight over the stream store.
_MUTEX_NAME = "Global\\AudioDaemon_SingleInstance"
_ERROR_ALREADY_EXISTS = 183

log = logging.getLogger("audiodaemon")


def _setup_logging() -> None:
    ensure_dirs()
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    # httpx logs a line per Ollama request, drowning everything worth reading;
    # websockets logs every CDP frame at debug.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("websockets").setLevel(logging.WARNING)

    # Under pythonw there is no console, so this is the only record.
    file_handler = RotatingFileHandler(
        LOG_PATH, maxBytes=1_000_000, backupCount=3, encoding="utf-8"
    )
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    )
    root.addHandler(file_handler)

    if sys.stdout is not None:
        console = logging.StreamHandler(sys.stdout)
        console.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(message)s"))
        root.addHandler(console)


def _claim_single_instance() -> bool:
    """Take the named mutex. False if another daemon already holds it.

    use_last_error rather than kernel32.GetLastError(): ctypes can make its own
    calls between the two and clobber the value. The handle is deliberately
    leaked -- Windows releases it on exit.
    """
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateMutexW(None, False, _MUTEX_NAME)
    return ctypes.get_last_error() != _ERROR_ALREADY_EXISTS


def _clear_staging() -> int:
    """Delete assemblies a previous run left half-finished."""
    removed = 0
    for path in STAGING_DIR.glob("*"):
        if path.is_file():
            path.unlink(missing_ok=True)
            removed += 1
    return removed


async def _run() -> None:
    database.init_db()
    await asyncio.to_thread(_clear_staging)

    store = StreamStore()
    sniffer = Sniffer(store)
    listener = Listener(sniffer, Pipeline(store, sniffer))
    # The sniffer reconnects on its own; the listener polls forever. Either
    # dying is a bug worth crashing (and logging) for.
    await asyncio.gather(sniffer.run(), listener.run())


def main() -> int:
    _setup_logging()

    if not _claim_single_instance():
        log.info("Another AudioDaemon instance is already running; exiting")
        return 0

    log.info("AudioDaemon starting (network streams through the browser)")
    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        log.info("Interrupted; shutting down")
    except Exception:
        log.exception("AudioDaemon crashed")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
