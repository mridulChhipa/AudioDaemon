"""SQLite memory -- the record of what the library already owns.

One table for both libraries, told apart by `kind` ("music" or "video"). A
music video is two rows: the track and the video are owned independently.

Synchronous and blocking; callers on the event loop wrap these in
`asyncio.to_thread`. A connection is opened per call, which is cheap and
sidesteps SQLite's cross-thread sharing rules.
"""
import re
import sqlite3
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

from core.config import DB_PATH

MUSIC, VIDEO = "music", "video"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS acquired_media (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    kind        TEXT NOT NULL,
    media_hash  TEXT NOT NULL,
    artist      TEXT NOT NULL,
    title       TEXT NOT NULL,
    source      TEXT,
    file_path   TEXT,
    acquired_at DATETIME NOT NULL
);
-- Dedup is enforced here, not by a check-then-insert in Python: two overlapping
-- events can both pass a check before either inserts.
CREATE UNIQUE INDEX IF NOT EXISTS idx_media ON acquired_media (kind, media_hash);
"""

_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def _connect(db_path: Path | None) -> sqlite3.Connection:
    # Resolved here, not as a default arg, which would bind DB_PATH at import.
    db_path = db_path or DB_PATH
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(db_path, timeout=10)


def init_db(db_path: Path | None = None) -> None:
    """Create the table and unique index. Safe to call on every startup."""
    with _connect(db_path) as conn:
        conn.executescript(_SCHEMA)


def _norm(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value or "")
    stripped = "".join(c for c in decomposed if not unicodedata.combining(c))
    return _NON_ALNUM.sub("", stripped.casefold())


def make_track_hash(artist: str, title: str) -> str:
    """Build the dedup key for a track.

    Collapses how differently sources spell the same track -- casing, padding,
    punctuation, accents. Separators are removed outright rather than folded to
    a space, so "T.N.T." and "TNT" land on one key.
    """
    return f"{_norm(artist)}::{_norm(title)}"


def make_video_hash(page: str) -> str:
    """Build the dedup key for a video.

    A video's identity is where it lives -- `yt:<id>`, or a page -- because
    titles repeat across channels and change after upload.
    """
    return f"page::{page}"


def add_media(
    media_hash: str,
    artist: str,
    title: str,
    source: str,
    file_path: str,
    db_path: Path | None = None,
    *,
    kind: str = MUSIC,
) -> bool:
    """Record an acquired item. Returns False if it was already recorded."""
    with _connect(db_path) as conn:
        cur = conn.execute(
            "INSERT OR IGNORE INTO acquired_media "
            "(kind, media_hash, artist, title, source, file_path, acquired_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                kind,
                media_hash,
                artist,
                title,
                source,
                file_path,
                datetime.now(timezone.utc).isoformat(timespec="seconds"),
            ),
        )
    return cur.rowcount > 0


def get_file_path(media_hash: str, db_path: Path | None = None, *,
                  kind: str = MUSIC) -> str | None:
    """The recorded path for an item, or None if it isn't in the library."""
    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT file_path FROM acquired_media WHERE kind = ? AND media_hash = ? LIMIT 1",
            (kind, media_hash),
        ).fetchone()
    return row[0] if row else None


def forget_media(media_hash: str, db_path: Path | None = None, *, kind: str = MUSIC) -> bool:
    """Drop an item from memory so it can be acquired again."""
    with _connect(db_path) as conn:
        cur = conn.execute(
            "DELETE FROM acquired_media WHERE kind = ? AND media_hash = ?",
            (kind, media_hash),
        )
    return cur.rowcount > 0
