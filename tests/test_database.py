from core import database


def test_init_db_is_idempotent(tmp_path):
    db = tmp_path / "memory.db"
    database.init_db(db)
    database.init_db(db)
    assert db.exists()


def test_add_and_query_round_trip(tmp_path):
    db = tmp_path / "memory.db"
    database.init_db(db)

    h = database.make_track_hash("The Weeknd", "Blinding Lights")
    assert database.get_file_path(h, db) is None

    assert database.add_media(h, "The Weeknd", "Blinding Lights", "SMTC", "C:/a.mp3", db)
    assert database.get_file_path(h, db) == "C:/a.mp3"


def test_duplicate_insert_is_ignored(tmp_path):
    """The unique index, not a Python check, is what makes dedup race-safe."""
    db = tmp_path / "memory.db"
    database.init_db(db)
    h = database.make_track_hash("Dua Lipa", "Levitating")

    assert database.add_media(h, "Dua Lipa", "Levitating", "SMTC", "C:/a.mp3", db) is True
    assert database.add_media(h, "Dua Lipa", "Levitating", "SMTC", "C:/b.mp3", db) is False
    assert database.get_file_path(h, db) == "C:/a.mp3"  # the first one stands


def test_forget_media(tmp_path):
    db = tmp_path / "memory.db"
    database.init_db(db)
    h = database.make_track_hash("Muse", "Hysteria")
    database.add_media(h, "Muse", "Hysteria", "SMTC", "C:/lib/x.mp3", db)

    assert database.forget_media(h, db) is True
    assert database.get_file_path(h, db) is None
    assert database.forget_media(h, db) is False  # already gone


class TestMakeTrackHash:
    """Cosmetic differences between sources must collapse to one key."""

    def test_casing_and_padding(self):
        assert database.make_track_hash("  The Weeknd ", "Blinding Lights") == (
            database.make_track_hash("the weeknd", "blinding lights")
        )

    def test_punctuation(self):
        assert database.make_track_hash("AC/DC", "T.N.T.") == (
            database.make_track_hash("AC DC", "TNT")
        )

    def test_accents_fold_rather_than_split(self):
        assert database.make_track_hash("Beyoncé", "Halo") == (
            database.make_track_hash("Beyonce", "Halo")
        )

    def test_different_tracks_stay_distinct(self):
        assert database.make_track_hash("Adele", "Hello") != (
            database.make_track_hash("Adele", "Halo")
        )

    def test_artist_and_title_are_not_conflated(self):
        # Without a separator, ("ab", "c") and ("a", "bc") would collide.
        assert database.make_track_hash("ab", "c") != database.make_track_hash("a", "bc")


class TestKinds:
    """One table, two libraries: a music video is a track *and* a video."""

    def test_music_and_video_rows_are_independent(self, tmp_path):
        db = tmp_path / "memory.db"
        database.init_db(db)
        h = "same::key"
        assert database.add_media(h, "A", "T", "s", "C:/m.opus", db)
        assert database.add_media(h, "A", "T", "s", "C:/v.webm", db, kind="video")
        assert database.get_file_path(h, db) == "C:/m.opus"
        assert database.get_file_path(h, db, kind="video") == "C:/v.webm"

        database.forget_media(h, db, kind="video")
        assert database.get_file_path(h, db) == "C:/m.opus"
        assert database.get_file_path(h, db, kind="video") is None


def test_a_video_is_known_by_its_page():
    """Not by its title, which repeats across channels and changes after upload."""
    assert database.make_video_hash("yt:abc") == "page::yt:abc"
    assert database.make_video_hash("yt:abc") != database.make_video_hash("yt:abd")
