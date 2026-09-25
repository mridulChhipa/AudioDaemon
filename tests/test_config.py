"""`.env` loading and its precedence.

config reads the environment at import, so each case reloads it in a clean
environment with a temporary `.env`.
"""
import importlib
import os

import pytest

from core import config


@pytest.fixture
def reload_with(tmp_path, monkeypatch):
    """Reload config against a given .env text and environment."""
    for name in ("AUDIODAEMON_BROWSER_DATA", "AUDIODAEMON_MODEL"):
        monkeypatch.delenv(name, raising=False)

    def load(env_text: str | None, **environ):
        env_file = tmp_path / ".env"
        if env_text is None:
            env_file.unlink(missing_ok=True)
        else:
            env_file.write_text(env_text, encoding="utf-8")
        for key, value in environ.items():
            monkeypatch.setenv(key, value)
        monkeypatch.setenv("AUDIODAEMON_ENV_FILE", str(env_file))
        return importlib.reload(config)

    yield load
    # load_dotenv wrote the file's values into os.environ; undo restores what
    # was there, and the reload leaves the real configuration for other tests.
    for name in ("AUDIODAEMON_BROWSER_DATA", "AUDIODAEMON_MODEL"):
        os.environ.pop(name, None)
    monkeypatch.undo()
    importlib.reload(config)


def test_defaults_without_a_file(reload_with):
    cfg = reload_with(None)
    assert cfg.MODEL_ID == "LiquidAI/lfm2.5-1.2b-instruct:latest"
    # Every known browser's own profile is looked in, Comet first.
    assert [d.parts[-3:] for d in cfg.BROWSER_DATA_DIRS][0] == ("Perplexity", "Comet", "User Data")
    assert len(cfg.BROWSER_DATA_DIRS) == 4


def test_file_sets_values(reload_with):
    cfg = reload_with("AUDIODAEMON_BROWSER_DATA=D:\\Profiles\\Chrome\nAUDIODAEMON_MODEL=other:1b\n")
    assert [str(d) for d in cfg.BROWSER_DATA_DIRS] == ["D:\\Profiles\\Chrome"]
    assert cfg.MODEL_ID == "other:1b"


def test_environment_beats_the_file(reload_with):
    cfg = reload_with("AUDIODAEMON_MODEL=from-file\n", AUDIODAEMON_MODEL="from-env")
    assert cfg.MODEL_ID == "from-env"
