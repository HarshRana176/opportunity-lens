"""
Characterizes app.config: Settings resolution, defaults, and the
"missing/invalid config fails clearly at first use, never at import"
contract that the rest of the app (app.database especially) depends on.
"""
import pytest

import app.config as config


@pytest.fixture(autouse=True)
def _isolate_from_real_environment(monkeypatch):
    """
    Prevent the repo's real .env (which has a working DATABASE_URL, per
    manual verification that `import app.main` succeeds in this dev
    environment) from leaking into these tests. Without this, tests
    asserting "DATABASE_URL is missing" behavior would incorrectly see
    the real value once app.config's `load_dotenv()` call populates
    os.environ from it.
    """
    monkeypatch.setattr(config, "load_dotenv", lambda *args, **kwargs: None)
    for var in ("DATABASE_URL", "UPLOAD_DIR", "MAX_UPLOAD_BYTES"):
        monkeypatch.delenv(var, raising=False)


def test_importing_config_does_not_require_database_url():
    # This module is already imported (above); the meaningful assertion
    # is that nothing at module level touched the environment -- proven
    # by the fact that every other test in this file can freely control
    # DATABASE_URL via monkeypatch without import having already
    # resolved (and cached) it.
    assert hasattr(config, "get_settings")


def test_missing_database_url_raises_configuration_error():
    with pytest.raises(config.ConfigurationError, match="DATABASE_URL"):
        config.get_settings()


def test_missing_database_url_error_does_not_echo_a_value():
    with pytest.raises(config.ConfigurationError) as exc_info:
        config.get_settings()

    message = str(exc_info.value)
    # There is no value to leak in this path (DATABASE_URL is absent),
    # but assert the message never resembles a DSN regardless.
    assert "://" not in message
    assert "@" not in message


def test_settings_reads_database_url_from_environment(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://user:pw@localhost/db")

    settings = config.get_settings()

    assert settings.database_url == "postgresql://user:pw@localhost/db"


def test_settings_defaults_upload_dir_and_max_upload_bytes(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://user:pw@localhost/db")

    settings = config.get_settings()

    assert settings.upload_dir == config.DEFAULT_UPLOAD_DIR
    assert settings.max_upload_bytes == config.DEFAULT_MAX_UPLOAD_BYTES


def test_settings_reads_custom_upload_dir_and_max_upload_bytes(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://user:pw@localhost/db")
    monkeypatch.setenv("UPLOAD_DIR", "custom_uploads")
    monkeypatch.setenv("MAX_UPLOAD_BYTES", "12345")

    settings = config.get_settings()

    assert settings.upload_dir == "custom_uploads"
    assert settings.max_upload_bytes == 12345


def test_invalid_max_upload_bytes_raises_configuration_error(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://user:pw@localhost/db")
    monkeypatch.setenv("MAX_UPLOAD_BYTES", "not-a-number")

    with pytest.raises(config.ConfigurationError, match="MAX_UPLOAD_BYTES"):
        config.get_settings()


def test_get_settings_is_cached_within_a_process(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://user:pw@localhost/db")

    first = config.get_settings()
    second = config.get_settings()

    assert first is second
