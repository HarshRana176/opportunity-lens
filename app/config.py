"""
Configuration boundary for the application.

Nothing here reads the environment at import time -- `get_settings()`
resolves (and caches) configuration on first call, so importing this
module (or anything that imports it) never requires DATABASE_URL, or
any other environment variable, to be set.
"""
import os
from functools import lru_cache

from dotenv import load_dotenv


class ConfigurationError(RuntimeError):
    """Raised when required configuration is missing or invalid."""


DEFAULT_UPLOAD_DIR = "uploads"
DEFAULT_MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB; not specified by any requirement.


class Settings:
    def __init__(self):
        load_dotenv()

        database_url = os.getenv("DATABASE_URL")
        if not database_url:
            raise ConfigurationError(
                "DATABASE_URL is not set. Configure it in the environment "
                "or in a .env file before starting the application."
            )
        self.database_url = database_url

        self.upload_dir = os.getenv("UPLOAD_DIR", DEFAULT_UPLOAD_DIR)

        raw_max_upload_bytes = os.getenv("MAX_UPLOAD_BYTES")
        if raw_max_upload_bytes:
            try:
                self.max_upload_bytes = int(raw_max_upload_bytes)
            except ValueError as exc:
                raise ConfigurationError(
                    "MAX_UPLOAD_BYTES must be an integer number of bytes."
                ) from exc
        else:
            self.max_upload_bytes = DEFAULT_MAX_UPLOAD_BYTES


@lru_cache
def get_settings() -> Settings:
    return Settings()
