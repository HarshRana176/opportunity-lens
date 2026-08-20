"""
Safe on-disk handling of uploaded files: filename sanitization,
collision-safe naming, path containment, size limiting, and PDF
magic-byte validation.

Writes use a temp-file -> validate -> promote flow: nothing lands at
its final name until it has been fully written and passed validation,
and any failure along the way removes the temp file. This keeps a
failed or oversized upload from leaving a partial or unvalidated file
under its real name.
"""
import os
import re
import uuid
from pathlib import Path

from app.config import get_settings

PDF_MAGIC_BYTES = b"%PDF"

_CHUNK_SIZE = 1024 * 1024

# Characters that are unsafe or reserved in filenames on at least one
# common target filesystem (Windows reserves <>:"/\|?*, POSIX reserves
# / and NUL), plus all control characters.
_UNSAFE_CHARS_PATTERN = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


class InvalidUploadError(ValueError):
    """Raised when an uploaded file fails name or content validation."""


class UploadTooLargeError(ValueError):
    """Raised when an uploaded file exceeds the configured size limit."""


def sanitize_filename(filename: str | None) -> str:
    """
    Reduce a client-supplied filename to a safe basename: no directory
    components (from either '/' or '\\' -- traversal and absolute
    paths from either convention), no null bytes or other control
    characters, no filesystem-reserved characters. Falls back to a
    generic name if nothing safe remains.
    """
    if not filename:
        filename = ""

    # Normalize both separator conventions so a basename split works
    # regardless of which OS's convention the client's filename uses.
    normalized = filename.replace("\\", "/")
    basename = normalized.rsplit("/", 1)[-1]

    basename = _UNSAFE_CHARS_PATTERN.sub("", basename)
    basename = basename.strip().strip(".")

    if not basename:
        basename = "upload.pdf"

    return basename


def resolve_within_upload_dir(upload_dir: Path, filename: str) -> Path:
    """Resolve `filename` inside `upload_dir`, refusing any escape."""
    upload_dir_resolved = upload_dir.resolve()
    resolved = (upload_dir_resolved / filename).resolve()

    if not resolved.is_relative_to(upload_dir_resolved):
        raise InvalidUploadError(
            "Resolved upload path escapes the upload directory."
        )

    return resolved


def prepare_destination(original_filename: str | None) -> tuple[Path, Path, str]:
    """
    Compute (temp_path, final_path, sanitized_original_filename) for a
    new upload, without writing anything. Both paths are UUID-prefixed
    so concurrent/repeated uploads of the same original filename never
    collide.
    """
    settings = get_settings()
    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)

    sanitized_original = sanitize_filename(original_filename)
    stored_filename = f"{uuid.uuid4().hex}_{sanitized_original}"

    final_path = resolve_within_upload_dir(upload_dir, stored_filename)
    temp_path = resolve_within_upload_dir(upload_dir, f".{stored_filename}.part")

    return temp_path, final_path, sanitized_original


def write_upload(file_obj, temp_path: Path, max_bytes: int) -> None:
    """
    Stream `file_obj` (any object with a synchronous `.read(size)`) to
    temp_path, enforcing `max_bytes` and validating PDF magic bytes.
    Raises UploadTooLargeError or InvalidUploadError without leaving a
    file behind; the caller does not need to clean up on these errors,
    only on errors it raises itself around this call.
    """
    total_bytes = 0
    header = b""

    try:
        with open(temp_path, "wb") as buffer:
            while True:
                chunk = file_obj.read(_CHUNK_SIZE)
                if not chunk:
                    break

                if len(header) < len(PDF_MAGIC_BYTES):
                    header += chunk

                total_bytes += len(chunk)
                if total_bytes > max_bytes:
                    raise UploadTooLargeError(
                        f"Upload exceeds the maximum allowed size of "
                        f"{max_bytes} bytes."
                    )

                buffer.write(chunk)

        if not header.startswith(PDF_MAGIC_BYTES):
            raise InvalidUploadError("File does not appear to be a valid PDF.")

    except Exception:
        cleanup(temp_path)
        raise


def promote(temp_path: Path, final_path: Path) -> None:
    """Atomically move a validated temp file to its final name."""
    os.replace(temp_path, final_path)


def cleanup(path: Path) -> None:
    """Best-effort removal of a file; never raises."""
    try:
        if path.exists():
            path.unlink()
    except OSError:
        pass


def save_upload(file_obj, original_filename: str | None) -> tuple[Path, str]:
    """
    Full temp-write -> validate -> promote flow for one uploaded file.

    On success, returns (final_path, sanitized_original_filename) and
    no temp file remains. On any failure -- bad size, bad content, or
    anything else raised while writing -- the temp file is removed and
    the exception propagates; nothing is left at the final path.
    """
    settings = get_settings()
    temp_path, final_path, sanitized_original = prepare_destination(original_filename)

    write_upload(file_obj, temp_path, settings.max_upload_bytes)

    try:
        promote(temp_path, final_path)
    except Exception:
        cleanup(temp_path)
        raise

    return final_path, sanitized_original
