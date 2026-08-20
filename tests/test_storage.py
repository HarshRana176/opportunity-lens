"""
Characterizes app.storage: filename sanitization, path containment,
size limiting, PDF magic-byte validation, and the temp-write ->
validate -> promote flow (including cleanup on every failure path).

`app.storage.get_settings` is monkeypatched to a fixed, isolated
SimpleNamespace per test (via the `fake_settings` fixture) rather than
going through the real app.config.get_settings() -- these tests are
about storage behavior given some configuration, not about config
resolution itself (that's tests/test_config.py).
"""
import io
from types import SimpleNamespace

import pytest

import app.storage as storage


@pytest.fixture
def fake_settings(tmp_path, monkeypatch):
    settings = SimpleNamespace(
        upload_dir=str(tmp_path),
        max_upload_bytes=10_000_000,
    )
    monkeypatch.setattr(storage, "get_settings", lambda: settings)
    return settings


class TestSanitizeFilename:
    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("resume.pdf", "resume.pdf"),
            ("My Resume (Final).pdf", "My Resume (Final).pdf"),
            ("../../etc/passwd", "passwd"),
            ("/etc/passwd", "passwd"),
            ("..\\..\\Windows\\evil.exe", "evil.exe"),
            ("C:\\Windows\\System32\\evil.exe", "evil.exe"),
            ("a\x00b.pdf", "ab.pdf"),
            ("con:evil.pdf", "conevil.pdf"),
            ("", "upload.pdf"),
            ("   ", "upload.pdf"),
            ("....", "upload.pdf"),
            (None, "upload.pdf"),
        ],
    )
    def test_sanitize_filename(self, raw, expected):
        assert storage.sanitize_filename(raw) == expected

    def test_sanitized_name_never_contains_a_path_separator(self):
        result = storage.sanitize_filename("../../../etc/passwd")
        assert "/" not in result
        assert "\\" not in result


class TestResolveWithinUploadDir:
    def test_accepts_a_safe_filename(self, tmp_path):
        result = storage.resolve_within_upload_dir(tmp_path, "abc_resume.pdf")
        assert result.parent == tmp_path.resolve()

    def test_rejects_an_escaping_path(self, tmp_path):
        with pytest.raises(storage.InvalidUploadError):
            storage.resolve_within_upload_dir(tmp_path, "../escape.pdf")

    def test_rejects_a_nested_escaping_path(self, tmp_path):
        with pytest.raises(storage.InvalidUploadError):
            storage.resolve_within_upload_dir(tmp_path, "sub/../../escape.pdf")


class TestPrepareDestination:
    def test_returns_paths_inside_the_upload_dir(self, fake_settings, tmp_path):
        temp_path, final_path, sanitized = storage.prepare_destination("My Resume.pdf")

        upload_dir = tmp_path.resolve()
        assert temp_path.is_relative_to(upload_dir)
        assert final_path.is_relative_to(upload_dir)
        assert sanitized == "My Resume.pdf"
        assert final_path.name.endswith("_My Resume.pdf")
        assert temp_path.name.startswith(".")
        assert temp_path.name.endswith(".part")

    def test_creates_the_upload_dir_if_missing(self, monkeypatch, tmp_path):
        missing_dir = tmp_path / "nested" / "uploads"
        settings = SimpleNamespace(upload_dir=str(missing_dir), max_upload_bytes=10_000_000)
        monkeypatch.setattr(storage, "get_settings", lambda: settings)

        assert not missing_dir.exists()
        storage.prepare_destination("resume.pdf")
        assert missing_dir.exists()

    def test_is_collision_safe_for_identical_original_names(self, fake_settings):
        _, final1, _ = storage.prepare_destination("resume.pdf")
        _, final2, _ = storage.prepare_destination("resume.pdf")

        assert final1 != final2


class TestWriteUpload:
    def test_accepts_valid_pdf_content(self, tmp_path):
        content = b"%PDF-1.4\nfake pdf body"
        temp_path = tmp_path / ".x.part"

        storage.write_upload(io.BytesIO(content), temp_path, max_bytes=10_000)

        assert temp_path.read_bytes() == content

    def test_rejects_non_pdf_content_and_cleans_up(self, tmp_path):
        temp_path = tmp_path / ".x.part"

        with pytest.raises(storage.InvalidUploadError):
            storage.write_upload(io.BytesIO(b"not a pdf"), temp_path, max_bytes=10_000)

        assert not temp_path.exists()

    def test_rejects_empty_content_and_cleans_up(self, tmp_path):
        temp_path = tmp_path / ".x.part"

        with pytest.raises(storage.InvalidUploadError):
            storage.write_upload(io.BytesIO(b""), temp_path, max_bytes=10_000)

        assert not temp_path.exists()

    def test_enforces_max_bytes_and_cleans_up(self, tmp_path):
        content = b"%PDF-1.4" + b"0" * 100
        temp_path = tmp_path / ".x.part"

        with pytest.raises(storage.UploadTooLargeError):
            storage.write_upload(io.BytesIO(content), temp_path, max_bytes=10)

        assert not temp_path.exists()

    def test_enforces_max_bytes_across_multiple_chunks(self, monkeypatch, tmp_path):
        # A tiny chunk size forces the size check to work across many
        # .read() calls, not just within a single chunk.
        monkeypatch.setattr(storage, "_CHUNK_SIZE", 4)
        content = b"%PDF" + b"X" * 20
        temp_path = tmp_path / ".x.part"

        with pytest.raises(storage.UploadTooLargeError):
            storage.write_upload(io.BytesIO(content), temp_path, max_bytes=10)

        assert not temp_path.exists()


class TestPromoteAndCleanup:
    def test_promote_moves_temp_to_final(self, tmp_path):
        temp_path = tmp_path / ".x.part"
        temp_path.write_bytes(b"data")
        final_path = tmp_path / "final.pdf"

        storage.promote(temp_path, final_path)

        assert not temp_path.exists()
        assert final_path.read_bytes() == b"data"

    def test_cleanup_is_a_no_op_when_the_file_is_already_gone(self, tmp_path):
        missing = tmp_path / "does_not_exist.pdf"
        storage.cleanup(missing)  # must not raise


class TestSaveUpload:
    def test_success_leaves_only_the_final_file(self, fake_settings, tmp_path):
        content = b"%PDF-1.4\nbody"

        final_path, sanitized = storage.save_upload(io.BytesIO(content), "My Resume.pdf")

        assert final_path.exists()
        assert final_path.read_bytes() == content
        assert sanitized == "My Resume.pdf"
        assert list(tmp_path.iterdir()) == [final_path]

    def test_invalid_content_leaves_no_files_behind(self, fake_settings, tmp_path):
        with pytest.raises(storage.InvalidUploadError):
            storage.save_upload(io.BytesIO(b"not a pdf"), "resume.pdf")

        assert list(tmp_path.iterdir()) == []

    def test_oversize_upload_leaves_no_files_behind(self, monkeypatch, tmp_path):
        settings = SimpleNamespace(upload_dir=str(tmp_path), max_upload_bytes=5)
        monkeypatch.setattr(storage, "get_settings", lambda: settings)

        with pytest.raises(storage.UploadTooLargeError):
            storage.save_upload(io.BytesIO(b"%PDF-1.4 way too big"), "resume.pdf")

        assert list(tmp_path.iterdir()) == []

    def test_two_uploads_of_the_same_original_name_do_not_collide(
        self, fake_settings, tmp_path
    ):
        path1, _ = storage.save_upload(io.BytesIO(b"%PDF-1.4\nfirst"), "resume.pdf")
        path2, _ = storage.save_upload(io.BytesIO(b"%PDF-1.4\nsecond"), "resume.pdf")

        assert path1 != path2
        assert path1.read_bytes() == b"%PDF-1.4\nfirst"
        assert path2.read_bytes() == b"%PDF-1.4\nsecond"
