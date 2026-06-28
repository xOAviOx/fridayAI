"""Tests for the five file-system skills.

Uses tmp_path (pytest fixture) so nothing touches the real filesystem.
All tests run without audio/LLM/TTS extras — pure Python stdlib.
"""

from __future__ import annotations

import pytest

from friday.skills.builtin import (
    create_folder,
    delete_file,
    delete_folder,
    read_file,
    write_file,
)


# --------------------------------------------------------------------------- #
# write_file                                                                  #
# --------------------------------------------------------------------------- #


class TestWriteFile:
    def test_creates_new_file(self, tmp_path):
        target = tmp_path / "hello.txt"
        result = write_file(str(target), "hello world")
        assert target.exists()
        assert target.read_text() == "hello world"
        assert "hello.txt" in result

    def test_overwrites_existing_file(self, tmp_path):
        target = tmp_path / "data.txt"
        target.write_text("old content")
        write_file(str(target), "new content")
        assert target.read_text() == "new content"

    def test_creates_missing_parent_dirs(self, tmp_path):
        target = tmp_path / "a" / "b" / "c.txt"
        write_file(str(target), "deep")
        assert target.read_text() == "deep"

    def test_empty_content_creates_empty_file(self, tmp_path):
        target = tmp_path / "empty.txt"
        write_file(str(target), "")
        assert target.exists()
        assert target.read_text() == ""

    def test_returns_char_and_line_count(self, tmp_path):
        target = tmp_path / "lines.txt"
        result = write_file(str(target), "line1\nline2\nline3")
        assert "3 lines" in result

    def test_raises_on_directory_path(self, tmp_path):
        with pytest.raises(IsADirectoryError):
            write_file(str(tmp_path), "content")  # tmp_path is a dir

    def test_multiline_unicode(self, tmp_path):
        target = tmp_path / "unicode.txt"
        content = "こんにちは\n世界\n"
        write_file(str(target), content)
        assert target.read_text(encoding="utf-8") == content


# --------------------------------------------------------------------------- #
# read_file                                                                   #
# --------------------------------------------------------------------------- #


class TestReadFile:
    def test_reads_existing_file(self, tmp_path):
        f = tmp_path / "note.txt"
        f.write_text("some content")
        assert read_file(str(f)) == "some content"

    def test_raises_on_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            read_file(str(tmp_path / "nope.txt"))

    def test_raises_on_directory(self, tmp_path):
        with pytest.raises(IsADirectoryError):
            read_file(str(tmp_path))

    def test_truncates_large_file(self, tmp_path):
        f = tmp_path / "big.txt"
        # Write 33 KB so it's over the 32 KB cap.
        f.write_bytes(b"x" * 33_792)
        result = read_file(str(f))
        assert "truncated" in result
        assert len(result) < 33_792 + 200  # content + notice, not full file

    def test_reads_unicode(self, tmp_path):
        f = tmp_path / "u.txt"
        f.write_text("日本語テスト", encoding="utf-8")
        assert read_file(str(f)) == "日本語テスト"

    def test_round_trips_with_write_file(self, tmp_path):
        f = tmp_path / "rt.txt"
        original = "Round-trip test.\nSecond line.\n"
        write_file(str(f), original)
        assert read_file(str(f)) == original


# --------------------------------------------------------------------------- #
# delete_file                                                                 #
# --------------------------------------------------------------------------- #


class TestDeleteFile:
    def test_deletes_existing_file(self, tmp_path):
        f = tmp_path / "bye.txt"
        f.write_text("bye")
        result = delete_file(str(f))
        assert not f.exists()
        assert "bye.txt" in result

    def test_raises_on_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            delete_file(str(tmp_path / "ghost.txt"))

    def test_raises_on_directory(self, tmp_path):
        d = tmp_path / "adir"
        d.mkdir()
        with pytest.raises(IsADirectoryError):
            delete_file(str(d))


# --------------------------------------------------------------------------- #
# create_folder                                                               #
# --------------------------------------------------------------------------- #


class TestCreateFolder:
    def test_creates_new_directory(self, tmp_path):
        d = tmp_path / "new_dir"
        result = create_folder(str(d))
        assert d.is_dir()
        assert "new_dir" in result

    def test_creates_nested_directories(self, tmp_path):
        d = tmp_path / "a" / "b" / "c"
        create_folder(str(d))
        assert d.is_dir()

    def test_idempotent_on_existing_directory(self, tmp_path):
        d = tmp_path / "existing"
        d.mkdir()
        # Should not raise.
        result = create_folder(str(d))
        assert d.is_dir()
        assert "existing" in result

    def test_raises_if_path_is_a_file(self, tmp_path):
        f = tmp_path / "afile"
        f.write_text("data")
        with pytest.raises(FileExistsError):
            create_folder(str(f))


# --------------------------------------------------------------------------- #
# delete_folder                                                               #
# --------------------------------------------------------------------------- #


class TestDeleteFolder:
    def test_deletes_empty_directory(self, tmp_path):
        d = tmp_path / "empty"
        d.mkdir()
        result = delete_folder(str(d))
        assert not d.exists()
        assert "empty" in result

    def test_deletes_directory_tree(self, tmp_path):
        root = tmp_path / "tree"
        (root / "sub").mkdir(parents=True)
        (root / "sub" / "file.txt").write_text("data")
        (root / "other.txt").write_text("other")
        delete_folder(str(root))
        assert not root.exists()

    def test_raises_on_missing_path(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            delete_folder(str(tmp_path / "ghost"))

    def test_raises_on_file_path(self, tmp_path):
        f = tmp_path / "afile.txt"
        f.write_text("content")
        with pytest.raises(NotADirectoryError):
            delete_folder(str(f))

    def test_refuses_home_directory(self):
        from pathlib import Path
        with pytest.raises(ValueError, match="protected"):
            delete_folder(str(Path.home()))

    def test_skill_is_destructive(self):
        """delete_folder must be flagged destructive so the safety gate gates it."""
        from friday.skills import default_registry
        meta = default_registry.get("delete_folder")
        assert meta is not None
        assert meta.destructive is True

    def test_write_and_delete_folder_round_trip(self, tmp_path):
        """Create files inside a folder, then delete the whole folder."""
        d = tmp_path / "project"
        create_folder(str(d))
        write_file(str(d / "README.md"), "# Project\n")
        write_file(str(d / "src" / "main.py"), "print('hello')\n")
        assert (d / "README.md").exists()
        assert (d / "src" / "main.py").exists()
        delete_folder(str(d))
        assert not d.exists()
