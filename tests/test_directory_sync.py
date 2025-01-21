import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from watchdog.observers import Observer

from app.utils.directory_sync import DirectorySync, FileHandler, resolve_path


@pytest.fixture
def temp_source_dir(tmp_path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    return source_dir


@pytest.fixture
def temp_dest_dir(tmp_path):
    dest_dir = tmp_path / "dest"
    dest_dir.mkdir()
    return dest_dir


@pytest.fixture
def app_root(tmp_path):
    return tmp_path


def test_resolve_path_absolute(app_root):
    abs_path = "/absolute/path"
    result = resolve_path(abs_path, app_root)
    assert result == Path(abs_path).resolve()


def test_resolve_path_relative(app_root):
    rel_path = "relative/path"
    result = resolve_path(rel_path, app_root)
    assert result == (app_root / rel_path).resolve()


def test_file_handler_init(temp_source_dir, temp_dest_dir):
    handler = FileHandler(temp_source_dir, temp_dest_dir)
    assert handler.source_dir == temp_source_dir
    assert handler.destination_dir == temp_dest_dir


def test_file_handler_on_created_directory_event(temp_source_dir, temp_dest_dir):
    handler = FileHandler(temp_source_dir, temp_dest_dir)
    event = MagicMock(is_directory=True, src_path=str(temp_source_dir / "subdir"))
    handler.on_created(event)
    assert not (temp_dest_dir / "subdir").exists()


def test_file_handler_on_created_file_event(temp_source_dir, temp_dest_dir):
    handler = FileHandler(temp_source_dir, temp_dest_dir)
    source_file = temp_source_dir / "test.txt"
    source_file.write_text("test content")

    event = MagicMock(is_directory=False, src_path=str(source_file))

    handler.on_created(event)
    dest_file = temp_dest_dir / "test.txt"
    assert dest_file.exists()
    assert dest_file.read_text() == "test content"


def test_file_handler_on_created_nested_file(temp_source_dir, temp_dest_dir):
    handler = FileHandler(temp_source_dir, temp_dest_dir)
    nested_dir = temp_source_dir / "nested"
    nested_dir.mkdir()
    source_file = nested_dir / "test.txt"
    source_file.write_text("nested content")

    event = MagicMock(is_directory=False, src_path=str(source_file))

    handler.on_created(event)
    dest_file = temp_dest_dir / "nested" / "test.txt"
    assert dest_file.exists()
    assert dest_file.read_text() == "nested content"


def test_file_handler_on_created_nonexistent_file(temp_source_dir, temp_dest_dir):
    handler = FileHandler(temp_source_dir, temp_dest_dir)
    nonexistent_file = temp_source_dir / "nonexistent.txt"

    event = MagicMock(is_directory=False, src_path=str(nonexistent_file))

    handler.on_created(event)
    assert not (temp_dest_dir / "nonexistent.txt").exists()


def test_file_handler_on_created_copy_error(temp_source_dir, temp_dest_dir):
    handler = FileHandler(temp_source_dir, temp_dest_dir)
    source_file = temp_source_dir / "test.txt"
    source_file.write_text("test content")

    event = MagicMock(is_directory=False, src_path=str(source_file))

    with patch("shutil.copy2") as mock_copy2:
        mock_copy2.side_effect = PermissionError("Permission denied")
        handler.on_created(event)
        assert not (temp_dest_dir / "test.txt").exists()


def test_directory_sync_init_disabled(app_root):
    with patch.dict(os.environ, {"DIRECTORY_SYNC_ENABLED": "false"}):
        sync = DirectorySync(app_root)
        assert not sync.enabled
        assert not sync.monitored_dirs


def test_directory_sync_init_enabled_empty_pairs(app_root):
    with patch.dict(
        os.environ, {"DIRECTORY_SYNC_ENABLED": "true", "DIRECTORY_SYNC_PAIRS": "[]"}
    ):
        sync = DirectorySync(app_root)
        assert sync.enabled
        assert not sync.monitored_dirs


def test_directory_sync_init_enabled_with_pairs(app_root):
    pairs = [
        {"source": "src", "destination": "dest"},
        {"source": "/abs/src", "destination": "/abs/dest"},
    ]
    with patch.dict(
        os.environ,
        {"DIRECTORY_SYNC_ENABLED": "true", "DIRECTORY_SYNC_PAIRS": json.dumps(pairs)},
    ):
        sync = DirectorySync(app_root)
        assert sync.enabled
        assert len(sync.monitored_dirs) == 2


def test_directory_sync_invalid_json_pairs(app_root):
    with patch.dict(
        os.environ,
        {"DIRECTORY_SYNC_ENABLED": "true", "DIRECTORY_SYNC_PAIRS": "invalid json"},
    ):
        sync = DirectorySync(app_root)
        assert sync.enabled
        assert not sync.monitored_dirs


def test_directory_sync_pairs_with_error(app_root):
    with patch.dict(
        os.environ,
        {
            "DIRECTORY_SYNC_ENABLED": "true",
            "DIRECTORY_SYNC_PAIRS": json.dumps([{"invalid": "pair"}]),
        },
    ):
        sync = DirectorySync(app_root)
        assert sync.enabled
        assert not sync.monitored_dirs


@pytest.mark.parametrize("enabled", [True, False])
def test_directory_sync_start_monitoring(app_root, enabled):
    pairs = [{"source": "src", "destination": "dest"}]
    with patch.dict(
        os.environ,
        {
            "DIRECTORY_SYNC_ENABLED": str(enabled).lower(),
            "DIRECTORY_SYNC_PAIRS": json.dumps(pairs),
        },
    ):
        with patch("app.utils.directory_sync.Observer") as mock_observer:
            sync = DirectorySync(app_root)
            sync.start_monitoring()

            if enabled:
                assert mock_observer.return_value.start.called
            else:
                assert not mock_observer.return_value.start.called


def test_directory_sync_start_monitoring_mkdir_error(app_root):
    pairs = [{"source": "src", "destination": "dest"}]
    with patch.dict(
        os.environ,
        {"DIRECTORY_SYNC_ENABLED": "true", "DIRECTORY_SYNC_PAIRS": json.dumps(pairs)},
    ):
        with patch("pathlib.Path.mkdir") as mock_mkdir:
            mock_mkdir.side_effect = PermissionError("Permission denied")
            sync = DirectorySync(app_root)
            sync.start_monitoring()
            assert not sync.observers


def test_directory_sync_stop_monitoring(app_root):
    pairs = [{"source": "src", "destination": "dest"}]
    with patch.dict(
        os.environ,
        {"DIRECTORY_SYNC_ENABLED": "true", "DIRECTORY_SYNC_PAIRS": json.dumps(pairs)},
    ):
        sync = DirectorySync(app_root)
        mock_observer = MagicMock(spec=Observer)
        sync.observers["src"] = mock_observer

        sync.stop_monitoring()

        mock_observer.stop.assert_called_once()
        mock_observer.join.assert_called_once()
        assert not sync.observers


def test_directory_sync_stop_monitoring_timeout(app_root):
    """Test case where observer doesn't stop within timeout"""
    pairs = [{"source": "src", "destination": "dest"}]
    with patch.dict(
        os.environ,
        {"DIRECTORY_SYNC_ENABLED": "true", "DIRECTORY_SYNC_PAIRS": json.dumps(pairs)},
    ):
        sync = DirectorySync(app_root)
        mock_observer = MagicMock(spec=Observer)
        mock_observer.is_alive.return_value = True  # Observer didn't stop within timeout
        sync.observers["src"] = mock_observer

        sync.stop_monitoring()

        mock_observer.stop.assert_called_once()
        mock_observer.join.assert_called_once_with(timeout=10.0)
        assert not sync.observers


def test_directory_sync_stop_monitoring_error(app_root):
    """Test error handling in stop_monitoring"""
    pairs = [{"source": "src", "destination": "dest"}]
    with patch.dict(
        os.environ,
        {"DIRECTORY_SYNC_ENABLED": "true", "DIRECTORY_SYNC_PAIRS": json.dumps(pairs)},
    ):
        sync = DirectorySync(app_root)
        mock_observer = MagicMock(spec=Observer)
        mock_observer.stop.side_effect = Exception("Test error")
        sync.observers["src"] = mock_observer

        with pytest.raises(Exception, match="Test error"):
            sync.stop_monitoring()

        mock_observer.stop.assert_called_once()
