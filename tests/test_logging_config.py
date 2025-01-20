import json
import logging
import os
import sys
from datetime import datetime, UTC
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch
import pytest
from pydantic import BaseModel

from app.utils.logging_config import JSONFormatter, setup_logging, get_logger, generate_request_id


@pytest.fixture
def test_model():
    class TestModel(BaseModel):
        name: str
        value: int
    return TestModel(name="test", value=42)


def test_json_formatter_serialize_object(test_model):
    formatter = JSONFormatter()
    
    # Test Pydantic model serialization
    assert formatter._serialize_object(test_model) == {"name": "test", "value": 42}
    
    # Test datetime serialization
    dt = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
    assert formatter._serialize_object(dt) == "2024-01-01T12:00:00+00:00"
    
    # Test Exception serialization
    exc = ValueError("test error")
    assert formatter._serialize_object(exc) == "test error"
    
    # Test list/dict serialization
    test_list = [1, 2, 3]
    test_dict = {"a": 1, "b": 2}
    assert formatter._serialize_object(test_list) == [1, 2, 3]
    assert formatter._serialize_object(test_dict) == {"a": 1, "b": 2}
    
    # Test other object serialization
    class CustomObj:
        def __str__(self):
            return "custom_obj"
    
    obj = CustomObj()
    assert formatter._serialize_object(obj) == "custom_obj"


def test_json_formatter_format():
    formatter = JSONFormatter()
    
    # Test without exception info
    record = logging.LogRecord(
        name="test_logger",
        level=logging.INFO,
        pathname="test.py",
        lineno=1,
        msg="Test message",
        args=(),
        exc_info=None
    )
    
    # Add custom attributes
    record.req_id = "test-id"
    record.custom_field = "value"
    record.additional = "extra_value"
    
    formatted = formatter.format(record)
    log_dict = json.loads(formatted)
    
    assert "timestamp" in log_dict
    assert log_dict["level"] == "INFO"
    assert log_dict["message"] == "Test message"
    assert log_dict["logger"] == "test_logger"
    assert log_dict["request_id"] == "test-id"
    assert log_dict["custom_field"] == "value"
    assert log_dict["additional"] == "extra_value"
    
    # Test with exception info
    try:
        raise ValueError("Test error")
    except ValueError:
        record.exc_info = sys.exc_info()
        formatted = formatter.format(record)
        log_dict = json.loads(formatted)
        assert "exc_info" in log_dict
        assert "ValueError: Test error" in log_dict["exc_info"]


def test_generate_request_id():
    request_id = generate_request_id()
    assert isinstance(request_id, str)
    # Verify UUID format
    assert len(request_id.split("-")) == 5


@pytest.fixture
def cleanup_logs():
    yield
    # Clean up test log files
    log_file = Path("logs/app.log")
    if log_file.exists():
        log_file.unlink()
    log_dir = Path("logs")
    if log_dir.exists():
        log_dir.rmdir()


def test_setup_logging(cleanup_logs):
    with patch.dict(os.environ, {"LOG_LEVEL": "DEBUG"}):
        setup_logging()
        
        # Verify log directory creation
        assert Path("logs").exists()
        
        # Verify root logger configuration
        root_logger = logging.getLogger()
        assert len(root_logger.handlers) == 2
        assert any(isinstance(h, logging.StreamHandler) for h in root_logger.handlers)
        assert any(isinstance(h, logging.FileHandler) for h in root_logger.handlers)
        assert root_logger.level == logging.DEBUG


@pytest.fixture(autouse=True)
def reset_logging():
    """Reset logging configuration before each test."""
    yield
    root = logging.getLogger()
    root.handlers = []
    root.setLevel(logging.WARNING)
    for logger in logging.Logger.manager.loggerDict.values():
        if isinstance(logger, logging.Logger):
            logger.handlers = []
            logger.level = logging.NOTSET


def test_get_logger(reset_logging):
    # Setup logging with a specific level
    with patch.dict(os.environ, {"LOG_LEVEL": "INFO"}):
        setup_logging()
        
        logger = get_logger("test_module")
        assert isinstance(logger, logging.Logger)
        assert logger.name == "test_module"
        
        # Verify logger inherits root logger settings
        root_logger = logging.getLogger()
        assert logger.getEffectiveLevel() == root_logger.level
        assert all(isinstance(h, type(rh)) for h, rh in zip(logger.handlers, root_logger.handlers))


@pytest.mark.asyncio  # Mark as async test
async def test_main_execution():
    with patch("app.utils.logging_config.setup_logging") as mock_setup, \
         patch("app.utils.logging_config.get_logger") as mock_get_logger, \
         patch("app.utils.logging_config.datetime") as mock_datetime, \
         patch("app.core.events.bus.EventBus._cleanup_old_events") as mock_cleanup:  # Mock the cleanup coroutine
        
        # Make cleanup return immediately
        mock_cleanup.return_value = None
        
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        
        # Mock datetime.now() to return a timezone-aware datetime
        mock_now = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
        mock_datetime.now.return_value = mock_now
        mock_datetime.UTC = UTC
        
        # Create a mock module with required imports
        mock_module = {
            "__name__": "__main__",
            "json": json,
            "logging": logging,
            "datetime": mock_datetime,
            "uuid": MagicMock(),
            "os": os,
            "Path": Path,
            "Any": Any,
            "BaseModel": BaseModel,
            "setup_logging": mock_setup,
            "get_logger": mock_get_logger,
            "JSONFormatter": JSONFormatter,
            "generate_request_id": generate_request_id,
        }
        
        # Execute the main block
        with patch.dict("sys.modules", {"__main__": MagicMock(__name__="__main__")}):
            with open("app/utils/logging_config.py") as f:
                exec(f.read(), mock_module) 