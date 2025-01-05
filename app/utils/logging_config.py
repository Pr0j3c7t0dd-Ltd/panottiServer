import json
import logging
import logging.config
import os
import sys
import uuid
from datetime import datetime
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from typing import Any

from app.models.recording.events import RecordingEvent, RecordingStartRequest, RecordingEndRequest
from app.plugins.events.models import Event, EventContext


class JSONFormatter(logging.Formatter):
    def __init__(self) -> None:
        super().__init__()

    def _serialize_object(self, obj: Any) -> Any:
        if isinstance(obj, (RecordingEvent, RecordingStartRequest, RecordingEndRequest, Event)):
            return obj.model_dump()
        if isinstance(obj, EventContext):
            return obj.model_dump()
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, Exception):
            return str(obj)
        return str(obj)

    def format(self, record: logging.LogRecord) -> str:
        log_record = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
            "request_id": getattr(record, "req_id", str(uuid.uuid4())),
            "levelname": record.levelname,
        }

        custom_fields = ["req_headers", "req_method", "req_path", "req_task"]
        for field in custom_fields:
            if hasattr(record, field):
                log_key = field[4:] if field.startswith("req_") else field
                log_record[log_key] = getattr(record, field)

        if hasattr(record, "extra"):
            for key, value in record.extra.items():
                log_record[key] = value

        if record.exc_info:
            log_record["exc_info"] = self.formatException(record.exc_info)

        return json.dumps(log_record, default=self._serialize_object)


def generate_request_id() -> str:
    return str(uuid.uuid4())


def setup_logging() -> None:
    """Configure logging for the application."""
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    logging_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "json": {
                "()": "app.utils.logging_config.JSONFormatter",
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "json",
                "stream": "ext://sys.stdout"
            },
            "file": {
                "class": "logging.FileHandler",
                "formatter": "json",
                "filename": "logs/app.log",
                "mode": "a"
            }
        },
        "root": {
            "level": os.getenv("LOG_LEVEL", "INFO"),
            "handlers": ["console", "file"]
        }
    }

    # Configure root logger first
    root = logging.getLogger()
    root.handlers = []  # Remove any existing handlers
    
    # Apply configuration
    logging.config.dictConfig(logging_config)


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance.
    
    This will return a logger that inherits settings from the root logger,
    including log level and handlers.
    """
    return logging.getLogger(name)


if __name__ == "__main__":
    setup_logging()
    logger = get_logger(__name__)

    request_headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer token",
    }

    logger.info("Processing request", extra={"req_headers": request_headers})
