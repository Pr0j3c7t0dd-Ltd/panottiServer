import logging
import json
import os
import uuid
from datetime import datetime
from logging.handlers import TimedRotatingFileHandler

class JSONFormatter(logging.Formatter):
    def format(self, record):
        # Create the base log record
        log_record = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
            "request_id": getattr(record, 'req_id', str(uuid.uuid4()))
        }

        # Add our custom request fields
        custom_fields = ['req_headers', 'req_method', 'req_path', 'req_task']
        for field in custom_fields:
            if hasattr(record, field):
                # Remove the 'req_' prefix in the output
                log_key = field[4:] if field.startswith('req_') else field
                log_record[log_key] = getattr(record, field)

        # Add any additional extra attributes that might be present
        for key, value in record.__dict__.items():
            if (key not in log_record and 
                key not in ['args', 'exc_info', 'exc_text', 'stack_info', 'lineno', 
                           'funcName', 'created', 'msecs', 'relativeCreated', 'levelno', 
                           'pathname', 'filename', 'module', 'processName', 'threadName', 
                           'thread', 'process', 'msg', 'name']):
                log_record[key] = value

        # Add exception information if present
        if record.exc_info:
            log_record['exc_info'] = self.formatException(record.exc_info)

        return json.dumps(log_record)

def generate_request_id():
    """Generate a unique request ID"""
    return str(uuid.uuid4())

def setup_logging():
    """
    Configure logging with JSON formatting, log rotation, and appropriate handlers.
    Environment variables:
    - LOG_LEVEL: Logging level (default: INFO)
    - LOG_ENABLED: Enable/disable logging (default: True)
    - LOG_RETENTION_DAYS: Days to keep logs, 0 for infinite (default: 30)
    """
    # Check if logging is enabled
    if os.getenv("LOG_ENABLED", "true").lower() == "false":
        logging.getLogger().handlers = []
        return

    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)

    # Configure root logger
    logger = logging.getLogger()
    log_level = os.getenv("LOG_LEVEL", "DEBUG")
    print(f"Setting log level to: {log_level}")
    logger.setLevel(getattr(logging, log_level.upper(), logging.DEBUG))

    # Remove existing handlers
    logger.handlers = []

    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(JSONFormatter())
    logger.addHandler(console_handler)

    # Create rotating file handler
    log_file = "logs/app.log"
    retention_days = int(os.getenv("LOG_RETENTION_DAYS", "30"))
    file_handler = TimedRotatingFileHandler(
        log_file,
        when="midnight",
        interval=1,  # Rotate daily
        backupCount=retention_days if retention_days > 0 else 0,  # Keep X days of logs, 0 for infinite
        encoding="utf-8"
    )
    file_handler.setFormatter(JSONFormatter())
    logger.addHandler(file_handler)

# Example usage
if __name__ == "__main__":
    setup_logging()
    logger = logging.getLogger()

    # Simulate an HTTP request with headers
    request_headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer token",
    }

    # Log with headers
    logger.info("Processing request", extra={"req_headers": request_headers})
