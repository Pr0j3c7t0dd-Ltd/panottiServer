import logging
import json
import os
from datetime import datetime
from logging.handlers import TimedRotatingFileHandler

class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_record = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
        }
        
        # Get any extra attributes from the record
        extras = {
            key: value
            for key, value in record.__dict__.items()
            if key not in ['args', 'asctime', 'created', 'exc_info', 'exc_text', 
                         'filename', 'funcName', 'levelname', 'levelno', 'lineno', 
                         'module', 'msecs', 'msg', 'name', 'pathname', 'process', 
                         'processName', 'relativeCreated', 'stack_info', 'thread', 
                         'threadName']
        }
        log_record.update(extras)
                
        # Add exception info if present
        if record.exc_info:
            log_record['exc_info'] = self.formatException(record.exc_info)
            
        return json.dumps(log_record)

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
    log_level = os.getenv("LOG_LEVEL", "INFO")
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    
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
