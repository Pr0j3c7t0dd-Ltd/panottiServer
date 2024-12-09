import logging
import json
import os
from datetime import datetime

class JSONFormatter(logging.Formatter):
    """
    Custom JSON formatter for structured logging.
    """
    def format(self, record):
        log_record = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
        }
        
        # Add extra fields if they exist
        if hasattr(record, 'extra'):
            for key, value in record.extra.items():
                log_record[key] = value
                
        # Add exception info if present
        if record.exc_info:
            log_record['exc_info'] = self.formatException(record.exc_info)
            
        return json.dumps(log_record, indent=2, default=str)

def setup_logging():
    """
    Configure logging with JSON formatting and appropriate handlers.
    """
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Remove existing handlers
    logger.handlers = []
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(JSONFormatter())
    logger.addHandler(console_handler)
    
    # Create file handler
    file_handler = logging.FileHandler("logs/app.log")
    file_handler.setFormatter(JSONFormatter())
    logger.addHandler(file_handler)
