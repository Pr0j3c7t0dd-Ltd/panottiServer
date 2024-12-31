"""Utility functions for the application."""
import logging
from logging import Logger
from typing import Any


def setup_logging() -> logging.Logger:
    """Set up logging configuration."""
    logger = logging.getLogger("app")
    logger.setLevel(logging.INFO)

    # Create console handler with a higher log level
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Create formatter and add it to the handlers
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(console_handler)

    return logger


def get_logger(name: str) -> Logger:
    """Get a logger with the given name."""
    return logging.getLogger(name)
