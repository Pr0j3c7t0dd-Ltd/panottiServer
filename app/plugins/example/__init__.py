"""Example plugin."""

import logging
import sys

logger = logging.getLogger(__name__)
logger.debug(
    "Importing example plugin module",
    extra={
        "module_name": __name__,
        "module_file": __file__,
        "sys_path": sys.path,
    },
)

from .plugin import ExamplePlugin as Plugin

logger.debug(
    "Imported plugin class",
    extra={
        "plugin_class": Plugin.__name__,
        "plugin_bases": [base.__name__ for base in Plugin.__bases__],
    },
)

__all__ = ["Plugin"]
