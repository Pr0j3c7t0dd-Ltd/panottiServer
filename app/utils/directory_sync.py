from fastapi import FastAPI
from pydantic import BaseModel
import time
import shutil
import os
import logging
from pathlib import Path
from watchdog.observers import Observer  # type: ignore
from watchdog.events import FileSystemEventHandler
from typing import Dict, List
from dotenv import load_dotenv
import json

# Configure logging
logger = logging.getLogger(__name__)

class FileHandler(FileSystemEventHandler):
    def __init__(self, source_dir: Path, destination_dir: Path):
        self.source_dir = source_dir
        self.destination_dir = destination_dir
        logger.debug(
            "Initializing FileHandler",
            extra={
                "source_dir": str(source_dir),
                "destination_dir": str(destination_dir)
            }
        )
        super().__init__()

    def on_created(self, event):
        if event.is_directory:
            logger.debug(
                "Ignoring directory creation event",
                extra={"path": event.src_path}
            )
            return

        source_path = Path(event.src_path)
        if not source_path.exists():
            logger.warning(
                "Source file does not exist",
                extra={"source_path": str(source_path)}
            )
            return

        relative_path = source_path.relative_to(self.source_dir)
        destination_path = self.destination_dir / relative_path

        logger.debug(
            "Processing file creation event",
            extra={
                "source_path": str(source_path),
                "destination_path": str(destination_path)
            }
        )

        try:
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            logger.debug(
                "Created destination directory",
                extra={"directory": str(destination_path.parent)}
            )

            time.sleep(0.5)  # Small delay to ensure file is fully written
            shutil.copy2(source_path, destination_path)
            logger.info(
                "Successfully copied file",
                extra={
                    "source": str(source_path),
                    "destination": str(destination_path)
                }
            )
        except Exception as e:
            logger.error(
                "Error copying file",
                extra={
                    "source": str(source_path),
                    "destination": str(destination_path),
                    "error": str(e)
                }
            )

def resolve_path(path_str: str, app_root: Path) -> Path:
    """
    Resolve a path string to an absolute Path object.
    If the path is relative, it will be resolved relative to the app root.
    """
    path = Path(path_str)
    resolved_path = path.resolve() if path.is_absolute() else (app_root / path).resolve()
    logger.debug(
        "Resolved path",
        extra={
            "original_path": path_str,
            "app_root": str(app_root),
            "resolved_path": str(resolved_path)
        }
    )
    return resolved_path

class DirectorySync:
    def __init__(self, app_root: Path):
        self.app_root = app_root
        self.observers: Dict[str, Observer] = {} # type: ignore
        self.monitored_dirs: List[dict] = []
        self.enabled = os.getenv('DIRECTORY_SYNC_ENABLED', 'false').lower() == 'true'
        logger.debug(
            "Initializing DirectorySync",
            extra={
                "app_root": str(app_root),
                "enabled": self.enabled
            }
        )
        if self.enabled:
            self.load_directory_pairs()
        else:
            logger.info("Directory sync is disabled")

    def load_directory_pairs(self):
        """Load directory pairs from environment variable and resolve paths"""
        dir_pairs_str = os.getenv('DIRECTORY_SYNC_PAIRS', '[]')
        logger.debug(
            "Loading directory pairs from environment",
            extra={"raw_config": dir_pairs_str}
        )
        
        try:
            raw_dirs = json.loads(dir_pairs_str)
            self.monitored_dirs = []
            for pair in raw_dirs:
                logger.debug(
                    "Processing directory pair",
                    extra={
                        "source": pair['source'],
                        "destination": pair['destination']
                    }
                )
                resolved_pair = {
                    'source': str(resolve_path(pair['source'], self.app_root)),
                    'destination': str(resolve_path(pair['destination'], self.app_root)),
                    'original_source': pair['source'],
                    'original_destination': pair['destination']
                }
                self.monitored_dirs.append(resolved_pair)
                logger.debug(
                    "Resolved directory pair",
                    extra=resolved_pair
                )
        except json.JSONDecodeError:
            logger.error(
                "Failed to parse DIRECTORY_SYNC_PAIRS environment variable",
                extra={"raw_config": dir_pairs_str}
            )
            self.monitored_dirs = []
        except Exception as e:
            logger.error(
                "Error processing directory pairs",
                extra={
                    "error": str(e),
                    "raw_config": dir_pairs_str
                }
            )
            self.monitored_dirs = []

    def start_monitoring(self):
        """Start monitoring all directory pairs"""
        if not self.enabled:
            logger.info("Directory sync is disabled, not starting monitoring")
            return

        logger.info(
            "Starting directory monitoring",
            extra={"pairs_count": len(self.monitored_dirs)}
        )
        
        for dir_pair in self.monitored_dirs:
            source = Path(dir_pair['source'])
            destination = Path(dir_pair['destination'])

            logger.debug(
                "Setting up monitoring for directory pair",
                extra={
                    "source": str(source),
                    "destination": str(destination)
                }
            )

            try:
                source.mkdir(parents=True, exist_ok=True)
                destination.mkdir(parents=True, exist_ok=True)
                logger.debug(
                    "Created directories",
                    extra={
                        "source": str(source),
                        "destination": str(destination)
                    }
                )
            except Exception as e:
                logger.error(
                    "Error creating directories",
                    extra={
                        "source": str(source),
                        "destination": str(destination),
                        "error": str(e)
                    }
                )
                continue

            observer = Observer()
            handler = FileHandler(source, destination)
            observer.schedule(handler, str(source), recursive=True)
            observer.start()
            
            self.observers[str(source)] = observer
            logger.info(
                "Started monitoring directory pair",
                extra={
                    "source": dir_pair['original_source'],
                    "destination": dir_pair['original_destination'],
                    "resolved_source": str(source),
                    "resolved_destination": str(destination)
                }
            )

    def stop_monitoring(self):
        """Stop all observers"""
        logger.info(
            "Stopping directory monitoring",
            extra={"observer_count": len(self.observers)}
        )
        
        for source, observer in self.observers.items():
            logger.debug(
                "Stopping observer",
                extra={"source": source}
            )
            observer.stop()
            
        for source, observer in self.observers.items():
            logger.debug(
                "Joining observer thread",
                extra={"source": source}
            )
            observer.join()
            
        self.observers.clear()
        logger.info("Directory monitoring stopped")

class MonitorStatus(BaseModel):
    status: str
    monitored_directories: List[Dict]

    model_config = {
        'from_attributes': True
    }