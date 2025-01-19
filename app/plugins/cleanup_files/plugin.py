"""Cleanup files plugin implementation."""

import asyncio
import os
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any

from app.core.events import ConcreteEventBus as EventBus
from app.core.events import Event, EventContext
from app.models.recording.events import RecordingEvent
from app.plugins.base import PluginBase, PluginConfig
from app.utils.logging_config import get_logger

logger = get_logger(__name__)

EventData = dict[str, Any] | RecordingEvent | Event


class CleanupFilesPlugin(PluginBase):
    """Plugin for cleaning up files after processing is complete."""

    def __init__(self, config: PluginConfig, event_bus: EventBus | None = None) -> None:
        """Initialize the cleanup files plugin.

        Args:
            config: Plugin configuration
            event_bus: Event bus for subscribing to events
        """
        super().__init__(config, event_bus)
        self._executor: ThreadPoolExecutor | None = None
        self._processing_lock = threading.Lock()

        # Get configured directories with defaults
        config_dict = config.config or {}
        self.include_dirs = config_dict.get("include_dirs", ["data"])
        self.exclude_dirs = config_dict.get("exclude_dirs", [])
        self.cleanup_delay = config_dict.get("cleanup_delay", 0)  # Default to no delay

        # Convert to Path objects
        self.include_dirs = [Path(d) for d in self.include_dirs]
        self.exclude_dirs = [Path(d) for d in self.exclude_dirs]

        logger.info(
            "Initializing cleanup files plugin",
            extra={
                "plugin_name": self.name,
                "include_dirs": [str(d) for d in self.include_dirs],
                "exclude_dirs": [str(d) for d in self.exclude_dirs],
                "config": config_dict,
                "event_bus_available": event_bus is not None,
            },
        )

    async def _initialize(self) -> None:
        """Initialize plugin."""
        if not self.event_bus:
            logger.warning(
                "No event bus available for plugin",
                extra={"plugin": self.name, "config": self.config.config},
            )
            return

        try:
            logger.debug(
                "Starting cleanup files plugin initialization",
                extra={"plugin": self.name, "thread_id": threading.get_ident()},
            )

            # Initialize thread pool for processing
            max_workers = 4  # Reasonable default for file operations
            self._executor = ThreadPoolExecutor(max_workers=max_workers)

            # Subscribe to desktop notification completed events
            await self.event_bus.subscribe(
                "desktop_notification.completed",
                self.handle_desktop_notification_completed,
            )

            logger.info(
                "Cleanup files plugin initialization complete",
                extra={
                    "plugin": self.name,
                    "subscribed_events": ["desktop_notification.completed"],
                    "handler": "handle_desktop_notification_completed",
                    "thread_pool_ready": self._executor is not None,
                },
            )

        except Exception as e:
            logger.error(
                "Failed to initialize cleanup files plugin",
                extra={
                    "plugin": self.name,
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "thread_id": threading.get_ident(),
                },
                exc_info=True,
            )
            raise

    async def _shutdown(self) -> None:
        """Shutdown plugin."""
        if not self.event_bus:
            return

        logger.debug(
            "Starting plugin shutdown",
            extra={
                "plugin": self.name,
                "thread_pool_active": self._executor is not None,
            },
        )

        # Unsubscribe from events
        await self.event_bus.unsubscribe(
            "desktop_notification.completed", self.handle_desktop_notification_completed
        )

        # Shutdown thread pool
        if self._executor:
            self._executor.shutdown(wait=True)
            logger.debug(
                "Thread pool shutdown complete",
                extra={"plugin": self.name, "thread_pool_id": id(self._executor)},
            )

        logger.info(
            "Cleanup files plugin shutdown complete", extra={"plugin": self.name}
        )

    async def handle_desktop_notification_completed(
        self, event_data: EventData
    ) -> None:
        """Handle desktop notification completed event."""
        event_id = str(uuid.uuid4())
        try:
            logger.debug(
                "Raw desktop notification completed event received",
                extra={
                    "plugin": self.name,
                    "event_id": event_id,
                    "event_type": type(event_data).__name__,
                    "event_data": str(event_data),
                    "thread_id": threading.get_ident(),
                },
            )

            # Extract data based on event type
            if isinstance(event_data, dict):
                data = event_data
                logger.debug(
                    "Processing dictionary event data",
                    extra={
                        "plugin": self.name,
                        "event_id": event_id,
                        "data_keys": list(data.keys()),
                    },
                )
            elif isinstance(event_data, (Event, RecordingEvent)):
                data = event_data.data if hasattr(event_data, "data") else {}
                logger.debug(
                    "Processing Event/RecordingEvent data",
                    extra={
                        "plugin": self.name,
                        "event_id": event_id,
                        "event_class": type(event_data).__name__,
                        "has_data": hasattr(event_data, "data"),
                    },
                )
            else:
                logger.error(
                    "Unsupported event type",
                    extra={
                        "plugin": self.name,
                        "event_id": event_id,
                        "event_type": type(event_data).__name__,
                        "supported_types": ["dict", "Event", "RecordingEvent"],
                    },
                )
                return

            # Extract recording ID
            recording_id = (
                data.get("recording_id")
                or data.get("data", {}).get("recording_id")
                or "unknown"
            )

            logger.debug(
                "Extracted recording ID",
                extra={
                    "plugin": self.name,
                    "event_id": event_id,
                    "recording_id": recording_id,
                    "data_path": "Found in: "
                    + (
                        "root.recording_id"
                        if "recording_id" in data
                        else "root.data.recording_id"
                        if "data" in data and "recording_id" in data["data"]
                        else "unknown"
                    ),
                },
            )

            if recording_id == "unknown":
                logger.warning(
                    "No recording ID in event data",
                    extra={
                        "plugin": self.name,
                        "event_id": event_id,
                        "event_data": str(data),
                    },
                )
                return

            # Clean up files
            cleaned_files = await self._cleanup_files(recording_id)

            # Emit completion event
            if self.event_bus:
                completion_event = Event(
                    name="cleanup_files.completed",
                    data={
                        "recording_id": recording_id,
                        "cleaned_files": cleaned_files,
                        "status": "completed",
                        "current_event": {
                            "cleanup_files": {
                                "status": "completed",
                                "timestamp": datetime.utcnow().isoformat(),
                                "cleaned_files": cleaned_files,
                            }
                        },
                    },
                    context=EventContext(
                        correlation_id=str(uuid.uuid4()),
                        timestamp=datetime.utcnow().isoformat(),
                        source_plugin=self.name,
                    ),
                )

                logger.debug(
                    "Publishing completion event",
                    extra={
                        "plugin": self.name,
                        "event_id": event_id,
                        "event_name": completion_event.name,
                        "recording_id": recording_id,
                        "num_cleaned_files": len(cleaned_files),
                        "cleaned_files": cleaned_files,
                    },
                )
                await self.event_bus.publish(completion_event)

        except Exception as e:
            error_msg = f"Failed to handle desktop notification completion: {e!s}"
            logger.error(
                error_msg,
                extra={
                    "plugin": self.name,
                    "event_id": event_id,
                    "recording_id": recording_id
                    if "recording_id" in locals()
                    else "unknown",
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "stack_info": True,
                },
                exc_info=True,
            )

            if self.event_bus and "recording_id" in locals():
                # Emit error event
                error_event = Event(
                    name="cleanup_files.error",
                    data={
                        "recording_id": recording_id,
                        "error": str(e),
                        "current_event": {
                            "cleanup_files": {
                                "status": "error",
                                "timestamp": datetime.utcnow().isoformat(),
                                "error": str(e),
                            }
                        },
                    },
                    context=EventContext(
                        correlation_id=str(uuid.uuid4()),
                        timestamp=datetime.utcnow().isoformat(),
                        source_plugin=self.name,
                    ),
                )
                await self.event_bus.publish(error_event)

    async def _cleanup_files(self, recording_id: str) -> list[str]:
        """Clean up files for a recording ID.

        Args:
            recording_id: The recording ID to clean up files for

        Returns:
            List of cleaned up file paths
        """
        cleaned_files = []
        scan_start_time = datetime.utcnow()

        try:
            logger.debug(
                "Starting file cleanup",
                extra={
                    "plugin": self.name,
                    "recording_id": recording_id,
                    "include_dirs": [str(d) for d in self.include_dirs],
                    "exclude_dirs": [str(d) for d in self.exclude_dirs],
                    "cleanup_delay": self.cleanup_delay,
                    "scan_start_time": scan_start_time.isoformat(),
                },
            )

            if self.cleanup_delay > 0:
                logger.debug(
                    f"Waiting {self.cleanup_delay} seconds before cleanup",
                    extra={
                        "plugin": self.name,
                        "recording_id": recording_id,
                        "cleanup_delay": self.cleanup_delay,
                    },
                )
                await asyncio.sleep(self.cleanup_delay)

            # Search through include directories
            for include_dir in self.include_dirs:
                if not include_dir.exists():
                    logger.warning(
                        f"Include directory does not exist: {include_dir}",
                        extra={
                            "plugin": self.name,
                            "recording_id": recording_id,
                            "directory": str(include_dir),
                        },
                    )
                    continue

                logger.debug(
                    f"Scanning directory: {include_dir}",
                    extra={
                        "plugin": self.name,
                        "recording_id": recording_id,
                        "directory": str(include_dir),
                    },
                )

                # Walk through directory
                for root, dirs, files in os.walk(str(include_dir)):
                    root_path = Path(root)

                    # Skip excluded directories
                    if any(
                        root_path == excl or excl in root_path.parents
                        for excl in self.exclude_dirs
                    ):
                        logger.debug(
                            f"Skipping excluded directory: {root_path}",
                            extra={
                                "plugin": self.name,
                                "recording_id": recording_id,
                                "directory": str(root_path),
                                "excluded_by": [
                                    str(excl)
                                    for excl in self.exclude_dirs
                                    if excl == root_path or excl in root_path.parents
                                ],
                            },
                        )
                        continue

                    logger.debug(
                        f"Scanning files in: {root_path}",
                        extra={
                            "plugin": self.name,
                            "recording_id": recording_id,
                            "directory": str(root_path),
                            "num_files": len(files),
                        },
                    )

                    # Find and remove files matching recording ID
                    for file in files:
                        if recording_id in file:
                            file_path = root_path / file
                            try:
                                file_path.unlink()
                                cleaned_files.append(str(file_path))
                                logger.debug(
                                    f"Removed file: {file_path}",
                                    extra={
                                        "plugin": self.name,
                                        "recording_id": recording_id,
                                        "file": str(file_path),
                                        "file_size": file_path.stat().st_size
                                        if file_path.exists()
                                        else None,
                                    },
                                )
                            except Exception as e:
                                logger.error(
                                    f"Failed to remove file: {file_path}",
                                    extra={
                                        "plugin": self.name,
                                        "recording_id": recording_id,
                                        "file": str(file_path),
                                        "error": str(e),
                                        "error_type": type(e).__name__,
                                    },
                                    exc_info=True,
                                )

            scan_end_time = datetime.utcnow()
            scan_duration = (scan_end_time - scan_start_time).total_seconds()

            logger.info(
                "Cleanup complete",
                extra={
                    "plugin": self.name,
                    "recording_id": recording_id,
                    "num_cleaned_files": len(cleaned_files),
                    "cleaned_files": cleaned_files,
                    "scan_duration_seconds": scan_duration,
                    "scan_start_time": scan_start_time.isoformat(),
                    "scan_end_time": scan_end_time.isoformat(),
                },
            )

            return cleaned_files

        except Exception as e:
            logger.error(
                "Error during file cleanup",
                extra={
                    "plugin": self.name,
                    "recording_id": recording_id,
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "scan_duration": (
                        datetime.utcnow() - scan_start_time
                    ).total_seconds(),
                },
                exc_info=True,
            )
            return cleaned_files
