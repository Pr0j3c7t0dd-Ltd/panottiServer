"""Example user plugin demonstrating the plugin structure."""
import logging
import pluggy
from fastapi import FastAPI, Request, Response

logger = logging.getLogger(__name__)
hookimpl = pluggy.HookimplMarker("panotti")

class ExampleUserPlugin:
    """Example user plugin that demonstrates hook implementations."""

    @hookimpl
    async def on_startup(self, app: FastAPI) -> None:
        """Called when the application starts."""
        logger.info("ExampleUserPlugin: Application starting")

    @hookimpl
    async def before_recording_start(self, recording_id: str) -> None:
        """Called before a recording starts."""
        logger.info(f"ExampleUserPlugin: About to start recording {recording_id}")
        # Add your custom logic here, e.g.:
        # - Prepare storage
        # - Initialize recording-specific resources
        # - Send notifications

    @hookimpl
    async def after_recording_end(self, recording_id: str) -> None:
        """Called after a recording ends."""
        logger.info(f"ExampleUserPlugin: Finished recording {recording_id}")
        # Add your custom logic here, e.g.:
        # - Process the recording
        # - Clean up resources
        # - Send notifications
