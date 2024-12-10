"""Hook specifications for the PanottiServer plugin system."""
import pluggy
from fastapi import FastAPI, Request, Response

hookspec = pluggy.HookspecMarker("panotti")

class PanottiSpecs:
    """Hook specifications for PanottiServer plugins."""

    @hookspec
    async def on_startup(self, app: FastAPI) -> None:
        """Run startup logic when the FastAPI application starts.
        
        Args:
            app: The FastAPI application instance
        """

    @hookspec
    async def on_shutdown(self, app: FastAPI) -> None:
        """Run cleanup logic when the FastAPI application shuts down.
        
        Args:
            app: The FastAPI application instance
        """

    @hookspec
    async def before_request(self, request: Request) -> None:
        """Run logic before a request is processed.
        
        Args:
            request: The FastAPI request object
        """

    @hookspec
    async def after_request(self, response: Response) -> None:
        """Run logic after a request is processed.
        
        Args:
            response: The FastAPI response object
        """

    @hookspec
    async def before_recording_start(self, recording_id: str) -> None:
        """Run logic before a recording starts.
        
        Args:
            recording_id: The ID of the recording that's about to start
        """

    @hookspec
    async def after_recording_end(self, recording_id: str) -> None:
        """Run logic after a recording ends.
        
        Args:
            recording_id: The ID of the recording that has ended
        """
