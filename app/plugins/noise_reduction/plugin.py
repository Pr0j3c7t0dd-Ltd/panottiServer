"""Noise reduction plugin implementation."""

import os
from pathlib import Path
from typing import Any

import numpy as np
from scipy import signal
from scipy.io import wavfile

from app.models.database import DatabaseManager
from app.models.recording.events import RecordingEvent
from app.plugins.base import EventType, PluginBase, PluginConfig
from app.plugins.events.bus import EventBus
from app.utils.logging_config import get_logger

logger = get_logger(__name__)


class NoiseReductionPlugin(PluginBase):
    """Plugin for reducing noise in audio recordings."""

    def __init__(self, config: PluginConfig, event_bus: EventBus | None = None) -> None:
        """Initialize plugin."""
        super().__init__(config, event_bus)
        self.db = DatabaseManager.get_instance()
        self._output_dir = Path(os.getenv("OUTPUT_DIR", "data/processed"))
        self._output_dir.mkdir(parents=True, exist_ok=True)

    @property
    def name(self) -> str:
        """Get plugin name."""
        return "noise_reduction"

    async def _initialize(self) -> None:
        """Initialize the plugin.

        This method is called during plugin startup to perform any necessary
        initialization.
        """
        logger.info(
            "Initializing noise reduction plugin",
            extra={
                "plugin_name": self.name,
                "plugin_version": self.config.version,
                "output_dir": str(self._output_dir),
            },
        )
        await self._register_handlers()

    async def _shutdown(self) -> None:
        """Clean up plugin resources.

        This method is called during plugin shutdown to perform cleanup.
        """
        logger.info(
            "Shutting down noise reduction plugin",
            extra={"plugin_name": self.name},
        )

    async def _register_handlers(self) -> None:
        """Register event handlers."""
        await self.subscribe("Recording Ended", self._handle_recording_ended)

    async def _handle_recording_ended(self, event: EventType) -> None:
        """Handle recording ended event."""
        if not isinstance(event, RecordingEvent | dict):
            logger.error("Invalid event type")
            return

        recording_id = (
            event.recording_id
            if isinstance(event, RecordingEvent)
            else event.get("recording_id")
        )
        if not recording_id:
            logger.error("No recording ID in event")
            return

        try:
            # Process system audio if available
            system_audio_path = (
                event.system_audio_path
                if isinstance(event, RecordingEvent)
                else event.get("system_audio_path")
            )
            if system_audio_path:
                await self._process_audio(
                    recording_id,
                    system_audio_path,
                    "system",
                    event,
                )

            # Process microphone audio if available
            microphone_audio_path = (
                event.microphone_audio_path
                if isinstance(event, RecordingEvent)
                else event.get("microphone_audio_path")
            )
            if microphone_audio_path:
                await self._process_audio(
                    recording_id,
                    microphone_audio_path,
                    "microphone",
                    event,
                )

        except Exception as e:
            logger.error(
                "Error processing audio",
                extra={
                    "error": str(e),
                    "recording_id": recording_id,
                },
                exc_info=True,
            )
            await self._emit_completion_event(
                recording_id,
                event,
                None,
                "error",
                {"error": str(e)},
            )

    async def _process_audio(
        self,
        recording_id: str,
        input_path: str,
        source: str,
        original_event: EventType,
    ) -> None:
        """Process audio file to reduce noise."""
        try:
            # Read audio file
            sample_rate, audio_data = wavfile.read(input_path)
            if len(audio_data.shape) > 1:
                audio_data = audio_data.mean(axis=1)

            # Apply noise reduction
            processed_data = self._reduce_noise(audio_data)

            # Save processed audio
            output_path = self._output_dir / f"{recording_id}_{source}_processed.wav"
            wavfile.write(
                str(output_path), sample_rate, processed_data.astype(np.int16)
            )

            # Emit completion event
            await self._emit_completion_event(
                recording_id,
                original_event,
                str(output_path),
                "completed",
            )

        except Exception as e:
            logger.error(
                "Error processing audio",
                extra={
                    "error": str(e),
                    "recording_id": recording_id,
                    "source": source,
                },
                exc_info=True,
            )
            # Emit completion event with error status
            await self._emit_completion_event(
                recording_id,
                original_event,
                None,  # No output file on error
                "error",
                {"error": str(e)},
            )
            return  # Exit early on error

    def _reduce_noise(self, audio_data: np.ndarray) -> np.ndarray:
        """Apply noise reduction to audio data."""
        # Simple noise reduction using a low-pass filter
        b, a = signal.butter(4, 0.1, btype="low", analog=False)
        return signal.filtfilt(b, a, audio_data)

    async def _emit_completion_event(
        self,
        recording_id: str,
        original_event: EventType,
        output_file: str | None,
        status: str,
        metrics: dict[str, Any] | None = None,
    ) -> None:
        """Emit completion event."""
        event_data = {
            "recording_id": recording_id,
            "output_file": output_file,
            "status": status,
        }
        if metrics:
            event_data.update(metrics)

        # Get timestamps from original event
        recording_timestamp = ""
        if isinstance(original_event, RecordingEvent):
            recording_timestamp = original_event.recording_timestamp
        elif isinstance(original_event, dict):
            recording_timestamp = original_event.get("recording_timestamp", "")

        # Create and emit the event
        event = RecordingEvent(
            recording_timestamp=recording_timestamp,
            recording_id=recording_id,
            event="Recording Ended",
            name="noise_reduction.completed",
            data=event_data,
            output_file=output_file,
            status=status,
        )
        await self.emit(event)
