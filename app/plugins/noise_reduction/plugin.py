"""Noise reduction plugin using basic highpass filtering."""

import logging
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import NamedTuple

import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, filtfilt

from app.models.database import DatabaseManager
from app.models.recording.events import RecordingEvent
from app.plugins.base import PluginBase, PluginConfig
from app.plugins.events.bus import EventBus as PluginEventBus
from app.plugins.events.models import EventContext

logger = logging.getLogger(__name__)


class AudioPaths(NamedTuple):
    """Container for audio file paths."""

    recording_id: str
    system_audio: str | None
    mic_audio: str | None


class NoiseReductionPlugin(PluginBase):
    """Plugin for basic noise reduction using highpass filtering."""

    def __init__(
        self, config: PluginConfig, event_bus: PluginEventBus | None = None
    ) -> None:
        """Initialize plugin."""
        super().__init__(config, event_bus)
        self._plugin_name = "noise_reduction"
        self.db: DatabaseManager | None = None

        # Load configuration values
        config_dict = config.config or {}
        self._output_dir = Path(
            config_dict.get("output_directory", "data/cleaned_audio")
        )
        self._thread_pool = ThreadPoolExecutor(
            max_workers=config_dict.get("max_concurrent_tasks", 4)
        )

        # Load values from config
        self._highpass_cutoff = config_dict.get("highpass_cutoff", 60)
        self._highpass_order = config_dict.get("highpass_order", 3)

        # Create output directory
        self._output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Basic highpass filter noise reduction plugin initialized",
            extra={
                "plugin_name": self._plugin_name,
                "config": {
                    "output_directory": str(self._output_dir),
                    "max_concurrent_tasks": self._thread_pool._max_workers,
                    "highpass_cutoff": self._highpass_cutoff,
                    "highpass_order": self._highpass_order,
                },
            },
        )

    def _apply_highpass_filter(
        self, data: np.ndarray, cutoff_freq: float, sample_rate: int, order: int = 3
    ) -> np.ndarray:
        """Apply a Butterworth highpass filter to the audio data."""
        nyquist = sample_rate * 0.5
        normalized_cutoff = cutoff_freq / nyquist
        b, a = butter(order, normalized_cutoff, btype="high", analog=False)
        return filtfilt(b, a, data)

    def _reduce_noise_worker(
        self,
        mic_file: str,
        noise_file: str,
        output_file: str,
    ) -> None:
        """Worker function for basic highpass filtering."""
        import warnings

        from scipy.io.wavfile import WavFileWarning

        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=WavFileWarning)
                mic_rate, mic_data = wavfile.read(mic_file)

            # Ensure mono audio
            if len(mic_data.shape) > 1:
                mic_data = mic_data[:, 0]

            # Convert to float32 and normalize
            mic_data = mic_data.astype(np.float32)
            mic_data = mic_data / np.max(np.abs(mic_data))

            # Apply highpass filter
            filtered_data = self._apply_highpass_filter(
                mic_data,
                self._highpass_cutoff,
                mic_rate,
                order=self._highpass_order,
            )

            # Convert back to int16 for saving
            filtered_data = np.clip(filtered_data * 32767, -32768, 32767).astype(
                np.int16
            )
            wavfile.write(output_file, mic_rate, filtered_data)

            logger.info(
                "Successfully processed audio file",
                extra={
                    "plugin_name": self._plugin_name,
                    "input_file": mic_file,
                    "output_file": output_file,
                },
            )

        except Exception as e:
            logger.error(
                "Error processing audio file",
                extra={
                    "plugin_name": self._plugin_name,
                    "error": str(e),
                    "input_file": mic_file,
                },
            )
            raise

    async def process_recording(self, recording_id: str) -> None:
        """Process a recording with the noise reduction plugin."""
        if not self.db:
            raise RuntimeError("Database connection not initialized")

        try:
            # Get paths for the recording
            recording_rows = await self.db.execute_fetchall(
                "SELECT mic_path FROM recordings WHERE id = ?", (recording_id,)
            )
            if not recording_rows:
                raise ValueError(f"Recording {recording_id} not found")

            mic_path = recording_rows[0]["mic_path"]
            if not mic_path:
                raise ValueError("No microphone audio path found")

            # Generate output path
            output_filename = f"{uuid.uuid4()}.wav"
            output_path = str(self._output_dir / output_filename)

            # Process the audio
            self._reduce_noise_worker(mic_path, "", output_path)

            # Update database with processed file path
            await self.db.execute(
                "UPDATE recordings SET processed_path = ? WHERE id = ?",
                (output_path, recording_id),
            )
            await self.db.commit()

            # Emit success event
            if self.event_bus:
                event = RecordingEvent(
                    recording_timestamp=datetime.utcnow().isoformat(),
                    recording_id=recording_id,
                    event="Recording Ended",
                    name="recording.processed",
                    data={
                        "recording_id": recording_id,
                        "processed_path": output_path,
                        "plugin_name": self._plugin_name,
                    },
                    context=EventContext(correlation_id=str(uuid.uuid4())),
                )
                await self.event_bus.emit(event)

        except Exception as e:
            logger.error(
                "Failed to process recording",
                extra={
                    "plugin_name": self._plugin_name,
                    "recording_id": recording_id,
                    "error": str(e),
                },
            )
            raise
