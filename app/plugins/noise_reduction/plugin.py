"""Noise reduction plugin implementation."""

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
from scipy import signal
from scipy.io import wavfile
from scipy.signal import butter, filtfilt

from app.models.database import DatabaseManager
from app.models.recording.events import (
    RecordingEndRequest,
    RecordingEvent,
    RecordingStartRequest,
)
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

        # Load configuration values
        config_dict = self.config.config or {}
        self._output_dir = Path(
            config_dict.get("output_directory", "data/cleaned_audio")
        )
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._thread_pool = ThreadPoolExecutor(
            max_workers=config_dict.get("max_concurrent_tasks", 4)
        )

        # High-pass filter settings
        self._highpass_cutoff = config_dict.get(
            "highpass_cutoff", 95
        )  # Higher cutoff for more aggressive filtering
        self._highpass_order = config_dict.get("highpass_order", 5)

        # STFT settings
        self._stft_window_size = config_dict.get("stft_window_size", 2048)
        self._stft_overlap_percent = config_dict.get("stft_overlap_percent", 75)

        # Noise reduction settings
        self._noise_reduce_factor = config_dict.get(
            "noise_reduce_factor", 1.0
        )  # Maximum reduction
        self._wiener_alpha = config_dict.get(
            "wiener_alpha", 2.5
        )  # Very strong Wiener filtering
        self._min_magnitude_threshold = config_dict.get(
            "spectral_floor", 0.04
        )  # Very low floor for maximum reduction

        # Noise profile settings
        self._noise_smooth_factor = config_dict.get(
            "smoothing_factor", 2
        )  # Keep moderate smoothing

        logger.info(
            "Noise reduction plugin initialized",
            extra={
                "plugin_name": self.name,
                "config": {
                    "output_directory": str(self._output_dir),
                    "max_concurrent_tasks": self._thread_pool._max_workers,
                    "highpass_cutoff": self._highpass_cutoff,
                    "highpass_order": self._highpass_order,
                    "stft_window_size": self._stft_window_size,
                    "stft_overlap_percent": self._stft_overlap_percent,
                    "noise_reduce_factor": self._noise_reduce_factor,
                    "wiener_alpha": self._wiener_alpha,
                    "spectral_floor": self._min_magnitude_threshold,
                    "smoothing_factor": self._noise_smooth_factor,
                },
            },
        )

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

        # Create processing state table
        await self.db.execute(
            """
            CREATE TABLE IF NOT EXISTS noise_reduction_processing (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                recording_id TEXT NOT NULL,
                status TEXT NOT NULL,
                source TEXT NOT NULL,
                input_file TEXT,
                output_file TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                completed_at TIMESTAMP,
                error TEXT,
                UNIQUE(recording_id, source)
            )
        """
        )

        await self._register_handlers()

    async def _shutdown(self) -> None:
        """Clean up plugin resources."""
        logger.info(
            "Shutting down noise reduction plugin",
            extra={"plugin_name": self.name},
        )
        self._thread_pool.shutdown(wait=True)

    async def _register_handlers(self) -> None:
        """Register event handlers."""
        logger.info("Registering noise reduction event handlers")
        await self.subscribe("recording.ended", self._handle_recording_ended)
        logger.info(
            "Registered noise reduction event handlers",
            extra={
                "event": "recording.ended",
                "plugin_name": self.name,
                "plugin_version": self.config.version,
            },
        )

    async def _handle_recording_ended(self, event: EventType) -> None:
        """Handle recording ended event."""
        logger.info(
            "Noise reduction plugin received recording ended event",
            extra={
                "plugin_name": self.name,
                "event_type": type(event).__name__,
                "event_dict": event.dict() if hasattr(event, "dict") else event,
            },
        )

        try:
            # Convert event to proper type if needed
            if isinstance(event, dict):
                # Skip completion events
                if event.get("name") == "noise_reduction.completed":
                    return
                try:
                    event = RecordingEvent(**event)
                except Exception as e:
                    logger.error(f"Failed to convert event: {e}")
                    return

            # Validate event type after potential conversion
            is_recording = isinstance(event, RecordingEvent)
            is_start = isinstance(event, RecordingStartRequest)
            is_end = isinstance(event, RecordingEndRequest)
            is_valid_event = is_recording or is_start or is_end

            if not is_valid_event:
                logger.error("Invalid event type")
                return

            # At this point event is a valid type, get recording ID
            recording_id = event.recording_id
            if not recording_id:
                logger.error("No recording ID in event")
                return

            # Process system audio if available
            system_audio_path = event.system_audio_path
            logger.info(
                "System audio path",
                extra={
                    "system_audio_path": system_audio_path,
                    "event_type": type(event).__name__,
                },
            )
            if system_audio_path:
                logger.info(
                    "Processing system audio", extra={"path": system_audio_path}
                )
                await self._process_audio(
                    recording_id,
                    system_audio_path,
                    "system",
                    event,
                )

            # Process microphone audio if available
            microphone_audio_path = event.microphone_audio_path
            logger.info(
                "Microphone audio path",
                extra={
                    "microphone_audio_path": microphone_audio_path,
                    "event_type": type(event).__name__,
                },
            )
            if microphone_audio_path:
                logger.info(
                    "Processing microphone audio", extra={"path": microphone_audio_path}
                )
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
                    "recording_id": recording_id
                    if "recording_id" in locals()
                    else None,
                },
                exc_info=True,
            )
            # Ensure recording_id is a string
            safe_recording_id = recording_id if isinstance(recording_id, str) else ""
            await self._emit_completion_event(
                recording_id=safe_recording_id,
                original_event=event,
                output_file="",  # Use empty string for error case
                status="error",
                metrics={"error": str(e)},
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
            logger.info(
                "Starting audio processing",
                extra={
                    "recording_id": recording_id,
                    "input_path": input_path,
                    "source": source,
                },
            )

            # Record processing start in database
            await self.db.execute(
                """
                INSERT INTO noise_reduction_processing
                (recording_id, status, source, input_file)
                VALUES (?, ?, ?, ?)
                """,
                (recording_id, "processing", source, input_path),
            )

            # Get system audio path for noise profile
            system_audio_path = (
                getattr(original_event, "system_audio_path", None)
                if isinstance(
                    original_event,
                    RecordingEvent | RecordingStartRequest | RecordingEndRequest,
                )
                else original_event.get("systemAudioPath")
            )

            if not system_audio_path:
                raise ValueError("No system audio available for noise profile")

            logger.info(
                "Processing audio with noise reduction",
                extra={
                    "recording_id": recording_id,
                    "input_path": input_path,
                    "system_audio_path": system_audio_path,
                    "source": source,
                },
            )

            # Process in thread pool
            output_path = self._output_dir / f"{recording_id}_{source}_cleaned.wav"
            logger.info(
                "Starting noise reduction worker",
                extra={
                    "recording_id": recording_id,
                    "input_path": os.path.join(os.getcwd(), input_path),
                    "system_audio_path": os.path.join(os.getcwd(), system_audio_path),
                    "output_path": str(output_path),
                },
            )
            future = self._thread_pool.submit(
                self._reduce_noise_worker,
                os.path.join(os.getcwd(), input_path),
                os.path.join(os.getcwd(), system_audio_path),
                str(output_path),
            )
            future.result()  # Wait for processing to complete

            # Update database with completion
            await self.db.execute(
                """
                UPDATE noise_reduction_processing
                SET status = ?, output_file = ?, completed_at = CURRENT_TIMESTAMP
                WHERE recording_id = ? AND source = ?
                """,
                ("completed", str(output_path), recording_id, source),
            )

            # Emit completion event
            await self._emit_completion_event(
                recording_id,
                original_event,
                str(output_path),
                "completed",
                {"output_file": str(output_path)},
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
            await self._emit_completion_event(
                recording_id=recording_id,
                original_event=original_event,
                output_file="",  # Use empty string for error case
                status="error",
                metrics={"error": str(e)},
            )

    def _reduce_noise_worker(
        self, mic_file: str, noise_file: str, output_file: str
    ) -> None:
        """Worker function for noise reduction processing."""
        # Read audio files
        mic_rate, mic_data = wavfile.read(mic_file)
        noise_rate, noise_data = wavfile.read(noise_file)

        # Ensure mono audio
        if len(mic_data.shape) > 1:
            mic_data = mic_data[:, 0]
        if len(noise_data.shape) > 1:
            noise_data = noise_data[:, 0]

        # Convert to float32 and normalize
        mic_data = mic_data.astype(np.float32)
        noise_data = noise_data.astype(np.float32)
        mic_data = mic_data / np.max(np.abs(mic_data))
        noise_data = noise_data / np.max(np.abs(noise_data))

        # Apply highpass filter
        mic_data = self._apply_highpass_filter(
            mic_data, self._highpass_cutoff, mic_rate, order=self._highpass_order
        )
        noise_data = self._apply_highpass_filter(
            noise_data, self._highpass_cutoff, noise_rate, order=self._highpass_order
        )

        # STFT parameters
        nperseg = self._stft_window_size
        noverlap = nperseg * self._stft_overlap_percent // 100

        # Compute noise profile
        noise_profile = self._compute_noise_profile(
            noise_data,
            noise_rate,
            nperseg,
            noverlap,
            smooth_factor=self._noise_smooth_factor,
        )

        # Process audio
        f, t, mic_spec = signal.stft(
            mic_data, fs=mic_rate, nperseg=nperseg, noverlap=noverlap
        )

        # Step 1: Spectral Subtraction
        mic_mag = np.abs(mic_spec)
        mic_phase = np.angle(mic_spec)

        # Subtract noise profile from magnitude spectrum
        subtracted_mag = np.maximum(
            mic_mag - noise_profile * self._noise_reduce_factor,
            mic_mag * self._min_magnitude_threshold,  # Spectral floor
        )

        # Reconstruct complex spectrogram
        subtracted_spec = subtracted_mag * np.exp(1j * mic_phase)

        # Step 2: Wiener Filtering
        # Estimate noise power from the noise profile
        noise_power = (noise_profile * self._noise_reduce_factor) ** 2

        # Apply Wiener filter to the spectral-subtracted signal
        cleaned_spec = self._wiener_filter(
            subtracted_spec, noise_power, alpha=self._wiener_alpha
        )

        # Additional magnitude adjustment
        cleaned_mag = np.abs(cleaned_spec)
        cleaned_mag = np.maximum(
            cleaned_mag, np.max(cleaned_mag) * self._min_magnitude_threshold
        )
        cleaned_spec = cleaned_mag * np.exp(1j * np.angle(cleaned_spec))

        # Inverse STFT
        _, cleaned_audio = signal.istft(
            cleaned_spec, fs=mic_rate, nperseg=nperseg, noverlap=noverlap
        )

        # Final cleanup and normalization
        cleaned_audio = np.clip(cleaned_audio, -1, 1)
        cleaned_audio = cleaned_audio / np.max(np.abs(cleaned_audio))
        cleaned_audio = (cleaned_audio * 32767).astype(np.int16)

        # Save cleaned audio
        wavfile.write(output_file, mic_rate, cleaned_audio)

    def _butter_highpass(self, cutoff: float, fs: float, order: int = 5) -> tuple:
        """Design a highpass filter."""
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype="high", analog=False)
        return b, a

    def _apply_highpass_filter(
        self, data: np.ndarray, cutoff: float, fs: float, order: int = 5
    ) -> np.ndarray:
        """Apply highpass filter to remove low frequency noise."""
        b, a = self._butter_highpass(cutoff, fs, order=order)
        return filtfilt(b, a, data)

    def _compute_noise_profile(
        self,
        noise_data: np.ndarray,
        fs: float,
        nperseg: int = 2048,
        noverlap: int = 1024,
        smooth_factor: int = 2,
    ) -> np.ndarray:
        """Compute and smooth the noise profile."""
        _, _, noise_spec = signal.stft(
            noise_data, fs=fs, nperseg=nperseg, noverlap=noverlap
        )
        noise_profile = np.mean(np.abs(noise_spec), axis=1)

        # Smooth the noise profile
        window_size = 2 * smooth_factor + 1
        noise_profile = np.convolve(
            noise_profile, np.ones(window_size) / window_size, mode="same"
        )

        return noise_profile.reshape(-1, 1)

    def _wiener_filter(
        self, spec: np.ndarray, noise_power: np.ndarray, alpha: float = 2.0
    ) -> np.ndarray:
        """Apply Wiener filter to reduce noise."""
        # Estimate signal power
        sig_power = np.abs(spec) ** 2

        # Calculate Wiener filter
        wiener_gain = np.maximum(1 - alpha * noise_power / (sig_power + 1e-10), 0.1)

        return spec * wiener_gain

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
            "original_event": getattr(original_event, "data", original_event),
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
            event="recording.ended",
            name="noise_reduction.completed",
            data=event_data,
            output_file=output_file,
            status=status,
        )
        await self.emit(event)
