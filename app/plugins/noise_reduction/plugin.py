"""Noise reduction plugin using basic highpass filtering."""

import asyncio
import logging
import os
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from scipy import signal
from scipy.io import wavfile
from scipy.signal import butter, filtfilt

from app.models.database import DatabaseManager, get_db_async
from app.models.recording import RecordingEvent
from app.plugins.base import EventType, PluginBase, PluginConfig
from app.plugins.events.bus import EventBus as PluginEventBus
from app.plugins.events.models import EventContext

logger = logging.getLogger(__name__)


class AudioPaths:
    """Container for audio file paths."""

    def __init__(
        self, recording_id: str, system_audio: str | None, mic_audio: str | None
    ):
        self.recording_id = recording_id
        self.system_audio = system_audio
        self.mic_audio = mic_audio


class NoiseReductionPlugin(PluginBase):
    """Plugin for basic noise reduction using highpass filtering."""

    def __init__(
        self, config: PluginConfig, event_bus: PluginEventBus | None = None
    ) -> None:
        """Initialize plugin."""
        super().__init__(config, event_bus)
        self._plugin_name = "noise_reduction"
        self.db: DatabaseManager | None = None

        # Initialize thread pool with configured max workers
        self._executor = ThreadPoolExecutor(max_workers=4)

        logger.info(
            "NoiseReductionPlugin initialized",
            extra={
                "plugin": self.name,
                "config": {
                    "output_directory": "data/cleaned_audio",
                    "noise_reduce_factor": 1.0,
                    "wiener_alpha": 2.5,
                    "highpass_cutoff": 95,
                    "spectral_floor": 0.04,
                    "smoothing_factor": 2,
                    "max_concurrent_tasks": 4,
                },
            },
        )

    async def _initialize(self) -> None:
        """Initialize plugin."""
        try:
            # Initialize database connection
            self.db = await get_db_async()

            # Load configuration values
            config_dict = self.config.config or {}

            # Convert configuration values to proper types
            self._output_directory = Path(
                config_dict.get("output_directory", "data/cleaned_audio")
            )
            self._noise_reduce_factor = float(
                config_dict.get("noise_reduce_factor", 1.0)
            )
            self._wiener_alpha = float(config_dict.get("wiener_alpha", 2.5))
            self._highpass_cutoff = float(config_dict.get("highpass_cutoff", 95))
            self._spectral_floor = float(config_dict.get("spectral_floor", 0.04))
            self._smoothing_factor = int(config_dict.get("smoothing_factor", 2))
            self._max_concurrent_tasks = int(config_dict.get("max_concurrent_tasks", 4))

            # Initialize thread pool with configured max workers
            self._executor = ThreadPoolExecutor(max_workers=self._max_concurrent_tasks)

            # Create output directory
            self._output_directory.mkdir(parents=True, exist_ok=True)

            # Subscribe to events
            if self.event_bus is not None:
                logger.info(
                    "Subscribing to events",
                    extra={"plugin": self.name, "events": ["recording.ended"]},
                )
                await self.event_bus.subscribe(
                    "recording.ended", self.handle_recording_ended
                )

            logger.info(
                "NoiseReductionPlugin initialized",
                extra={
                    "plugin": self.name,
                    "config": {
                        "output_directory": str(self._output_directory),
                        "noise_reduce_factor": self._noise_reduce_factor,
                        "wiener_alpha": self._wiener_alpha,
                        "highpass_cutoff": self._highpass_cutoff,
                        "spectral_floor": self._spectral_floor,
                        "smoothing_factor": self._smoothing_factor,
                        "max_concurrent_tasks": self._max_concurrent_tasks,
                    },
                },
            )
        except Exception as e:
            logger.error(
                "Failed to initialize plugin",
                extra={"plugin": self.name, "error": str(e)},
                exc_info=True,
            )
            raise

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
                order=3,
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

    def reduce_noise(
        self,
        mic_file: str,
        noise_file: str,
        output_file: str,
        noise_reduce_factor: float = 1.0,
        wiener_alpha: float = 2.5,
        highpass_cutoff: float = 95,
        spectral_floor: float = 0.04,
        smoothing_factor: int = 2,
    ) -> None:
        """
        Enhanced noise reduction using spectral subtraction and Wiener filtering.

        Args:
            mic_file: Path to microphone recording WAV file
            noise_file: Path to system recording WAV file (noise profile)
            output_file: Path to save cleaned audio
            noise_reduce_factor: Amount of noise reduction
                (0 to 1, default 1.0 for maximum)
            wiener_alpha: Wiener filter strength (default 2.5 for aggressive filtering)
            highpass_cutoff: Highpass filter cutoff in Hz (default 95Hz)
            spectral_floor: Minimum spectral magnitude (default 0.04)
            smoothing_factor: Noise profile smoothing (default 2)
        """
        import warnings

        from scipy.io.wavfile import WavFileWarning

        try:
            # Validate input files exist
            if not os.path.exists(mic_file):
                raise FileNotFoundError(f"Microphone file not found: {mic_file}")
            if not os.path.exists(noise_file):
                raise FileNotFoundError(f"System audio file not found: {noise_file}")

            # Create output directory
            output_dir = os.path.dirname(output_file)
            os.makedirs(output_dir, exist_ok=True)

            # Verify output directory is writable
            if not os.access(output_dir, os.W_OK):
                raise PermissionError(f"Output directory not writable: {output_dir}")

            logger.info(
                "Starting noise reduction",
                extra={
                    "plugin": self.name,
                    "mic_file": mic_file,
                    "noise_file": noise_file,
                    "output_file": output_file,
                    "mic_size": os.path.getsize(mic_file),
                    "noise_size": os.path.getsize(noise_file),
                },
            )

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=WavFileWarning)
                logger.debug("Reading mic file")
                mic_rate, mic_data = wavfile.read(mic_file)
                logger.debug("Reading noise file")
                noise_rate, noise_data = wavfile.read(noise_file)

            logger.debug(
                "Audio file details",
                extra={
                    "mic_rate": int(mic_rate),
                    "mic_shape": self._format_shape(mic_data.shape),
                    "mic_dtype": self._format_dtype(mic_data.dtype),
                    "noise_rate": int(noise_rate),
                    "noise_shape": self._format_shape(noise_data.shape),
                    "noise_dtype": self._format_dtype(noise_data.dtype),
                },
            )

            # Ensure mono audio
            if len(mic_data.shape) > 1:
                mic_data = mic_data[:, 0]
            if len(noise_data.shape) > 1:
                noise_data = noise_data[:, 0]

            # Convert to float32 and normalize
            mic_data = mic_data.astype(np.float32)
            mic_data = mic_data / np.max(np.abs(mic_data))
            noise_data = noise_data.astype(np.float32)
            noise_data = noise_data / np.max(np.abs(noise_data))

            logger.debug("Applying highpass filter")
            # Apply highpass filter if cutoff is specified
            if highpass_cutoff > 0:
                mic_data = self._apply_highpass_filter(
                    mic_data, highpass_cutoff, mic_rate, order=3
                )
                noise_data = self._apply_highpass_filter(
                    noise_data, highpass_cutoff, noise_rate, order=3
                )

            # Simple STFT parameters - adjusted for better frequency resolution
            nperseg = 4096  # Increased for better frequency resolution
            noverlap = 3072  # 75% overlap for better time resolution

            logger.debug("Computing STFT")
            # Compute STFT of microphone signal
            _, _, mic_spec = signal.stft(
                mic_data,
                fs=mic_rate,
                nperseg=nperseg,
                noverlap=noverlap,
                window="hann",
            )

            logger.debug("Computing noise profile")
            # Compute noise profile with specified smoothing
            noise_profile = self.compute_noise_profile(
                noise_data,
                noise_rate,
                nperseg=nperseg,
                noverlap=noverlap,
                smooth_factor=smoothing_factor,
            )

            logger.debug("Applying noise reduction")
            # Enhanced spectral subtraction with Wiener filtering
            if wiener_alpha > 0:
                # Apply Wiener filter with specified alpha
                cleaned_spec = self.wiener_filter(
                    mic_spec, noise_profile[:, np.newaxis] ** 2, alpha=wiener_alpha
                )
            else:
                # Simple spectral subtraction
                mic_mag = np.abs(mic_spec)
                reduction = noise_profile[:, np.newaxis] * noise_reduce_factor
                cleaned_mag = np.maximum(mic_mag - reduction, mic_mag * spectral_floor)
                cleaned_spec = cleaned_mag * np.exp(1j * np.angle(mic_spec))

            logger.debug("Computing inverse STFT")
            # Inverse STFT
            _, cleaned_data = signal.istft(
                cleaned_spec,
                fs=mic_rate,
                nperseg=nperseg,
                noverlap=noverlap,
                window="hann",
            )

            logger.debug("Converting to int16")
            # Convert back to int16 for saving
            cleaned_data = np.clip(cleaned_data * 32767, -32768, 32767).astype(np.int16)

            logger.debug(f"Writing output file to {output_file}")
            try:
                wavfile.write(output_file, mic_rate, cleaned_data)
                logger.debug(
                    f"Successfully wrote file, size: {os.path.getsize(output_file)}"
                )
            except Exception as write_error:
                logger.error(
                    "Failed to write output file",
                    extra={
                        "error": str(write_error),
                        "output_file": output_file,
                        "output_dir": output_dir,
                        "output_dir_exists": os.path.exists(output_dir),
                        "output_dir_writable": os.access(output_dir, os.W_OK),
                    },
                )
                raise

            logger.info(
                "Successfully processed audio file",
                extra={
                    "plugin_name": self._plugin_name,
                    "input_file": mic_file,
                    "output_file": output_file,
                    "output_size": os.path.getsize(output_file),
                },
            )
        except Exception as e:
            logger.error(
                "Error in reduce_noise",
                extra={
                    "plugin_name": self._plugin_name,
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "traceback": traceback.format_exc(),
                    "mic_file": mic_file,
                    "noise_file": noise_file,
                    "output_file": output_file,
                    "cwd": os.getcwd(),
                },
            )
            raise

    def _format_dtype(self, dtype) -> str:
        """Convert numpy dtype to string representation."""
        return str(dtype)

    def _format_shape(self, shape) -> tuple:
        """Convert numpy shape to regular tuple."""
        return tuple(int(x) for x in shape)

    def compute_noise_profile(
        self,
        noise_data: np.ndarray,
        sample_rate: int,
        nperseg: int = 4096,
        noverlap: int = 3072,
        smooth_factor: int = 2,
    ) -> np.ndarray:
        """
        Compute noise profile from noise data using STFT.

        Args:
            noise_data: Noise audio data
            sample_rate: Sample rate of the audio
            nperseg: Number of samples per FFT segment
            noverlap: Number of samples to overlap between segments
            smooth_factor: Factor for smoothing the noise profile

        Returns:
            Smoothed noise profile magnitude spectrum
        """
        # Compute STFT of noise
        _, _, noise_spec = signal.stft(
            noise_data,
            fs=sample_rate,
            nperseg=nperseg,
            noverlap=noverlap,
            window="hann",
        )

        # Compute magnitude spectrum
        noise_mag = np.abs(noise_spec)

        # Average across time
        noise_profile = np.mean(noise_mag, axis=1)

        # Apply smoothing if requested
        if smooth_factor > 1:
            kernel = np.ones(smooth_factor) / smooth_factor
            noise_profile = np.convolve(noise_profile, kernel, mode="same")

        return noise_profile

    def wiener_filter(
        self, signal_spec: np.ndarray, noise_power: np.ndarray, alpha: float = 2.0
    ) -> np.ndarray:
        """
        Apply Wiener filter to the signal spectrum.

        Args:
            signal_spec: Complex spectrum of the signal
            noise_power: Power spectrum of the noise
            alpha: Wiener filter parameter (larger values = more aggressive filtering)

        Returns:
            Filtered signal spectrum
        """
        # Compute signal power
        signal_power = np.abs(signal_spec) ** 2

        # Compute Wiener filter
        wiener_gain = np.maximum(1 - (alpha * noise_power) / (signal_power + 1e-10), 0)

        # Apply filter
        return signal_spec * wiener_gain

    async def _handle_recording_ended(self, event: EventType) -> None:
        """Handle recording ended event."""
        recording_id = (
            event.recording_id
            if isinstance(event, RecordingEvent)
            else event["recording_id"]
            if isinstance(event, dict)
            else None
        )
        if not recording_id:
            logger.error("No recording ID in event")
            return

        try:
            await self.process_recording(recording_id)
        except Exception as e:
            logger.error(
                "Failed to process recording",
                extra={
                    "plugin_name": self._plugin_name,
                    "recording_id": recording_id,
                    "error": str(e),
                },
            )

    async def process_recording(self, recording_id: str) -> None:
        """Process a recording with the noise reduction plugin."""
        if not self.db:
            raise RuntimeError("Database connection not initialized")

        try:
            # Get paths for the recording
            recording_rows = await self.db.execute_fetchall(
                """
                SELECT microphone_audio_path, system_audio_path
                FROM recordings
                WHERE recording_id = ?
                """,
                (recording_id,),
            )
            if not recording_rows:
                raise ValueError(f"Recording {recording_id} not found")

            mic_path = recording_rows[0]["microphone_audio_path"]
            sys_path = recording_rows[0]["system_audio_path"]

            if not mic_path:
                raise ValueError("No microphone audio path found")
            if not sys_path:
                raise ValueError("No system audio path found")

            # Generate output path (base directory only, actual filename set in _process_audio)
            base_dir = self._output_directory
            output_base = base_dir / recording_id

            # Process the audio
            await self._process_audio(
                recording_id, mic_path, sys_path, str(output_base)
            )

            # Get the actual output filename that was created
            output_path = self._output_directory / f"{Path(mic_path).stem}_cleaned.wav"

            # Update database with processed file path
            await self.db.execute(
                """
                UPDATE recordings
                SET processed_audio_path = ?
                WHERE recording_id = ?
                """,
                (str(output_path), recording_id),
            )
            await self.db.commit()

            # Emit noise reduction completed event
            if self.event_bus:
                event = RecordingEvent(
                    recording_timestamp=datetime.utcnow().isoformat(),
                    recording_id=recording_id,
                    event="noise_reduction.completed",
                    name="noise_reduction.completed",
                    data={
                        "recording_id": recording_id,
                        "noise_reduced_microphone_path": str(output_path),
                        "output_file": str(
                            output_path
                        ),  # Required by audio_transcription plugin
                        "status": "completed",
                        "processing_details": {
                            "plugin": self._plugin_name,
                            "noise_reduction_settings": {
                                "noise_reduce_factor": self._noise_reduce_factor,
                                "wiener_alpha": self._wiener_alpha,
                                "highpass_cutoff": self._highpass_cutoff,
                                "spectral_floor": self._spectral_floor,
                                "smoothing_factor": self._smoothing_factor,
                            },
                        },
                        "original_event": {
                            "systemAudioPath": sys_path,
                            "microphoneAudioPath": mic_path,
                            "metadata": {
                                "recordingId": recording_id,
                                "processedPath": str(output_path),
                                "processingTimestamp": datetime.utcnow().isoformat(),
                            },
                        },
                        "audio_files": {
                            "original": {"system": sys_path, "microphone": mic_path},
                            "processed": {"noise_reduced_microphone": str(output_path)},
                        },
                    },
                    context=EventContext(correlation_id=str(uuid.uuid4())),
                )
                await self.event_bus.emit(event)
                logger.info(
                    "Emitted noise reduction completed event",
                    extra={
                        "plugin": self.name,
                        "recording_id": recording_id,
                        "noise_reduced_microphone_path": str(output_path),
                    },
                )

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

    async def _process_audio(
        self, recording_id: str, mic_path: str, sys_path: str, output_base: str
    ) -> None:
        """Process audio files in a separate thread"""
        try:
            # Create output directory if it doesn't exist
            output_dir = Path(output_base).parent
            output_dir.mkdir(parents=True, exist_ok=True)

            # Extract source filename
            source_name = Path(mic_path).stem  # Gets filename without extension
            output_filename = f"{source_name}_cleaned.wav"
            output_path = output_dir / output_filename

            # Log paths before processing
            logger.info(
                "Starting audio processing",
                extra={
                    "plugin": self.name,
                    "mic_path_exists": os.path.exists(mic_path),
                    "sys_path_exists": os.path.exists(sys_path),
                    "output_dir_exists": output_dir.exists(),
                    "output_dir_writable": os.access(output_dir, os.W_OK),
                    "mic_path": mic_path,
                    "sys_path": sys_path,
                    "output_file": str(output_path),
                },
            )

            # Use configuration values from plugin.yaml
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                self._executor,
                self.reduce_noise,
                str(mic_path),
                str(sys_path),
                str(output_path),
                self._noise_reduce_factor,
                self._wiener_alpha,
                self._highpass_cutoff,
                self._spectral_floor,
                self._smoothing_factor,
            )
        except Exception as e:
            logger.error(
                "Error processing audio file",
                extra={
                    "plugin_name": self._plugin_name,
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "mic_path": str(mic_path),
                    "sys_path": str(sys_path),
                    "output_file": str(output_base),
                    "traceback": traceback.format_exc(),
                },
            )
            raise

    async def handle_recording_ended(self, event: dict[str, Any]) -> None:
        """Handle recording_ended events"""
        try:
            logger.debug(
                "Raw event data",
                extra={
                    "plugin": self.name,
                    "raw_event": str(event),
                    "event_type": str(type(event)),
                    "event_keys": str(
                        event.keys() if isinstance(event, dict) else dir(event)
                    ),
                },
            )

            # Handle both event types
            if isinstance(event, dict):
                if "payload" in event:
                    event = event["payload"]
            else:
                event = event.__dict__

            # Extract recording information from event
            recording_id = event.get("recording_id")
            mic_path = event.get("microphone_audio_path")
            sys_path = event.get("system_audio_path")

            logger.info(
                "Handling recording ended event",
                extra={
                    "plugin": self.name,
                    "event_data": {
                        "recording_id": recording_id,
                        "mic_path": mic_path,
                        "sys_path": sys_path,
                    },
                },
            )

            if not recording_id:
                raise ValueError("No recording ID in event")
            if not mic_path:
                raise ValueError("No microphone path in event")
            if not sys_path:
                raise ValueError("No system audio path in event")

            # Process the recording
            await self.process_recording(recording_id)

        except Exception as e:
            logger.error(
                "Failed to handle recording ended event",
                extra={"plugin": self.name, "error": str(e), "event": str(event)},
            )
            raise

    async def _shutdown(self) -> None:
        """Shutdown plugin."""
        if self.event_bus is not None:
            logger.info("Unsubscribing from recording.ended event")
            await self.event_bus.unsubscribe(
                "recording.ended", self.handle_recording_ended
            )

        if self._executor is not None:
            logger.info("Shutting down thread pool")
            self._executor.shutdown(wait=True)
