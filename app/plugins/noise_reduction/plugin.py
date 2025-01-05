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
import threading

import numpy as np
from scipy import signal
from scipy.io import wavfile
from scipy.signal import butter, filtfilt

from app.models.database import get_db_async
from app.models.recording.events import RecordingEvent
from app.plugins.base import PluginBase, PluginConfig
from app.plugins.events.bus import EventBus as PluginEventBus
from app.plugins.events.models import EventContext
from app.utils.logging_config import get_logger

# Update logger to use the get_logger function
logger = get_logger("app.plugins.noise_reduction.plugin")

# Set the logger level to DEBUG to ensure we see all logs
logger.setLevel(logging.DEBUG)

EventData = dict[str, Any] | RecordingEvent


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
        
        # Initialize configuration values
        config_dict = config.config or {}
        self._output_directory = Path(config_dict.get("output_directory", "data/cleaned_audio"))
        self._noise_reduce_factor = float(config_dict.get("noise_reduce_factor", 1.0))
        self._wiener_alpha = float(config_dict.get("wiener_alpha", 2.5))
        self._highpass_cutoff = float(config_dict.get("highpass_cutoff", 95))
        self._spectral_floor = float(config_dict.get("spectral_floor", 0.04))
        self._smoothing_factor = int(config_dict.get("smoothing_factor", 2))
        self._max_workers = int(config_dict.get("max_concurrent_tasks", 4))
        
        # Initialize thread pool and other attributes
        self._executor = ThreadPoolExecutor(max_workers=self._max_workers)
        self._req_id = str(uuid.uuid4())
        self.db = None

    async def _initialize(self) -> None:
        """Initialize plugin"""
        if not self.event_bus:
            logger.warning(
                "No event bus available for plugin",
                extra={
                    "req_id": self._req_id,
                    "plugin_name": self.name,
                    "event_bus": None,
                    "plugin_enabled": self.config.enabled,
                    "plugin_version": self.config.version
                }
            )
            return

        try:
            logger.debug(
                "Starting noise reduction plugin initialization",
                extra={
                    "req_id": self._req_id,
                    "plugin_name": self.name,
                    "plugin_enabled": self.config.enabled,
                    "plugin_version": self.config.version,
                    "config": {
                        "output_directory": str(self._output_directory),
                        "noise_reduce_factor": self._noise_reduce_factor,
                        "wiener_alpha": self._wiener_alpha,
                        "highpass_cutoff": self._highpass_cutoff,
                        "spectral_floor": self._spectral_floor,
                        "smoothing_factor": self._smoothing_factor,
                        "max_concurrent_tasks": self._max_workers,
                    }
                }
            )

            # Subscribe to recording.ended event
            logger.debug(
                "Subscribing to recording.ended event",
                extra={
                    "req_id": self._req_id,
                    "plugin_name": self.name,
                    "event": "recording.ended",
                    "event_bus_type": type(self.event_bus).__name__,
                    "event_bus_id": id(self.event_bus)
                }
            )

            await self.event_bus.subscribe("recording.ended", self.handle_recording_ended)
            
            # Create output directory
            self._output_directory.mkdir(parents=True, exist_ok=True)
            
            logger.info(
                "Noise reduction plugin initialized successfully",
                extra={
                    "req_id": self._req_id,
                    "plugin_name": self.name,
                    "output_directory": str(self._output_directory),
                    "plugin_enabled": self.config.enabled,
                    "plugin_version": self.config.version,
                    "event_subscriptions": ["recording.ended"]
                }
            )

        except Exception as e:
            logger.error(
                "Failed to initialize noise reduction plugin",
                extra={
                    "req_id": self._req_id,
                    "plugin_name": self.name,
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "traceback": traceback.format_exc(),
                    "plugin_enabled": self.config.enabled,
                    "plugin_version": self.config.version
                }
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

    def _preprocess_audio(
        self,
        mic_data: np.ndarray,
        noise_data: np.ndarray,
        highpass_cutoff: float,
        mic_rate: int,
        noise_rate: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Process audio data for noise reduction.

        Converts audio to mono, normalizes values, and applies highpass filtering.
        """
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

        # Apply highpass filter if cutoff is specified
        if highpass_cutoff > 0:
            mic_data = self._apply_highpass_filter(
                mic_data, highpass_cutoff, mic_rate, order=3
            )
            noise_data = self._apply_highpass_filter(
                noise_data, highpass_cutoff, noise_rate, order=3
            )

        return mic_data, noise_data

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
                    "plugin_name": self.config.name,
                    "input_file": mic_file,
                    "output_file": output_file,
                },
            )

        except Exception as e:
            logger.error(
                "Error processing audio file",
                extra={
                    "plugin_name": self.config.name,
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

            # Preprocess audio data
            mic_data, noise_data = self._preprocess_audio(
                mic_data, noise_data, highpass_cutoff, mic_rate, noise_rate
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
                    "plugin_name": self.config.name,
                    "input_file": mic_file,
                    "output_file": output_file,
                    "output_size": os.path.getsize(output_file),
                },
            )
        except Exception as e:
            logger.error(
                "Error in reduce_noise",
                extra={
                    "plugin_name": self.config.name,
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

    def _format_dtype(self, dtype: np.dtype) -> str:
        """Convert numpy dtype to string representation."""
        return str(dtype)

    def _format_shape(self, shape: tuple[int, ...]) -> tuple[int, ...]:
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

    async def process_recording(self, recording_id: str, event_data: EventData) -> None:
        """Process a recording by applying noise reduction."""
        try:
            logger.info(
                "Starting audio processing",
                extra={
                    "req_id": self._req_id,
                    "plugin_name": self.name,
                    "recording_id": recording_id
                }
            )

            # Get audio paths from event data
            if isinstance(event_data, dict):
                system_audio_path = event_data.get("system_audio_path")
                microphone_audio_path = event_data.get("microphone_audio_path")
            else:
                system_audio_path = getattr(event_data, "system_audio_path", None)
                microphone_audio_path = getattr(event_data, "microphone_audio_path", None)

            # Process the audio files
            await self._process_audio_files(recording_id, system_audio_path, microphone_audio_path)

            # Emit completion event
            if self.event_bus:
                event = RecordingEvent(
                    recording_timestamp=datetime.utcnow().isoformat(),
                    recording_id=recording_id,
                    event="noise_reduction.completed",  
                    data={
                        "recording_id": recording_id,
                        "current_event": {
                            "noise_reduction": {
                                "status": "completed",
                                "timestamp": datetime.utcnow().isoformat(),
                                "output_paths": {
                                    "system": str(self._output_directory / f"{recording_id}_system_cleaned.wav"),
                                    "microphone": str(self._output_directory / f"{recording_id}_microphone_cleaned.wav")
                                }
                            }
                        },
                        "event_history": {
                            "recording": event_data
                        }
                    },
                    context=EventContext(
                        correlation_id=str(uuid.uuid4()),
                        source_plugin=self.name
                    )
                )
                await self.event_bus.publish(event)
                logger.info(
                    "Emitted noise reduction completed event",
                    extra={
                        "req_id": self._req_id,
                        "plugin_name": self.name,
                        "recording_id": recording_id
                    }
                )

        except Exception as e:
            logger.error(
                "Failed to process recording",
                extra={
                    "req_id": self._req_id,
                    "plugin_name": self.name,
                    "recording_id": recording_id,
                    "error": str(e)
                },
                exc_info=True
            )
            raise

    async def _process_audio(
        self,
        recording_id: str,
        input_path: str,
        output_path: str,
    ) -> None:
        """Process a single audio file with noise reduction.
        
        Args:
            recording_id: ID of the recording
            input_path: Path to input audio file
            output_path: Path to output audio file
        """
        try:
            logger.debug(
                "Reading audio file",
                extra={
                    "req_id": self._req_id,
                    "plugin_name": self.name,
                    "recording_id": recording_id,
                    "input_path": input_path
                }
            )

            # Read audio file
            sample_rate, audio_data = wavfile.read(input_path)

            # Apply noise reduction
            logger.debug(
                "Applying noise reduction",
                extra={
                    "req_id": self._req_id,
                    "plugin_name": self.name,
                    "recording_id": recording_id
                }
            )

            # Convert to float32 for processing
            audio_float = audio_data.astype(np.float32) / 32768.0

            # Apply highpass filter
            b, a = butter(4, self._highpass_cutoff / (sample_rate/2), 'high')
            filtered_audio = filtfilt(b, a, audio_float)

            # Convert back to int16
            logger.debug(
                "Converting to int16",
                extra={
                    "req_id": self._req_id,
                    "plugin_name": self.name,
                    "recording_id": recording_id
                }
            )
            output_audio = np.clip(filtered_audio * 32768.0, -32768, 32767).astype(np.int16)

            # Write output file
            logger.debug(
                f"Writing output file to {output_path}",
                extra={
                    "req_id": self._req_id,
                    "plugin_name": self.name,
                    "recording_id": recording_id
                }
            )
            wavfile.write(output_path, sample_rate, output_audio)

            # Log success
            logger.debug(
                "Successfully wrote file",
                extra={
                    "req_id": self._req_id,
                    "plugin_name": self.name,
                    "recording_id": recording_id,
                    "size": os.path.getsize(output_path)
                }
            )

        except Exception as e:
            logger.error(
                "Error processing audio file",
                extra={
                    "req_id": self._req_id,
                    "plugin_name": self.name,
                    "recording_id": recording_id,
                    "error": str(e),
                    "input_path": input_path
                },
                exc_info=True
            )
            raise

    async def handle_recording_ended(self, event: EventData) -> None:
        """Handle recording ended event."""
        event_id = str(uuid.uuid4())
        try:
            logger.info(
                "Received recording ended event",
                extra={
                    "req_id": event_id,
                    "plugin_name": self.name,
                    "plugin_enabled": self.config.enabled,
                    "plugin_version": self.config.version,
                    "recording_id": event.recording_id if hasattr(event, "recording_id") else event.get("recording_id"),
                    "event_id": event.event_id if hasattr(event, "event_id") else event.get("event_id"),
                    "event_type": type(event).__name__,
                    "event_data": str(event),
                    "event_bus_type": type(self.event_bus).__name__ if self.event_bus else None,
                    "event_bus_id": id(self.event_bus) if self.event_bus else None,
                    "handler_id": id(self),
                    "handler_method": "handle_recording_ended",
                    "thread_id": threading.get_ident()
                }
            )

            # Extract recording ID and paths
            if isinstance(event, dict):
                recording_id = event.get("recording_id")
                current_event = event.get("current_event", {})
                recording_data = current_event.get("recording", {})
                audio_paths = recording_data.get("audio_paths", {})
                mic_path = audio_paths.get("microphone")
                sys_path = audio_paths.get("system")
                logger.debug(
                    "Extracted paths from dict event",
                    extra={
                        "req_id": event_id,
                        "plugin_name": self.name,
                        "recording_id": recording_id,
                        "mic_path": mic_path,
                        "sys_path": sys_path,
                        "current_event": current_event,
                        "event_type": "dict",
                        "audio_paths_found": bool(audio_paths),
                        "event_data": str(event),
                        "handler_id": id(self)
                    }
                )
            else:
                recording_id = event.recording_id
                mic_path = event.microphone_audio_path
                sys_path = event.system_audio_path
                logger.debug(
                    "Extracted paths from object event",
                    extra={
                        "req_id": event_id,
                        "plugin_name": self.name,
                        "recording_id": recording_id,
                        "mic_path": mic_path,
                        "sys_path": sys_path,
                        "event_type": type(event).__name__,
                        "event_attrs": dir(event),
                        "event_data": str(event),
                        "handler_id": id(self)
                    }
                )

            if not recording_id:
                logger.error(
                    "No recording_id found in event data",
                    extra={
                        "req_id": event_id,
                        "plugin_name": self.name,
                        "event_data": str(event),
                        "event_type": type(event).__name__,
                        "event_dict": event if isinstance(event, dict) else event.__dict__,
                        "handler_id": id(self)
                    }
                )
                return

            # Check if this event originated from us to prevent loops
            source_plugin = (
                event.get("source_plugin") if isinstance(event, dict)
                else getattr(event.context, "source_plugin", None) if hasattr(event, "context")
                else None
            )
            
            logger.debug(
                "Checking event source",
                extra={
                    "req_id": event_id,
                    "plugin_name": self.name,
                    "recording_id": recording_id,
                    "source_plugin": source_plugin,
                    "our_plugin_name": self.name,
                    "is_our_event": source_plugin == self.name
                }
            )
            
            if source_plugin == self.name:
                logger.warning(
                    "Skipping event that originated from us",
                    extra={
                        "req_id": event_id,
                        "plugin_name": self.name,
                        "recording_id": recording_id,
                        "source_plugin": source_plugin,
                        "event_type": type(event).__name__
                    }
                )
                return

            # Check if we've already processed this recording
            if not self.db:
                logger.debug(
                    "Initializing database connection",
                    extra={
                        "req_id": event_id,
                        "plugin_name": self.name,
                        "recording_id": recording_id
                    }
                )
                self.db = await get_db_async()

            # Check both recording existence and processing status
            logger.debug(
                "Checking recording status in database",
                extra={
                    "req_id": event_id,
                    "plugin_name": self.name,
                    "recording_id": recording_id,
                    "query": "SELECT pt.status FROM recordings r LEFT JOIN plugin_tasks pt ON r.recording_id = pt.recording_id AND pt.plugin_name = ?"
                }
            )
            
            rows = await self.db.execute_fetchall(
                """
                SELECT pt.status 
                FROM recordings r
                LEFT JOIN plugin_tasks pt ON r.recording_id = pt.recording_id 
                    AND pt.plugin_name = ?
                WHERE r.recording_id = ?
                """,
                (self.name, recording_id)
            )

            logger.debug(
                "Database query results",
                extra={
                    "req_id": event_id,
                    "plugin_name": self.name,
                    "recording_id": recording_id,
                    "row_count": len(rows) if rows else 0,
                    "status": rows[0]['status'] if rows and rows[0]['status'] else None
                }
            )

            if not rows:
                logger.warning(
                    "Recording not found in database - skipping noise reduction",
                    extra={
                        "req_id": event_id,
                        "plugin_name": self.name,
                        "recording_id": recording_id
                    }
                )
                return

            # Check if already processed
            if rows[0]['status'] in ('completed', 'processing'):
                logger.warning(
                    "Recording already processed or being processed - skipping",
                    extra={
                        "req_id": event_id,
                        "plugin_name": self.name,
                        "recording_id": recording_id,
                        "status": rows[0]['status']
                    }
                )
                return

            # Mark as processing
            logger.info(
                "Marking recording for processing",
                extra={
                    "req_id": event_id,
                    "plugin_name": self.name,
                    "recording_id": recording_id,
                    "status": "processing"
                }
            )
            
            await self.db.execute(
                """
                INSERT INTO plugin_tasks 
                (id, plugin_name, recording_id, status, created_at, updated_at)
                VALUES (?, ?, ?, 'processing', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                """,
                (str(uuid.uuid4()), self.name, recording_id)
            )

            # Process the audio files
            logger.debug(
                "Starting audio processing",
                extra={
                    "req_id": event_id,
                    "plugin_name": self.name,
                    "recording_id": recording_id,
                    "mic_path": mic_path,
                    "sys_path": sys_path
                }
            )

            # Rest of the handler implementation...

        except Exception as e:
            logger.error(
                "Failed to handle recording ended event",
                extra={
                    "req_id": event_id,
                    "plugin_name": self.name,
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "event_data": str(event),
                    "traceback": traceback.format_exc()
                },
                exc_info=True
            )
            raise

    async def _shutdown(self) -> None:
        """Shutdown plugin."""
        if self.event_bus is not None:
            logger.info(
                "Unsubscribing from recording.ended event",
                extra={
                    "req_id": self._req_id,
                    "plugin_name": self.name
                }
            )
            await self.event_bus.unsubscribe(
                "recording.ended",
                self.handle_recording_ended
            )

        if self._executor is not None:
            logger.info(
                "Shutting down thread pool",
                extra={
                    "req_id": self._req_id,
                    "plugin_name": self.name
                }
            )
            self._executor.shutdown(wait=True)

    async def _process_audio_files(self, recording_id: str, system_audio_path: str | None, microphone_audio_path: str | None) -> None:
        """Process audio files with noise reduction.
        
        Args:
            recording_id: ID of the recording
            system_audio_path: Path to system audio file
            microphone_audio_path: Path to microphone audio file
        """
        logger.info(
            "Starting noise reduction",
            extra={
                "req_id": self._req_id,
                "plugin_name": self.name,
                "recording_id": recording_id
            }
        )

        if not microphone_audio_path and not system_audio_path:
            raise ValueError("No audio paths provided")

        # Process microphone audio if available
        if microphone_audio_path:
            mic_output = self._output_directory / f"{recording_id}_microphone_cleaned.wav"
            await self._process_audio(
                recording_id,
                microphone_audio_path,
                str(mic_output)
            )

        # Process system audio if available
        if system_audio_path:
            sys_output = self._output_directory / f"{recording_id}_system_cleaned.wav"
            await self._process_audio(
                recording_id,
                system_audio_path,
                str(sys_output)
            )

        logger.info(
            "Successfully processed audio file",
            extra={
                "req_id": self._req_id,
                "plugin_name": self.name,
                "recording_id": recording_id
            }
        )
