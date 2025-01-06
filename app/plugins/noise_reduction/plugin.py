"""Noise reduction plugin using basic highpass filtering."""

import asyncio
import logging
import os
import traceback
import uuid
import warnings
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any
import threading

import numpy as np
from scipy import signal
from scipy.io import wavfile
from scipy.signal import butter, filtfilt

from app.models.database import get_db_async, DatabaseManager
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
        self._db: DatabaseManager | None = None

    async def _initialize(self) -> None:
        """Initialize the plugin."""
        try:
            logger.info(
                "Initializing noise reduction plugin",
                extra={
                    "req_id": self._req_id,
                    "plugin_name": self.name
                }
            )

            # Initialize database connection
            for attempt in range(3):  # Try 3 times
                try:
                    db_manager = await get_db_async()
                    self._db = await db_manager
                    break
                except Exception as e:
                    if attempt == 2:  # Last attempt
                        logger.error(
                            "Failed to initialize database connection",
                            extra={
                                "req_id": self._req_id,
                                "plugin_name": self.name,
                                "error": str(e)
                            },
                            exc_info=True
                        )
                        raise
                    await asyncio.sleep(1)  # Wait before retrying

            # Create output directory if it doesn't exist
            os.makedirs(self._output_directory, exist_ok=True)

            # Subscribe to recording.ended event
            if self.event_bus:
                await self.event_bus.subscribe("recording.ended", self.handle_recording_ended)
                logger.info(
                    "Subscribed to recording.ended event",
                    extra={
                        "req_id": self._req_id,
                        "plugin_name": self.name
                    }
                )

            logger.info(
                "Noise reduction plugin initialized successfully",
                extra={
                    "req_id": self._req_id,
                    "plugin_name": self.name,
                    "output_directory": str(self._output_directory)
                }
            )

            self._initialized = True

        except Exception as e:
            logger.error(
                "Failed to initialize noise reduction plugin",
                extra={
                    "req_id": self._req_id,
                    "plugin_name": self.name,
                    "error": str(e)
                },
                exc_info=True
            )
            raise

    def _apply_highpass_filter(
        self, data: np.ndarray, cutoff_freq: float, sample_rate: int, order: int = 2
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
        noise_reduce_factor: float = 0.3,
        wiener_alpha: float = 1.8,
        highpass_cutoff: float = 80,
        spectral_floor: float = 0.15,
        smoothing_factor: int = 2
    ) -> None:
        """
        Enhanced noise reduction using spectral subtraction and Wiener filtering.
        
        Args:
            mic_file: Path to microphone recording WAV file
            noise_file: Path to system recording WAV file (noise profile)
            output_file: Path to save cleaned audio
            noise_reduce_factor: Amount of noise reduction (0 to 1)
            wiener_alpha: Wiener filter strength
            highpass_cutoff: Highpass filter cutoff in Hz
            spectral_floor: Minimum spectral magnitude
            smoothing_factor: Noise profile smoothing
        """
        logger.info(
            "Starting enhanced noise reduction",
            extra={
                "plugin": self.name,
                "mic_file": mic_file,
                "noise_file": noise_file,
                "output_file": output_file,
                "settings": {
                    "noise_reduce_factor": noise_reduce_factor,
                    "wiener_alpha": wiener_alpha,
                    "highpass_cutoff": highpass_cutoff,
                    "spectral_floor": spectral_floor,
                    "smoothing_factor": smoothing_factor
                }
            }
        )
        
        # Validate input files
        if not os.path.exists(mic_file):
            raise FileNotFoundError(f"Microphone audio file not found: {mic_file}")
        if not os.path.exists(noise_file):
            raise FileNotFoundError(f"System audio file not found: {noise_file}")
            
        # Create output directory
        output_dir = os.path.dirname(os.path.abspath(output_file))
        os.makedirs(output_dir, exist_ok=True)
        
        # Read audio files
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", wavfile.WavFileWarning)
            mic_rate, mic_data = wavfile.read(mic_file)
            noise_rate, noise_data = wavfile.read(noise_file)
        
        # Convert to mono if stereo
        if len(mic_data.shape) > 1:
            mic_data = np.mean(mic_data, axis=1)
        if len(noise_data.shape) > 1:
            noise_data = np.mean(noise_data, axis=1)
            
        # Convert to float32 and normalize
        mic_data = mic_data.astype(np.float32)
        noise_data = noise_data.astype(np.float32)
        
        mic_max = np.max(np.abs(mic_data))
        noise_max = np.max(np.abs(noise_data))
        
        if mic_max == 0 or noise_max == 0:
            raise ValueError("Input audio is silent")
            
        mic_data = mic_data / mic_max
        noise_data = noise_data / noise_max
        
        # Apply highpass filter if configured
        if highpass_cutoff > 0:
            b, a = butter(5, highpass_cutoff / (mic_rate/2), btype='high')
            mic_data = filtfilt(b, a, mic_data)
            noise_data = filtfilt(b, a, noise_data)
        
        # STFT parameters optimized for speech
        nperseg = 2048  # Longer window for better frequency resolution
        noverlap = nperseg // 2  # 50% overlap
        
        # Compute STFTs
        _, _, mic_spec = signal.stft(mic_data, fs=mic_rate, nperseg=nperseg, noverlap=noverlap)
        
        # Compute noise profile with speech emphasis
        noise_profile = self.compute_noise_profile(
            noise_data,
            fs=noise_rate,
            nperseg=nperseg,
            noverlap=noverlap,
            smooth_factor=smoothing_factor
        )
        
        # Apply Wiener filter with speech preservation
        if wiener_alpha > 0:
            cleaned_spec = self.wiener_filter(
                mic_spec,
                noise_profile * noise_reduce_factor,
                alpha=wiener_alpha
            )
        else:
            # Simple spectral subtraction with floor
            mic_mag = np.abs(mic_spec)
            reduction = noise_profile * noise_reduce_factor
            cleaned_mag = np.maximum(mic_mag - reduction, mic_mag * spectral_floor)
            cleaned_spec = cleaned_mag * np.exp(1j * np.angle(mic_spec))
        
        # Inverse STFT
        _, cleaned_audio = signal.istft(cleaned_spec, fs=mic_rate, nperseg=nperseg, noverlap=noverlap)
        
        # Normalize output
        cleaned_max = np.max(np.abs(cleaned_audio))
        if cleaned_max > 0:
            cleaned_audio = cleaned_audio / cleaned_max
            
        # Convert to int16
        cleaned_audio = np.clip(cleaned_audio * 32767, -32768, 32767).astype(np.int16)
        
        # Save output
        try:
            wavfile.write(output_file, mic_rate, cleaned_audio)
            
            if not os.path.exists(output_file):
                raise IOError("Failed to write output file")
                
            logger.info(
                "Successfully saved cleaned audio",
                extra={
                    "plugin": self.name,
                    "output_file": output_file,
                    "output_size": os.path.getsize(output_file)
                }
            )
        except Exception as e:
            raise IOError(f"Failed to save cleaned audio: {str(e)}")

    def _format_dtype(self, dtype: np.dtype) -> str:
        """Convert numpy dtype to string representation."""
        return str(dtype)

    def _format_shape(self, shape: tuple[int, ...]) -> tuple[int, ...]:
        """Convert numpy shape to regular tuple."""
        return tuple(int(x) for x in shape)

    def compute_noise_profile(
        self,
        noise_data: np.ndarray,
        fs: float,
        nperseg: int = 2048,
        noverlap: int = 1024,
        smooth_factor: int = 2
    ) -> np.ndarray:
        """Compute and smooth the noise profile with emphasis on speech frequencies."""
        # Compute STFT
        f, t, noise_spec = signal.stft(noise_data, fs=fs, nperseg=nperseg, noverlap=noverlap)
        
        # Get frequency axis
        freqs = f
        
        # Compute average magnitude spectrum
        noise_profile = np.mean(np.abs(noise_spec), axis=1)
        
        # Apply frequency-dependent weighting
        # Emphasize frequencies in speech range (300-3000 Hz)
        speech_mask = np.ones_like(freqs)
        speech_range = (freqs >= 300) & (freqs <= 3000)
        speech_mask[speech_range] = 1.2  # Boost speech frequencies
        
        noise_profile = noise_profile * speech_mask
        
        if smooth_factor > 0:
            # Smooth the profile
            window_size = 2 * smooth_factor + 1
            noise_profile = np.convolve(noise_profile, np.ones(window_size)/window_size, mode='same')
        
        return noise_profile.reshape(-1, 1)

    def wiener_filter(
        self,
        spec: np.ndarray,
        noise_power: np.ndarray,
        alpha: float = 1.8
    ) -> np.ndarray:
        """Apply Wiener filter with speech-focused processing."""
        # Compute signal power
        sig_power = np.abs(spec)**2
        
        # Compute SNR-dependent Wiener filter
        snr = sig_power / (noise_power + 1e-10)
        wiener_gain = np.maximum(1 - alpha / (snr + 1), 0.1)
        
        # Apply additional weighting for speech preservation
        # This helps preserve speech transients
        power_ratio = sig_power / (np.max(sig_power) + 1e-10)
        speech_weight = np.minimum(1.0, 2.0 * power_ratio)
        wiener_gain = wiener_gain * speech_weight
        
        return spec * wiener_gain

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

    def _process_spectral_subtraction(self, audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
        """Apply spectral subtraction for noise reduction.
        
        Args:
            audio_data: Input audio data
            sample_rate: Audio sample rate
            
        Returns:
            Noise reduced audio data
        """
        # Convert to mono if stereo
        if len(audio_data.shape) > 1:
            audio_data = audio_data.mean(axis=1)
            
        # Convert to float32 and normalize
        audio_float = audio_data.astype(np.float32)
        audio_float = audio_float / np.max(np.abs(audio_float))

        # STFT parameters
        nperseg = int(0.025 * sample_rate)  # 25ms window
        noverlap = int(0.015 * sample_rate)  # 15ms overlap
        
        # Compute STFT
        _, _, stft = signal.stft(
            audio_float,
            fs=sample_rate,
            nperseg=nperseg,
            noverlap=noverlap,
            window='hann'
        )
        
        # Estimate noise from first 100ms
        noise_duration = int(0.1 * sample_rate)
        noise_frames = min(int(noise_duration / (nperseg - noverlap)), stft.shape[1])
        noise_profile = np.mean(np.abs(stft[:, :noise_frames]) ** 2, axis=1)
        
        # Apply spectral subtraction with Wiener filtering
        stft_mag = np.abs(stft)
        stft_phase = np.angle(stft)
        power_spec = stft_mag ** 2
        
        # Compute SNR-based Wiener filter
        snr = np.maximum(power_spec - noise_profile[:, np.newaxis], 0) / (noise_profile[:, np.newaxis] + 1e-10)
        wiener_gain = snr / (1 + snr)
        
        # Apply flooring to prevent musical noise
        wiener_gain = np.maximum(wiener_gain, self._spectral_floor)
        
        # Apply the filter
        enhanced_mag = stft_mag * wiener_gain
        enhanced_stft = enhanced_mag * np.exp(1j * stft_phase)
        
        # Inverse STFT
        _, enhanced_audio = signal.istft(
            enhanced_stft,
            fs=sample_rate,
            nperseg=nperseg,
            noverlap=noverlap,
            window='hann'
        )
        
        # Apply highpass filter if configured
        if self._highpass_cutoff > 0:
            nyquist = sample_rate / 2
            cutoff = self._highpass_cutoff / nyquist
            b, a = butter(2, cutoff, 'high')
            enhanced_audio = filtfilt(b, a, enhanced_audio)
        
        return enhanced_audio

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
                    "input_path": input_path,
                    "exists": os.path.exists(input_path),
                    "size": os.path.getsize(input_path) if os.path.exists(input_path) else 0
                }
            )

            # Read audio file
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                sample_rate, audio_data = wavfile.read(input_path)

            # Apply noise reduction
            logger.debug(
                "Applying noise reduction",
                extra={
                    "req_id": self._req_id,
                    "plugin_name": self.name,
                    "recording_id": recording_id,
                    "audio_shape": audio_data.shape if hasattr(audio_data, 'shape') else None,
                    "sample_rate": sample_rate,
                    "output_path": output_path,
                    "settings": {
                        "noise_reduce_factor": self._noise_reduce_factor,
                        "wiener_alpha": self._wiener_alpha,
                        "highpass_cutoff": self._highpass_cutoff,
                        "spectral_floor": self._spectral_floor,
                        "smoothing_factor": self._smoothing_factor
                    }
                }
            )

            # Process audio using spectral subtraction
            enhanced_audio = self._process_spectral_subtraction(audio_data, sample_rate)

            # Convert back to int16
            logger.debug(
                "Converting to int16",
                extra={
                    "req_id": self._req_id,
                    "plugin_name": self.name,
                    "recording_id": recording_id,
                    "min_val": float(np.min(enhanced_audio)),
                    "max_val": float(np.max(enhanced_audio))
                }
            )
            output_audio = np.clip(enhanced_audio * 32768.0, -32768, 32767).astype(np.int16)

            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # Write output file
            logger.debug(
                f"Writing output file",
                extra={
                    "req_id": self._req_id,
                    "plugin_name": self.name,
                    "recording_id": recording_id,
                    "output_path": output_path,
                    "output_shape": output_audio.shape,
                    "output_dtype": str(output_audio.dtype)
                }
            )
            wavfile.write(output_path, sample_rate, output_audio)

            # Log success
            logger.info(
                "Successfully processed audio file",
                extra={
                    "req_id": self._req_id,
                    "plugin_name": self.name,
                    "recording_id": recording_id,
                    "input_path": input_path,
                    "output_path": output_path,
                    "output_size": os.path.getsize(output_path)
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
                    "error_type": type(e).__name__,
                    "input_path": input_path,
                    "output_path": output_path,
                    "traceback": traceback.format_exc()
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
                logger.debug(
                    "Skipping our own event",
                    extra={
                        "req_id": event_id,
                        "plugin_name": self.name,
                        "recording_id": recording_id
                    }
                )
                return

            # Initialize database connection if needed
            if not self._db:
                self._db = await get_db_async()

            # Insert or update task status - always allow reprocessing
            await self._db.execute(
                """
                INSERT INTO plugin_tasks (recording_id, plugin_name, status, created_at, updated_at)
                VALUES (?, ?, 'processing', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                ON CONFLICT(recording_id, plugin_name) 
                DO UPDATE SET status = 'processing', updated_at = CURRENT_TIMESTAMP
                """,
                (recording_id, self.name)
            )

            # Process the audio files
            await self._process_audio_files(recording_id, sys_path, mic_path)

        except Exception as e:
            logger.error(
                "Error handling recording ended event",
                extra={
                    "req_id": event_id,
                    "plugin_name": self.name,
                    "error": str(e)
                },
                exc_info=True
            )

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
        """Process the audio files for noise reduction."""
        try:
            logger.info(
                "Starting audio processing",
                extra={
                    "req_id": self._req_id,
                    "plugin_name": self.name,
                    "recording_id": recording_id,
                    "system_audio_path": system_audio_path,
                    "microphone_audio_path": microphone_audio_path
                }
            )

            # Create output directory if it doesn't exist
            os.makedirs(self._output_directory, exist_ok=True)

            # Process audio files
            if system_audio_path and microphone_audio_path and os.path.exists(system_audio_path) and os.path.exists(microphone_audio_path):
                output_path = Path(self._output_directory) / f"{recording_id}_microphone_cleaned.wav"
                
                # Process in thread pool
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    self._executor,
                    self.reduce_noise,
                    microphone_audio_path,
                    system_audio_path,
                    str(output_path),
                    self._noise_reduce_factor,
                    self._wiener_alpha,
                    self._highpass_cutoff,
                    self._spectral_floor,
                    self._smoothing_factor
                )

                # Update task status to completed
                if self._db:
                    await self._db.execute(
                        """
                        UPDATE plugin_tasks 
                        SET status = 'completed', updated_at = CURRENT_TIMESTAMP,
                            output_paths = ?
                        WHERE recording_id = ? AND plugin_name = ?
                        """,
                        (str(output_path), recording_id, self.name)
                    )

                logger.info(
                    "Audio processing completed successfully",
                    extra={
                        "req_id": self._req_id,
                        "plugin_name": self.name,
                        "recording_id": recording_id,
                        "output_path": str(output_path)
                    }
                )
            else:
                logger.warning(
                    "Missing audio files",
                    extra={
                        "req_id": self._req_id,
                        "plugin_name": self.name,
                        "recording_id": recording_id,
                        "system_exists": os.path.exists(system_audio_path) if system_audio_path else False,
                        "mic_exists": os.path.exists(microphone_audio_path) if microphone_audio_path else False
                    }
                )

        except Exception as e:
            logger.error(
                "Error processing audio files",
                extra={
                    "req_id": self._req_id,
                    "plugin_name": self.name,
                    "recording_id": recording_id,
                    "error": str(e)
                },
                exc_info=True
            )
            # Update task status to failed
            if self._db:
                try:
                    await self._db.execute(
                        """
                        UPDATE plugin_tasks 
                        SET status = 'failed', updated_at = CURRENT_TIMESTAMP,
                            error_message = ?
                        WHERE recording_id = ? AND plugin_name = ?
                        """,
                        (str(e), recording_id, self.name)
                    )
                except Exception as db_error:
                    logger.error(
                        "Failed to update task status",
                        extra={
                            "req_id": self._req_id,
                            "plugin_name": self.name,
                            "recording_id": recording_id,
                            "error": str(db_error)
                        },
                        exc_info=True
                    )
