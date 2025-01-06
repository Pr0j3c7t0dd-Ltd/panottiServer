"""Noise reduction plugin with optional time-domain bleed removal."""

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
import soundfile as sf

from app.models.database import get_db_async, DatabaseManager
from app.models.recording.events import RecordingEvent
from app.plugins.base import PluginBase, PluginConfig
from app.plugins.events.bus import EventBus as PluginEventBus
from app.plugins.events.models import EventContext
from app.utils.logging_config import get_logger

logger = get_logger("app.plugins.noise_reduction.plugin")
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
    """Plugin for noise reduction and time-domain bleed removal."""

    def __init__(
        self, config: PluginConfig, event_bus: PluginEventBus | None = None
    ) -> None:
        super().__init__(config, event_bus)
        
        config_dict = config.config or {}
        self._output_directory = Path(config_dict.get("output_directory", "data/cleaned_audio"))
        self._noise_reduce_factor = float(config_dict.get("noise_reduce_factor", 1.0))
        self._wiener_alpha = float(config_dict.get("wiener_alpha", 2.5))
        self._highpass_cutoff = float(config_dict.get("highpass_cutoff", 95))
        self._spectral_floor = float(config_dict.get("spectral_floor", 0.04))
        self._smoothing_factor = int(config_dict.get("smoothing_factor", 2))
        self._max_workers = int(config_dict.get("max_concurrent_tasks", 4))
        
        # Toggle if we want to do time-domain bleed removal vs. spectral reduction
        self._time_domain_subtraction = bool(config_dict.get("time_domain_subtraction", False))

        # Alignment options
        self._use_fft_alignment = bool(config_dict.get("use_fft_alignment", True))
        self._alignment_chunk_seconds = int(config_dict.get("alignment_chunk_seconds", 10))

        # Thread pool
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

            for attempt in range(3):
                try:
                    db_manager = await get_db_async()
                    self._db = await db_manager
                    break
                except Exception as e:
                    if attempt == 2:
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
                    await asyncio.sleep(1)

            os.makedirs(self._output_directory, exist_ok=True)

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
                    "output_directory": str(self._output_directory),
                    "time_domain_subtraction": self._time_domain_subtraction,
                    "use_fft_alignment": self._use_fft_alignment,
                    "alignment_chunk_seconds": self._alignment_chunk_seconds
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

    # ------------------------------------------------------------------------
    # Alignment helpers
    # ------------------------------------------------------------------------
    def _align_signals_by_fft(
        self, mic_data: np.ndarray, sys_data: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Align sys_data to mic_data using FFT-based cross-correlation on a chunk
        of the signals to avoid huge O(N^2) costs.
        """
        logger.debug("Preparing chunk-based alignment via FFT cross-correlation",
                     extra={
                         "req_id": self._req_id,
                         "plugin_name": self.name,
                         "chunk_secs": self._alignment_chunk_seconds
                     })

        # We'll take ~N samples from each file where N = alignment_chunk_seconds * 48000 (approx)
        # or clamp to the length of the shorter array.
        chunk_len = min(len(mic_data), len(sys_data))
        sr_guess = 48000
        if self._alignment_chunk_seconds > 0:
            chunk_limit = int(self._alignment_chunk_seconds * sr_guess)
            chunk_len = min(chunk_len, chunk_limit)

        mic_chunk = mic_data[:chunk_len].astype(np.float32)
        sys_chunk = sys_data[:chunk_len].astype(np.float32)

        logger.debug("Performing FFT-based cross-correlation on chunk",
                     extra={
                         "req_id": self._req_id,
                         "plugin_name": self.name,
                         "mic_chunk_len": len(mic_chunk),
                         "sys_chunk_len": len(sys_chunk)
                     })

        corr = signal.correlate(mic_chunk, sys_chunk, mode='full', method='fft')
        best_lag = np.argmax(corr) - (len(sys_chunk) - 1)

        logger.debug("FFT cross-correlation complete",
                     extra={
                         "req_id": self._req_id,
                         "plugin_name": self.name,
                         "best_lag": best_lag,
                         "corr_len": len(corr)
                     })

        if best_lag > 0:
            # Shift sys_data to the right
            sys_aligned = np.pad(sys_data, (best_lag, 0), 'constant')
            mic_aligned = mic_data
        else:
            mic_aligned = np.pad(mic_data, (-best_lag, 0), 'constant')
            sys_aligned = sys_data

        min_len = min(len(mic_aligned), len(sys_aligned))
        mic_aligned = mic_aligned[:min_len]
        sys_aligned = sys_aligned[:min_len]

        logger.debug("Alignment done, returning aligned signals",
                     extra={
                         "req_id": self._req_id,
                         "plugin_name": self.name,
                         "aligned_len": min_len
                     })
        return mic_aligned, sys_aligned

    def _align_signals_by_ccf(
        self, mic_data: np.ndarray, sys_data: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Naive cross-correlation using np.correlate, O(N^2).
        Used if the user disables FFT alignment (use_fft_alignment=False).
        """
        logger.debug("Starting naive cross-correlation (O(N^2))",
                     extra={
                         "req_id": self._req_id,
                         "plugin_name": self.name,
                         "mic_len": len(mic_data),
                         "sys_len": len(sys_data),
                     })
        mic_data = mic_data.astype(np.float32)
        sys_data = sys_data.astype(np.float32)
        corr = np.correlate(mic_data, sys_data, mode='full')
        best_lag = np.argmax(corr) - (len(sys_data) - 1)

        logger.debug("Naive correlation complete",
                     extra={
                         "req_id": self._req_id,
                         "plugin_name": self.name,
                         "best_lag": best_lag
                     })

        if best_lag > 0:
            sys_aligned = np.pad(sys_data, (best_lag, 0), 'constant')
            mic_aligned = mic_data
        else:
            mic_aligned = np.pad(mic_data, (-best_lag, 0), 'constant')
            sys_aligned = sys_data

        min_len = min(len(mic_aligned), len(sys_aligned))
        mic_aligned = mic_aligned[:min_len]
        sys_aligned = sys_aligned[:min_len]
        return mic_aligned, sys_aligned

    # ------------------------------------------------------------------------
    # New time-domain bleed removal approach
    # ------------------------------------------------------------------------
    def _subtract_bleed_time_domain(
        self,
        mic_file: str,
        sys_file: str,
        output_file: str,
        do_alignment: bool = True,
        auto_scale: bool = True
    ) -> None:
        """
        Time-domain bleed removal:
          1) Optionally align via cross-correlation (FFT-based by default).
          2) Estimate scale factor alpha = (mic·sys)/(sys·sys) if auto_scale=True.
          3) difference = mic - alpha * sys
        """
        logger.info(
            "Performing time-domain bleed removal",
            extra={
                "plugin_name": self.name,
                "mic_file": mic_file,
                "sys_file": sys_file,
                "output_file": output_file,
                "do_alignment": do_alignment,
                "auto_scale": auto_scale
            },
        )

        if not os.path.exists(mic_file):
            raise FileNotFoundError(f"Mic file not found: {mic_file}")
        if not os.path.exists(sys_file):
            raise FileNotFoundError(f"System file not found: {sys_file}")

        mic_data, mic_sr = sf.read(mic_file)
        sys_data, sys_sr = sf.read(sys_file)

        if mic_sr != sys_sr:
            raise ValueError("Mic and system sample rates differ. Resample before subtracting.")

        # Convert to mono if stereo
        if mic_data.ndim > 1:
            mic_data = mic_data.mean(axis=1)
        if sys_data.ndim > 1:
            sys_data = sys_data.mean(axis=1)

        # Align if requested
        if do_alignment:
            if self._use_fft_alignment:
                mic_data, sys_data = self._align_signals_by_fft(mic_data, sys_data)
            else:
                mic_data, sys_data = self._align_signals_by_ccf(mic_data, sys_data)

        # Auto-estimate scale factor
        alpha = 1.0
        if auto_scale:
            denom = np.dot(sys_data, sys_data)
            if denom > 1e-9:
                alpha = np.dot(mic_data, sys_data) / denom

        logger.debug("Estimated scaling factor for bleed removal",
                     extra={"req_id": self._req_id, "plugin_name": self.name, "alpha": alpha})

        difference = mic_data - alpha * sys_data

        # Normalize if necessary
        max_val = np.max(np.abs(difference))
        if max_val > 1.0:
            difference /= max_val

        sf.write(output_file, difference, mic_sr)
        logger.info(
            "Saved time-domain-subtracted audio (bleed removal)",
            extra={
                "plugin_name": self.name,
                "output_file": output_file,
                "length_samples": len(difference),
            }
        )

    # ------------------------------------------------------------------------
    # SPECTRAL NOISE REDUCTION SECTION (Unchanged, except for logs)
    # ------------------------------------------------------------------------
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
        logger.info(
            "Starting enhanced noise reduction (spectral subtraction + Wiener)",
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
        
        if not os.path.exists(mic_file):
            raise FileNotFoundError(f"Microphone audio file not found: {mic_file}")
        if not os.path.exists(noise_file):
            raise FileNotFoundError(f"System audio file not found: {noise_file}")
            
        output_dir = os.path.dirname(os.path.abspath(output_file))
        os.makedirs(output_dir, exist_ok=True)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", wavfile.WavFileWarning)
            mic_rate, mic_data = wavfile.read(mic_file)
            noise_rate, noise_data = wavfile.read(noise_file)
        
        if len(mic_data.shape) > 1:
            mic_data = np.mean(mic_data, axis=1)
        if len(noise_data.shape) > 1:
            noise_data = np.mean(noise_data, axis=1)
            
        mic_data = mic_data.astype(np.float32)
        noise_data = noise_data.astype(np.float32)
        
        mic_max = np.max(np.abs(mic_data))
        noise_max = np.max(np.abs(noise_data))
        
        if mic_max == 0 or noise_max == 0:
            raise ValueError("Input audio is silent")
            
        mic_data = mic_data / mic_max
        noise_data = noise_data / noise_max
        
        if highpass_cutoff > 0:
            b, a = butter(5, highpass_cutoff / (mic_rate/2), btype='high')
            mic_data = filtfilt(b, a, mic_data)
            noise_data = filtfilt(b, a, noise_data)
        
        nperseg = 2048
        noverlap = nperseg // 2
        _, _, mic_spec = signal.stft(mic_data, fs=mic_rate, nperseg=nperseg, noverlap=noverlap)
        
        noise_profile = self.compute_noise_profile(
            noise_data, fs=noise_rate, nperseg=nperseg, noverlap=noverlap, smooth_factor=smoothing_factor
        )
        
        if wiener_alpha > 0:
            cleaned_spec = self.wiener_filter(
                mic_spec, noise_profile * noise_reduce_factor, alpha=wiener_alpha
            )
        else:
            mic_mag = np.abs(mic_spec)
            reduction = noise_profile * noise_reduce_factor
            cleaned_mag = np.maximum(mic_mag - reduction, mic_mag * spectral_floor)
            cleaned_spec = cleaned_mag * np.exp(1j * np.angle(mic_spec))
        
        _, cleaned_audio = signal.istft(cleaned_spec, fs=mic_rate, nperseg=nperseg, noverlap=noverlap)
        
        cleaned_max = np.max(np.abs(cleaned_audio))
        if cleaned_max > 0:
            cleaned_audio = cleaned_audio / cleaned_max
            
        cleaned_audio = np.clip(cleaned_audio * 32767, -32768, 32767).astype(np.int16)
        
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

    def compute_noise_profile(
        self,
        noise_data: np.ndarray,
        fs: float,
        nperseg: int = 2048,
        noverlap: int = 1024,
        smooth_factor: int = 2
    ) -> np.ndarray:
        f, t, noise_spec = signal.stft(noise_data, fs=fs, nperseg=nperseg, noverlap=noverlap)
        freqs = f
        noise_profile = np.minimum(
            np.mean(np.abs(noise_spec), axis=1),
            np.percentile(np.abs(noise_spec), 25, axis=1)
        )
        
        speech_mask = np.ones_like(freqs)
        background_range = (freqs >= 100) & (freqs <= 1000)
        speech_mask[background_range] = 0.8
        noise_profile = noise_profile * speech_mask * 0.5
        
        if smooth_factor > 0:
            window_size = 2 * smooth_factor + 1
            noise_profile = np.convolve(
                noise_profile, np.ones(window_size)/window_size, mode='same'
            )
        
        return noise_profile.reshape(-1, 1)

    def wiener_filter(
        self,
        spec: np.ndarray,
        noise_power: np.ndarray,
        alpha: float = 2.2,
        second_pass: bool = True
    ) -> np.ndarray:
        sig_power = np.abs(spec)**2
        noise_power = noise_power * 0.5
        snr = sig_power / (noise_power + 1e-10)
        wiener_gain = np.maximum(1 - alpha / (snr + 1), 0.3)
        
        power_ratio = sig_power / (np.max(sig_power) + 1e-10)
        speech_weight = np.minimum(1.0, 1.5 * power_ratio)
        wiener_gain = wiener_gain * speech_weight
        wiener_gain = np.maximum(wiener_gain, 0.4)
        
        enhanced_spec = spec * wiener_gain
        
        if second_pass:
            sig_power_2 = np.abs(enhanced_spec)**2
            snr_2 = sig_power_2 / (noise_power + 1e-10)
            wiener_gain_2 = np.maximum(1 - (alpha + 0.5) / (snr_2 + 1), 0.35)
            wiener_gain_2 = np.maximum(wiener_gain_2, 0.4)
            enhanced_spec = enhanced_spec * wiener_gain_2
        
        return enhanced_spec

    # ------------------------------------------------------------------------
    # PLUGIN EVENT HANDLING
    # ------------------------------------------------------------------------
    async def handle_recording_ended(self, event: EventData) -> None:
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

            if isinstance(event, dict):
                recording_id = event.get("recording_id")
                current_event = event.get("current_event", {})
                recording_data = current_event.get("recording", {})
                audio_paths = recording_data.get("audio_paths", {})
                mic_path = audio_paths.get("microphone")
                sys_path = audio_paths.get("system")
            else:
                recording_id = event.recording_id
                mic_path = event.microphone_audio_path
                sys_path = event.system_audio_path

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

            source_plugin = (
                event.get("source_plugin") if isinstance(event, dict)
                else getattr(event.context, "source_plugin", None) if hasattr(event, "context")
                else None
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

            if not self._db:
                self._db = await get_db_async()

            await self._db.execute(
                """
                INSERT INTO plugin_tasks (recording_id, plugin_name, status, created_at, updated_at)
                VALUES (?, ?, 'processing', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                ON CONFLICT(recording_id, plugin_name) 
                DO UPDATE SET status = 'processing', updated_at = CURRENT_TIMESTAMP
                """,
                (recording_id, self.name)
            )

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

    async def process_recording(self, recording_id: str, event_data: EventData) -> None:
        try:
            logger.info(
                "Starting audio processing",
                extra={
                    "req_id": self._req_id,
                    "plugin_name": self.name,
                    "recording_id": recording_id
                }
            )

            if isinstance(event_data, dict):
                system_audio_path = event_data.get("system_audio_path")
                microphone_audio_path = event_data.get("microphone_audio_path")
            else:
                system_audio_path = getattr(event_data, "system_audio_path", None)
                microphone_audio_path = getattr(event_data, "microphone_audio_path", None)

            await self._process_audio_files(recording_id, system_audio_path, microphone_audio_path)

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

    async def _process_audio_files(self, recording_id: str, system_audio_path: str | None, microphone_audio_path: str | None) -> None:
        """
        Decide which approach to use based on _time_domain_subtraction flag:
          - If True: Use the new time-domain bleed removal
          - If False: Use spectral noise reduction
        """
        try:
            logger.info(
                "Starting audio processing for both system and microphone",
                extra={
                    "req_id": self._req_id,
                    "plugin_name": self.name,
                    "recording_id": recording_id,
                    "system_audio_path": system_audio_path,
                    "microphone_audio_path": microphone_audio_path,
                    "time_domain_subtraction": self._time_domain_subtraction
                }
            )

            os.makedirs(self._output_directory, exist_ok=True)

            if (
                system_audio_path
                and microphone_audio_path
                and os.path.exists(system_audio_path)
                and os.path.exists(microphone_audio_path)
            ):
                loop = asyncio.get_event_loop()

                if self._time_domain_subtraction:
                    # We'll call our new bleed-removal method
                    output_path_mic = Path(self._output_directory) / f"{recording_id}_microphone_bleed_removed.wav"
                    logger.debug("Scheduling time-domain bleed removal in Executor",
                                 extra={
                                     "req_id": self._req_id,
                                     "plugin_name": self.name,
                                     "recording_id": recording_id,
                                     "output_path": str(output_path_mic)
                                 })
                    await loop.run_in_executor(
                        self._executor,
                        self._subtract_bleed_time_domain,
                        microphone_audio_path,
                        system_audio_path,
                        str(output_path_mic),
                        True,   # do_alignment
                        True    # auto_scale
                    )
                    final_output_path = output_path_mic
                else:
                    # Use the existing spectral approach
                    output_path_mic = Path(self._output_directory) / f"{recording_id}_microphone_cleaned.wav"
                    logger.debug("Scheduling spectral noise reduction in Executor",
                                 extra={
                                     "req_id": self._req_id,
                                     "plugin_name": self.name,
                                     "recording_id": recording_id,
                                     "output_path": str(output_path_mic)
                                 })
                    await loop.run_in_executor(
                        self._executor,
                        self.reduce_noise,
                        microphone_audio_path,
                        system_audio_path,
                        str(output_path_mic),
                        self._noise_reduce_factor,
                        self._wiener_alpha,
                        self._highpass_cutoff,
                        self._spectral_floor,
                        self._smoothing_factor
                    )
                    final_output_path = output_path_mic

                # Update DB status
                if self._db:
                    await self._db.execute(
                        """
                        UPDATE plugin_tasks 
                        SET status = 'completed', updated_at = CURRENT_TIMESTAMP,
                            output_paths = ?
                        WHERE recording_id = ? AND plugin_name = ?
                        """,
                        (str(final_output_path), recording_id, self.name)
                    )

                logger.info(
                    "Audio processing completed successfully",
                    extra={
                        "req_id": self._req_id,
                        "plugin_name": self.name,
                        "recording_id": recording_id,
                        "output_path": str(final_output_path),
                        "time_domain_subtraction": self._time_domain_subtraction
                    }
                )
            else:
                logger.warning(
                    "Missing or invalid audio files, skipping noise reduction",
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
                        "Failed to update task status in database",
                        extra={
                            "req_id": self._req_id,
                            "plugin_name": self.name,
                            "recording_id": recording_id,
                            "error": str(db_error)
                        },
                        exc_info=True
                    )

    async def _shutdown(self) -> None:
        """Shutdown plugin."""
        try:
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
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, self._executor.shutdown, True)
                self._executor = None

            if self._db is not None:
                logger.info(
                    "Closing database connection",
                    extra={
                        "req_id": self._req_id,
                        "plugin_name": self.name
                    }
                )
                await self._db.close()
                self._db = None

            logger.info(
                "Plugin shutdown complete",
                extra={
                    "req_id": self._req_id,
                    "plugin_name": self.name
                }
            )

        except Exception as e:
            logger.error(
                "Error during plugin shutdown",
                extra={
                    "req_id": self._req_id,
                    "plugin_name": self.name,
                    "error": str(e)
                },
                exc_info=True
            )
            raise
