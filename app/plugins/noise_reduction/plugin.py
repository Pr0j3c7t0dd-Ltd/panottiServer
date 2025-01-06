"""Noise reduction plugin with an advanced frequency-domain bleed removal."""

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
import soundfile as sf
from scipy import signal
from scipy.io import wavfile
from scipy.signal import butter, filtfilt

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
    """
    Plugin for noise reduction and bleed removal.
    Includes:
      - Time-domain bleed removal (simple).
      - Frequency-domain bleed removal (advanced).
      - Original spectral noise reduction (Wiener/subtraction).
    """

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
        
        # Existing toggles
        self._time_domain_subtraction = bool(config_dict.get("time_domain_subtraction", False))

        # NEW: Frequency-domain bleed approach
        self._freq_domain_bleed_removal = bool(config_dict.get("freq_domain_bleed_removal", False))

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
                    "freq_domain_bleed_removal": self._freq_domain_bleed_removal,
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
    #  Time alignment helpers
    # ------------------------------------------------------------------------
    def _align_signals_by_fft(
        self, mic_data: np.ndarray, sys_data: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Align sys_data to mic_data using FFT-based cross-correlation on a chunk.
        """
        logger.debug("Preparing chunk-based alignment via FFT cross-correlation",
                     extra={
                         "req_id": self._req_id,
                         "plugin_name": self.name,
                         "chunk_secs": self._alignment_chunk_seconds
                     })

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
    # 1) Basic time-domain bleed removal (unchanged)
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
        Simple time-domain approach with a global scale factor.
        difference = mic - alpha*sys
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
            raise ValueError("Mic and system sample rates differ.")

        if mic_data.ndim > 1:
            mic_data = mic_data.mean(axis=1)
        if sys_data.ndim > 1:
            sys_data = sys_data.mean(axis=1)

        if do_alignment:
            if self._use_fft_alignment:
                mic_data, sys_data = self._align_signals_by_fft(mic_data, sys_data)
            else:
                mic_data, sys_data = self._align_signals_by_ccf(mic_data, sys_data)

        alpha = 1.0
        if auto_scale:
            denom = np.dot(sys_data, sys_data)
            if denom > 1e-9:
                alpha = np.dot(mic_data, sys_data) / denom

        difference = mic_data - alpha * sys_data
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
    # 2) Advanced frequency-domain bleed removal
    # ------------------------------------------------------------------------
    def _remove_bleed_frequency_domain(
        self,
        mic_file: str,
        sys_file: str,
        output_file: str,
        do_alignment: bool = True,
        randomize_phase: bool = True
    ) -> None:
        """
        Frequency-domain bleed removal:
          1) Align signals.
          2) STFT mic and system.
          3) Subtract system's spectrum from mic's, bin-by-bin,
             with a small spectral floor + optional random phase
             to ensure leftover is unintelligible.
        """

        logger.info(
            "Performing frequency-domain bleed removal",
            extra={
                "plugin_name": self.name,
                "mic_file": mic_file,
                "sys_file": sys_file,
                "output_file": output_file,
                "do_alignment": do_alignment,
                "randomize_phase": randomize_phase
            },
        )

        if not os.path.exists(mic_file):
            raise FileNotFoundError(f"Mic file not found: {mic_file}")
        if not os.path.exists(sys_file):
            raise FileNotFoundError(f"System file not found: {sys_file}")

        # Read audio
        mic_data, mic_sr = sf.read(mic_file)
        sys_data, sys_sr = sf.read(sys_file)

        if mic_sr != sys_sr:
            raise ValueError("Mic and system sample rates differ.")

        # Mono
        if mic_data.ndim > 1:
            mic_data = mic_data.mean(axis=1)
        if sys_data.ndim > 1:
            sys_data = sys_data.mean(axis=1)

        # Optionally align
        if do_alignment:
            if self._use_fft_alignment:
                mic_data, sys_data = self._align_signals_by_fft(mic_data, sys_data)
            else:
                mic_data, sys_data = self._align_signals_by_ccf(mic_data, sys_data)

        # Normalize
        mic_max = np.max(np.abs(mic_data)) or 1e-9
        sys_max = np.max(np.abs(sys_data)) or 1e-9
        mic_data = mic_data.astype(np.float32) / mic_max
        sys_data = sys_data.astype(np.float32) / sys_max

        # STFT parameters
        nperseg = 2048
        noverlap = nperseg // 2
        window = "hann"

        # STFT
        f_mic, t_mic, mic_stft = signal.stft(
            mic_data,
            fs=mic_sr,
            nperseg=nperseg,
            noverlap=noverlap,
            window=window
        )
        _, _, sys_stft = signal.stft(
            sys_data,
            fs=mic_sr,
            nperseg=nperseg,
            noverlap=noverlap,
            window=window
        )

        # Ensure both STFT have the same shape (if not, truncate to the min shape)
        min_time_frames = min(mic_stft.shape[1], sys_stft.shape[1])
        mic_stft = mic_stft[:, :min_time_frames]
        sys_stft = sys_stft[:, :min_time_frames]

        # Subtraction approach
        # mic_stft_clean = mic_stft - alpha * sys_stft,
        # but we can find alpha per frequency or per bin for better removal. 
        # For simplicity, do an auto-scale per bin:
        mic_mag = np.abs(mic_stft)
        sys_mag = np.abs(sys_stft)
        mic_phase = np.angle(mic_stft)
        sys_phase = np.angle(sys_stft)

        # To degrade system bleed, we can compute alpha per time-frequency bin:
        # alpha(f,t) = (mic_mag * sys_mag) / (sys_mag^2 + epsilon)
        # or simpler: alpha(f,t) = 1 if sys is significant,
        # but let's do a scaled approach.
        epsilon = 1e-9
        alpha = (mic_mag * sys_mag) / (sys_mag**2 + epsilon)

        # We'll limit alpha to [0, 1.2], so we don't over-subtract:
        alpha = np.clip(alpha, 0.0, 1.2)

        # Perform the bin-by-bin subtraction
        bleed_removed_mag = mic_mag - alpha * sys_mag

        # Floor any negative result to a small spectral floor:
        # e.g., 0.02 * mic_mag => ensures we don't produce huge "musical" holes
        # You can tweak these constants to degrade the bleed more/less.
        spectral_floor = 0.02 * mic_mag  
        bleed_removed_mag = np.maximum(bleed_removed_mag, spectral_floor)

        # Optionally randomize the phase where system was dominant
        # if sys_mag > mic_mag, randomize that bin's phase:
        if randomize_phase:
            dominant_mask = sys_mag > mic_mag
            rand_phase = 2.0 * np.pi * np.random.rand(*dominant_mask.shape)
            # We'll apply random phase only in dominant bins
            final_phase = np.where(dominant_mask, rand_phase, mic_phase)
        else:
            final_phase = mic_phase

        # Reconstruct complex STFT
        bleed_removed_stft = bleed_removed_mag * np.exp(1j * final_phase)

        # iSTFT
        _, cleaned_audio = signal.istft(
            bleed_removed_stft,
            fs=mic_sr,
            nperseg=nperseg,
            noverlap=noverlap,
            window=window
        )

        # Optional final highpass to remove rumble
        if self._highpass_cutoff > 0:
            nyq = mic_sr / 2
            cutoff = self._highpass_cutoff / nyq
            b, a = butter(2, cutoff, btype="high")
            cleaned_audio = filtfilt(b, a, cleaned_audio)

        # Normalize to avoid clipping
        max_val = np.max(np.abs(cleaned_audio))
        if max_val > 0:
            cleaned_audio /= max_val

        # Save
        cleaned_audio = (cleaned_audio * 32767).astype(np.int16)
        sf.write(output_file, cleaned_audio, mic_sr)

        logger.info(
            "Saved frequency-domain bleed removal output",
            extra={
                "plugin_name": self.name,
                "output_file": output_file,
                "final_size": os.path.getsize(output_file)
            }
        )

    # ------------------------------------------------------------------------
    # Original spectral noise reduction (Wiener/subtraction)
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
        """
        Existing spectral approach ...
        """
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
        # ... [existing code unchanged] ...
        # [Truncated for brevity in this snippet]
        pass

    def compute_noise_profile(
        self,
        noise_data: np.ndarray,
        fs: float,
        nperseg: int = 2048,
        noverlap: int = 1024,
        smooth_factor: int = 2
    ) -> np.ndarray:
        """Existing helper for spectral approach."""
        # ... [unchanged] ...
        pass

    def wiener_filter(
        self,
        spec: np.ndarray,
        noise_power: np.ndarray,
        alpha: float = 2.2,
        second_pass: bool = True
    ) -> np.ndarray:
        """Existing Wiener filter approach."""
        # ... [unchanged] ...
        pass

    # ------------------------------------------------------------------------
    # Event handling
    # ------------------------------------------------------------------------
    async def handle_recording_ended(self, event: EventData) -> None:
        """
        Called when a recording ends. We decide how to process it
        based on config toggles.
        """
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
                logger.error("No recording_id found in event data", extra={...})
                return

            source_plugin = (
                event.get("source_plugin") if isinstance(event, dict)
                else getattr(event.context, "source_plugin", None) if hasattr(event, "context")
                else None
            )
            if source_plugin == self.name:
                logger.debug("Skipping our own event", extra={...})
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
            logger.error("Error handling recording ended event", extra={...}, exc_info=True)

    async def process_recording(self, recording_id: str, event_data: EventData) -> None:
        """Called from code to process a recording with the configured approach."""
        try:
            logger.info("Starting audio processing", extra={...})
            # Extract mic/system paths from event_data ...
            # Then call _process_audio_files ...
            pass
        except Exception as e:
            logger.error("Failed to process recording", extra={...}, exc_info=True)
            raise

    async def _process_audio_files(self, recording_id: str, system_audio_path: str | None, microphone_audio_path: str | None) -> None:
        """
        Decide which approach to use:
          1) time_domain_subtraction => simple time-domain approach
          2) freq_domain_bleed_removal => advanced frequency-domain approach
          3) else => fallback to original spectral noise reduction
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
                    "time_domain_subtraction": self._time_domain_subtraction,
                    "freq_domain_bleed_removal": self._freq_domain_bleed_removal
                }
            )

            if system_audio_path and microphone_audio_path \
               and os.path.exists(system_audio_path) \
               and os.path.exists(microphone_audio_path):
                loop = asyncio.get_event_loop()

                if self._freq_domain_bleed_removal:
                    # 2) Frequency-domain bleed removal
                    output_path = self._output_directory / f"{recording_id}_mic_bleed_removed_freq.wav"
                    await loop.run_in_executor(
                        self._executor,
                        self._remove_bleed_frequency_domain,
                        microphone_audio_path,
                        system_audio_path,
                        str(output_path),
                        True,   # do_alignment
                        True    # randomize_phase
                    )
                    final_output = output_path

                elif self._time_domain_subtraction:
                    # 1) Simple time-domain approach
                    output_path = self._output_directory / f"{recording_id}_mic_bleed_removed_time.wav"
                    await loop.run_in_executor(
                        self._executor,
                        self._subtract_bleed_time_domain,
                        microphone_audio_path,
                        system_audio_path,
                        str(output_path),
                        True,   # do_alignment
                        True    # auto_scale
                    )
                    final_output = output_path

                else:
                    # 3) Fallback to original spectral noise reduction
                    output_path = self._output_directory / f"{recording_id}_microphone_cleaned.wav"
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
                    final_output = output_path

                # Mark DB status
                if self._db:
                    await self._db.execute(
                        """
                        UPDATE plugin_tasks 
                        SET status = 'completed', updated_at = CURRENT_TIMESTAMP,
                            output_paths = ?
                        WHERE recording_id = ? AND plugin_name = ?
                        """,
                        (str(final_output), recording_id, self.name)
                    )

                logger.info(
                    "Audio processing completed successfully",
                    extra={
                        "req_id": self._req_id,
                        "plugin_name": self.name,
                        "recording_id": recording_id,
                        "output_path": str(final_output),
                        "time_domain_subtraction": self._time_domain_subtraction,
                        "freq_domain_bleed_removal": self._freq_domain_bleed_removal
                    }
                )

                # Emit completion event
                if self.event_bus:
                    await self.event_bus.publish({
                        "event": "noise_reduction.completed",
                        "recording_id": recording_id,
                        "output_path": str(final_output),
                        "original_audio_path": microphone_audio_path,
                        "event_id": f"{recording_id}_noise_reduction_completed",
                        "timestamp": datetime.utcnow().isoformat(),
                        "plugin_id": self.name,
                        "metadata": {
                            "time_domain_subtraction": self._time_domain_subtraction,
                            "freq_domain_bleed_removal": self._freq_domain_bleed_removal
                        }
                    })

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
            logger.error("Error processing audio files", extra={...}, exc_info=True)
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
                    logger.error("Failed to update task status", extra={...}, exc_info=True)

    async def _shutdown(self) -> None:
        """Shutdown plugin."""
        try:
            if self.event_bus is not None:
                logger.info("Unsubscribing from recording.ended event", extra={...})
                await self.event_bus.unsubscribe("recording.ended", self.handle_recording_ended)

            if self._executor is not None:
                logger.info("Shutting down thread pool", extra={...})
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, self._executor.shutdown, True)
                self._executor = None

            if self._db is not None:
                logger.info("Closing database connection", extra={...})
                await self._db.close()
                self._db = None

            logger.info("Plugin shutdown complete", extra={...})

        except Exception as e:
            logger.error("Error during plugin shutdown", extra={...}, exc_info=True)
            raise
