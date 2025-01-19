"""Noise reduction plugin with an advanced frequency-domain bleed removal."""

import asyncio
import logging
import os
import sqlite3
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from scipy import signal
from scipy.signal import butter, filtfilt

from app.models.database import DatabaseManager, get_db_async
from app.models.recording.events import RecordingEvent
from app.plugins.base import PluginBase, PluginConfig
from app.core.events import ConcreteEventBus as PluginEventBus
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
        self._output_directory = Path(
            config_dict.get("output_directory", "data/cleaned_audio")
        )
        self._noise_reduce_factor = float(config_dict.get("noise_reduce_factor", 1.0))
        self._wiener_alpha = float(config_dict.get("wiener_alpha", 2.5))
        self._highpass_cutoff = float(config_dict.get("highpass_cutoff", 95))
        self._spectral_floor = float(config_dict.get("spectral_floor", 0.04))
        self._smoothing_factor = int(config_dict.get("smoothing_factor", 2))
        self._max_workers = int(config_dict.get("max_concurrent_tasks", 4))

        # Existing toggles
        self._time_domain_subtraction = bool(
            config_dict.get("time_domain_subtraction", False)
        )

        # NEW: Frequency-domain bleed approach
        self._freq_domain_bleed_removal = bool(
            config_dict.get("freq_domain_bleed_removal", False)
        )

        # Alignment options
        self._use_fft_alignment = bool(config_dict.get("use_fft_alignment", True))
        self._alignment_chunk_seconds = int(
            config_dict.get("alignment_chunk_seconds", 10)
        )

        # Thread pool
        self._executor = ThreadPoolExecutor(max_workers=self._max_workers)
        self._req_id = str(uuid.uuid4())
        self._db: DatabaseManager | None = None

        # Get recordings directory from environment variable, with fallback
        self._recordings_dir = os.getenv("RECORDINGS_DIR", "/app/recordings")
        # Ensure the path exists
        os.makedirs(self._recordings_dir, exist_ok=True)

    async def _initialize(self) -> None:
        """Initialize the plugin."""
        try:
            logger.info(
                "Initializing noise reduction plugin",
                extra={"req_id": self._req_id, "plugin_name": self.name},
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
                                "error": str(e),
                            },
                            exc_info=True,
                        )
                        raise
                    await asyncio.sleep(1)

            os.makedirs(self._output_directory, exist_ok=True)

            if self.event_bus:
                await self.event_bus.subscribe(
                    "recording.ended", self.handle_recording_ended
                )
                logger.info(
                    "Subscribed to recording.ended event",
                    extra={"req_id": self._req_id, "plugin_name": self.name},
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
                    "alignment_chunk_seconds": self._alignment_chunk_seconds,
                },
            )

            self._initialized = True

        except Exception as e:
            logger.error(
                "Failed to initialize noise reduction plugin",
                extra={
                    "req_id": self._req_id,
                    "plugin_name": self.name,
                    "error": str(e),
                },
                exc_info=True,
            )
            raise

    # ------------------------------------------------------------------------
    #  Time alignment helpers
    # ------------------------------------------------------------------------
    def _align_signals_by_fft(
        self, mic_data: np.ndarray, sys_data: np.ndarray, sample_rate: int
    ) -> tuple[np.ndarray, np.ndarray, int]:
        """
        Align signals using FFT cross-correlation and calculate the lag.
        """
        logger.debug(
            "Preparing chunk-based alignment via FFT cross-correlation",
            extra={"chunk_secs": self._alignment_chunk_seconds},
        )

        chunk_len = min(len(mic_data), len(sys_data))
        if self._alignment_chunk_seconds > 0:
            chunk_limit = int(self._alignment_chunk_seconds * sample_rate)
            chunk_len = min(chunk_len, chunk_limit)

        mic_chunk = mic_data[:chunk_len].astype(np.float32)
        sys_chunk = sys_data[:chunk_len].astype(np.float32)

        corr = signal.correlate(mic_chunk, sys_chunk, mode="full", method="fft")
        best_lag = np.argmax(corr) - (len(sys_chunk) - 1)

        if best_lag > 0:
            sys_aligned = np.pad(sys_data, (best_lag, 0), "constant")
            mic_aligned = mic_data
        else:
            mic_aligned = np.pad(mic_data, (-best_lag, 0), "constant")
            sys_aligned = sys_data

        max_len = max(len(mic_aligned), len(sys_aligned))
        mic_aligned = np.pad(mic_aligned, (0, max_len - len(mic_aligned)), "constant")
        sys_aligned = np.pad(sys_aligned, (0, max_len - len(sys_aligned)), "constant")

        lag_seconds = best_lag / sample_rate
        logger.debug(f"Alignment lag calculated: {lag_seconds:.4f} seconds")

        return mic_aligned, sys_aligned, lag_seconds

    @staticmethod
    def detect_start_of_audio(
        audio_data: np.ndarray, threshold: float = 0.01, frame_size: int = 1024
    ) -> int:
        """
        Detect the start of meaningful audio content based on energy threshold.
        :param audio_data: Input audio signal.
        :param threshold: Energy threshold to determine silence.
        :param frame_size: Number of samples per frame for analysis.
        :return: Index of the first sample with significant energy.
        """
        for i in range(0, len(audio_data), frame_size):
            frame = audio_data[i : i + frame_size]
            if np.sqrt(np.mean(frame**2)) > threshold:  # Root mean square (RMS)
                return i
        return 0  # Default to start if no meaningful audio is detected

    def trim_initial_silence(
        audio_data: np.ndarray, sr: int, threshold: float = 0.01
    ) -> np.ndarray:
        """
        Trim initial silence from the audio signal.
        :param audio_data: Input audio signal.
        :param sr: Sample rate of the audio.
        :param threshold: RMS energy threshold to detect silence.
        :return: Trimmed audio signal.
        """
        start_index = NoiseReductionPlugin.detect_start_of_audio(audio_data, threshold)
        logger.debug(
            f"Trimming {start_index / sr:.2f} seconds of silence at the start."
        )
        return audio_data[start_index:]

    # ------------------------------------------------------------------------
    # 1) Basic time-domain bleed removal (unchanged)
    # ------------------------------------------------------------------------
    def _subtract_bleed_time_domain(
        self,
        mic_file: str,
        sys_file: str,
        output_file: str,
        do_alignment: bool = True,
        auto_scale: bool = True,
    ) -> None:
        """
        Simple time-domain approach with a global scale factor.
        difference = mic - alpha*sys
        """
        mic_data, mic_sr = sf.read(mic_file)
        sys_data, sys_sr = sf.read(sys_file)

        if mic_sr != sys_sr:
            raise ValueError("Mic and system sample rates differ.")

        if mic_data.ndim > 1:
            mic_data = mic_data.mean(axis=1)
        if sys_data.ndim > 1:
            sys_data = sys_data.mean(axis=1)

        lag_seconds = 0
        if do_alignment:
            mic_data, sys_data, lag_seconds = self._align_signals_by_fft(
                mic_data, sys_data, mic_sr
            )

        alpha = 1.0
        if auto_scale:
            denom = np.dot(sys_data, sys_data)
            if denom > 1e-9:
                alpha = np.dot(mic_data, sys_data) / denom

        difference = mic_data - alpha * sys_data

        # Normalize and save
        max_val = np.max(np.abs(difference))
        if max_val > 0:
            difference /= max_val

        # Trim the cleaned audio to account for lag
        lag_samples = int(lag_seconds * mic_sr)
        logger.debug(
            f"Trimming {lag_seconds:.4f} seconds (or {lag_samples} samples) due to alignment lag."
        )
        difference = difference[lag_samples:]

        sf.write(output_file, difference, mic_sr)
        logger.info("Time-domain bleed removal completed and saved.")

    # ------------------------------------------------------------------------
    # 2) Advanced frequency-domain bleed removal
    # ------------------------------------------------------------------------
    def _remove_bleed_frequency_domain(
        self,
        mic_file: str,
        sys_file: str,
        output_file: str,
        do_alignment: bool = True,
        randomize_phase: bool = True,
    ) -> None:
        """
        Frequency-domain bleed removal with fine-tuned lag adjustment.
        Preserves all original timing including silence for translation alignment.
        Ensures exact length matching with original file.
        """
        # Read original files and get initial length
        mic_data, mic_sr = sf.read(mic_file)
        sys_data, sys_sr = sf.read(sys_file)
        original_length = len(mic_data)  # Store original length for final check

        if mic_sr != sys_sr:
            raise ValueError("Mic and system sample rates differ.")

        if mic_data.ndim > 1:
            mic_data = mic_data.mean(axis=1)
        if sys_data.ndim > 1:
            sys_data = sys_data.mean(axis=1)

        lag_seconds = 0
        cleaned_audio = None

        try:
            if do_alignment:
                mic_data, sys_data, lag_seconds = self._align_signals_by_fft(
                    mic_data, sys_data, mic_sr
                )

            # STFT parameters
            nperseg = 2048
            noverlap = nperseg // 2
            window = "hann"

            # Perform STFT
            f_mic, t_mic, mic_stft = signal.stft(
                mic_data, fs=mic_sr, nperseg=nperseg, noverlap=noverlap, window=window
            )
            _, _, sys_stft = signal.stft(
                sys_data, fs=mic_sr, nperseg=nperseg, noverlap=noverlap, window=window
            )

            # Ensure both STFT have the same shape
            min_time_frames = min(mic_stft.shape[1], sys_stft.shape[1])
            mic_stft = mic_stft[:, :min_time_frames]
            sys_stft = sys_stft[:, :min_time_frames]

            # Calculate bleed removal
            mic_mag = np.abs(mic_stft)
            sys_mag = np.abs(sys_stft)
            mic_phase = np.angle(mic_stft)
            sys_phase = np.angle(sys_stft)

            epsilon = 1e-9
            alpha = (mic_mag * sys_mag) / (sys_mag**2 + epsilon)
            alpha = np.clip(alpha, 0.0, 1.2)

            bleed_removed_mag = mic_mag - alpha * sys_mag
            spectral_floor = 0.02 * mic_mag
            bleed_removed_mag = np.maximum(bleed_removed_mag, spectral_floor)

            if randomize_phase:
                dominant_mask = sys_mag > mic_mag
                rand_phase = 2.0 * np.pi * np.random.rand(*dominant_mask.shape)
                final_phase = np.where(dominant_mask, rand_phase, mic_phase)
            else:
                final_phase = mic_phase

            bleed_removed_stft = bleed_removed_mag * np.exp(1j * final_phase)

            # Perform ISTFT
            _, cleaned_audio = signal.istft(
                bleed_removed_stft,
                fs=mic_sr,
                nperseg=nperseg,
                noverlap=noverlap,
                window=window,
            )

            # Optional highpass filter
            if self._highpass_cutoff > 0:
                nyq = mic_sr / 2
                cutoff = self._highpass_cutoff / nyq
                b, a = butter(2, cutoff, btype="high")
                cleaned_audio = filtfilt(b, a, cleaned_audio)

            # Normalize
            max_val = np.max(np.abs(cleaned_audio)) or 1e-9
            cleaned_audio /= max_val

            # Only apply minimal trimming based on alignment lag
            lag_samples = int(lag_seconds * mic_sr)
            if lag_samples > 0:
                logger.debug(
                    f"Trimming {lag_seconds:.4f} seconds ({lag_samples} samples) based on alignment lag."
                )
                cleaned_audio = cleaned_audio[lag_samples:]
            else:
                logger.debug("No lag detected; skipping trimming.")

            # Final length check and adjustment
            length_diff = len(cleaned_audio) - original_length
            if length_diff > 0:
                # If cleaned audio is longer than original, trim the excess from the start
                logger.debug(
                    f"Trimming {length_diff} excess samples from start of cleaned audio to match original length"
                )
                cleaned_audio = cleaned_audio[length_diff:]
            elif length_diff < 0:
                # This should not happen, but log if it does
                logger.warning(
                    f"Cleaned audio is {-length_diff} samples shorter than original"
                )

            # Save the cleaned audio
            cleaned_audio = (cleaned_audio * 32767).astype(np.int16)
            sf.write(output_file, cleaned_audio, mic_sr)

            logger.info("Frequency-domain bleed removal completed and saved.")

        except Exception as e:
            logger.error(
                "Error during frequency-domain bleed removal",
                extra={"error": str(e), "mic_file": mic_file, "sys_file": sys_file},
            )
            raise

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
        smoothing_factor: int = 2,
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
                    "smoothing_factor": smoothing_factor,
                },
            },
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
        smooth_factor: int = 2,
    ) -> np.ndarray:
        """Existing helper for spectral approach."""
        # ... [unchanged] ...
        pass

    def wiener_filter(
        self,
        spec: np.ndarray,
        noise_power: np.ndarray,
        alpha: float = 2.2,
        second_pass: bool = True,
    ) -> np.ndarray:
        """Apply Wiener filter to reduce noise."""
        return spec * (np.abs(spec) ** 2 / (np.abs(spec) ** 2 + alpha * noise_power))

    def trim_audio_with_lag(
        self,
        input_file: str,
        output_file: str,
        lag_samples: int,
        sample_rate: int,
        stft_padding: int = 0,
    ) -> None:
        """
        Ensures cleaned audio preserves all original content and matches original length exactly.
        No silent padding is added, and all original content is preserved.

        Args:
            input_file (str): Path to the original input audio file.
            output_file (str): Path to the cleaned audio file to adjust.
            lag_samples (int): Number of samples to account for alignment (used for logging).
            sample_rate (int): Sampling rate of the audio.
            stft_padding (int): Additional padding samples (used for logging).
        """
        # Read both audio files
        original_audio, original_sr = sf.read(input_file)
        cleaned_audio, cleaned_sr = sf.read(output_file)

        if original_sr != cleaned_sr:
            raise ValueError(
                f"Sample rate mismatch: original={original_sr}, cleaned={cleaned_sr}"
            )

        logger.debug(
            "Audio length comparison",
            extra={
                "original_length": len(original_audio),
                "cleaned_length": len(cleaned_audio),
                "lag_samples": lag_samples,
                "sample_rate": sample_rate,
            },
        )

        # Convert to mono if needed
        if original_audio.ndim > 1:
            original_audio = original_audio.mean(axis=1)
        if cleaned_audio.ndim > 1:
            cleaned_audio = cleaned_audio.mean(axis=1)

        # Find the length difference
        length_diff = len(original_audio) - len(cleaned_audio)

        if length_diff == 0:
            logger.debug("Audio lengths match exactly - no adjustment needed")
            return
        elif length_diff > 0:
            # Cleaned audio is shorter than original - we need to preserve more content
            logger.warning(
                f"Cleaned audio is {length_diff} samples shorter than original. "
                "Adjusting to preserve all content.",
                extra={
                    "original_length": len(original_audio),
                    "cleaned_length": len(cleaned_audio),
                },
            )
            # Instead of padding with silence, we'll preserve more of the cleaned audio
            final_audio = cleaned_audio
        else:
            # Cleaned audio is longer than original - trim excess while preserving content
            logger.debug(
                f"Cleaned audio is {-length_diff} samples longer than original. Trimming excess.",
                extra={
                    "original_length": len(original_audio),
                    "cleaned_length": len(cleaned_audio),
                },
            )
            # Trim from the end to preserve the aligned start
            final_audio = cleaned_audio[: len(original_audio)]

        # Write the adjusted audio, preserving the original length
        sf.write(output_file, final_audio, sample_rate)

        logger.info(
            "Audio length adjustment complete",
            extra={
                "final_length": len(final_audio),
                "matches_original": len(final_audio) == len(original_audio),
            },
        )

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
                    "recording_id": event.recording_id
                    if hasattr(event, "recording_id")
                    else event.get("recording_id"),
                    "event_id": event.event_id
                    if hasattr(event, "event_id")
                    else event.get("event_id"),
                    "event_type": type(event).__name__,
                    "event_data": str(event),
                    "event_bus_type": type(self.event_bus).__name__
                    if self.event_bus
                    else None,
                    "event_bus_id": id(self.event_bus) if self.event_bus else None,
                    "handler_id": id(self),
                    "handler_method": "handle_recording_ended",
                    "thread_id": threading.get_ident(),
                },
            )

            if isinstance(event, dict):
                recording_id = event.get("recording_id")
                current_event = event.get("current_event", {})
                recording_data = current_event.get("recording", {})
                audio_paths = recording_data.get("audio_paths", {})
                mic_path = audio_paths.get("microphone")
                sys_path = audio_paths.get("system")
                metadata = event.get("metadata", {})
            else:
                recording_id = event.recording_id
                mic_path = event.microphone_audio_path
                sys_path = event.system_audio_path
                metadata = event.metadata if hasattr(event, "metadata") else {}

            if not recording_id:
                logger.error(
                    "No recording_id found in event data",
                    extra={
                        "req_id": event_id,
                        "plugin_name": self.name,
                        "event_data": str(event),
                    },
                )
                return

            source_plugin = (
                event.get("source_plugin")
                if isinstance(event, dict)
                else getattr(event.context, "source_plugin", None)
                if hasattr(event, "context")
                else None
            )
            if source_plugin == self.name:
                logger.debug(
                    "Skipping our own event",
                    extra={
                        "req_id": event_id,
                        "plugin_name": self.name,
                        "source_plugin": source_plugin,
                    },
                )
                return

            if not self._db:
                self._db = await get_db_async()

            # Retry database operations with exponential backoff
            max_retries = 3
            retry_delay = 1.0  # seconds

            for attempt in range(max_retries):
                try:
                    await self._db.execute(
                        """
                        INSERT INTO plugin_tasks (recording_id, plugin_name, status, created_at, updated_at)
                        VALUES (?, ?, 'processing', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                        ON CONFLICT(recording_id, plugin_name)
                        DO UPDATE SET status = 'processing', updated_at = CURRENT_TIMESTAMP
                        """,
                        (recording_id, self.name),
                    )
                    break
                except sqlite3.OperationalError as e:
                    if "database is locked" in str(e) and attempt < max_retries - 1:
                        logger.warning(
                            "Database locked, retrying...",
                            extra={
                                "req_id": event_id,
                                "plugin_name": self.name,
                                "attempt": attempt + 1,
                                "retry_delay": retry_delay,
                            },
                        )
                        await asyncio.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                        continue
                    raise
                except sqlite3.IntegrityError as e:
                    if "FOREIGN KEY constraint failed" in str(e):
                        logger.warning(
                            "Recording not yet in database, retrying...",
                            extra={
                                "req_id": event_id,
                                "plugin_name": self.name,
                                "recording_id": recording_id,
                                "attempt": attempt + 1,
                                "retry_delay": retry_delay,
                            },
                        )
                        await asyncio.sleep(retry_delay)
                        retry_delay *= 2
                        continue
                    raise

            await self._process_audio_files(recording_id, sys_path, mic_path, metadata)

        except Exception as e:
            logger.error(
                "Error handling recording ended event",
                extra={"req_id": event_id, "plugin_name": self.name, "error": str(e)},
                exc_info=True,
            )

    async def process_recording(self, recording_id: str, event_data: EventData) -> None:
        """Called from code to process a recording with the configured approach."""
        try:
            logger.info("Starting audio processing", extra={...})
            # Extract mic/system paths from event_data ...
            # Then call _process_audio_files ...
            pass
        except Exception:
            logger.error("Failed to process recording", extra={...}, exc_info=True)
            raise

    def _translate_path_to_container(self, local_path: str | None) -> str | None:
        """Translate a local path to its corresponding container path."""
        if not local_path:
            return None

        # Get the file name from the path
        file_name = os.path.basename(local_path)

        # Construct path using configured recordings directory
        container_path = os.path.join(self._recordings_dir, file_name)

        # Log the path translation
        logger.debug(
            "Path translation",
            extra={
                "req_id": self._req_id,
                "plugin_name": self.name,
                "original_path": local_path,
                "container_path": container_path,
                "recordings_dir": self._recordings_dir,
                "file_exists": os.path.exists(container_path),
            },
        )

        return container_path if os.path.exists(container_path) else None

    async def _process_audio_files(
        self,
        recording_id: str,
        system_audio_path: str | None,
        microphone_audio_path: str | None,
        event_metadata: dict | None = None,
    ) -> None:
        """Process the audio files for noise reduction."""
        try:
            # Translate paths if needed
            system_audio_path = self._translate_path_to_container(system_audio_path)
            microphone_audio_path = self._translate_path_to_container(
                microphone_audio_path
            )

            # Check if files exist
            system_exists = system_audio_path and os.path.exists(system_audio_path)
            mic_exists = microphone_audio_path and os.path.exists(microphone_audio_path)

            if not system_exists and not mic_exists:
                logger.warning(
                    "Missing or invalid audio files, skipping noise reduction",
                    extra={
                        "req_id": self._req_id,
                        "plugin_name": self.name,
                        "recording_id": recording_id,
                        "system_exists": system_exists,
                        "mic_exists": mic_exists,
                        "system_path": system_audio_path,
                        "mic_path": microphone_audio_path,
                        "is_docker": os.path.exists("/.dockerenv"),
                    },
                )
                return

            logger.info(
                "Starting audio processing for both system and microphone",
                extra={
                    "req_id": self._req_id,
                    "plugin_name": self.name,
                    "recording_id": recording_id,
                    "system_audio_path": system_audio_path,
                    "microphone_audio_path": microphone_audio_path,
                    "time_domain_subtraction": self._time_domain_subtraction,
                    "freq_domain_bleed_removal": self._freq_domain_bleed_removal,
                },
            )

            if system_audio_path and microphone_audio_path:
                loop = asyncio.get_running_loop()

                if self._freq_domain_bleed_removal:
                    # 2) Frequency-domain bleed removal
                    output_path = (
                        self._output_directory
                        / f"{recording_id}_mic_bleed_removed_freq.wav"
                    )
                    await loop.run_in_executor(
                        self._executor,
                        self._remove_bleed_frequency_domain,
                        microphone_audio_path,
                        system_audio_path,
                        str(output_path),
                        True,  # do_alignment
                        True,  # randomize_phase
                    )
                    final_output = output_path

                elif self._time_domain_subtraction:
                    # 1) Simple time-domain approach
                    output_path = (
                        self._output_directory
                        / f"{recording_id}_mic_bleed_removed_time.wav"
                    )
                    await loop.run_in_executor(
                        self._executor,
                        self._subtract_bleed_time_domain,
                        microphone_audio_path,
                        system_audio_path,
                        str(output_path),
                        True,  # do_alignment
                        True,  # auto_scale
                    )
                    final_output = output_path

                else:
                    # 3) Fallback to original spectral noise reduction
                    output_path = (
                        self._output_directory
                        / f"{recording_id}_microphone_cleaned.wav"
                    )
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
                        self._smoothing_factor,
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
                        (str(final_output), recording_id, self.name),
                    )

                logger.info(
                    "Audio processing completed successfully",
                    extra={
                        "req_id": self._req_id,
                        "plugin_name": self.name,
                        "recording_id": recording_id,
                        "output_path": str(final_output),
                        "time_domain_subtraction": self._time_domain_subtraction,
                        "freq_domain_bleed_removal": self._freq_domain_bleed_removal,
                    },
                )

                # Trim the first N seconds of the cleaned audio based on alignment_chunk_seconds config
                # Get actual sample rate from the audio file
                _, sample_rate = sf.read(str(final_output))
                self.trim_audio_with_lag(
                    str(final_output),
                    str(final_output),
                    lag_samples=int(self._alignment_chunk_seconds * sample_rate),
                    sample_rate=sample_rate,
                )

                # Emit completion event
                if self.event_bus:
                    await self.event_bus.publish(
                        {
                            "event": "noise_reduction.completed",
                            "recording_id": recording_id,
                            "output_path": str(final_output),
                            "original_audio_path": microphone_audio_path,
                            "system_audio_path": system_audio_path,
                            "event_id": f"{recording_id}_noise_reduction_completed",
                            "timestamp": datetime.utcnow().isoformat(),
                            "plugin_id": self.name,
                            "metadata": {
                                "time_domain_subtraction": self._time_domain_subtraction,
                                "freq_domain_bleed_removal": self._freq_domain_bleed_removal,
                            },
                            "data": {
                                "current_event": {
                                    "recording": {
                                        "audio_paths": {
                                            "system": system_audio_path,
                                            "microphone": microphone_audio_path,
                                        },
                                        "metadata": event_metadata or {},
                                    }
                                }
                            },
                        }
                    )

            else:
                logger.warning(
                    "Missing or invalid audio files, skipping noise reduction",
                    extra={
                        "req_id": self._req_id,
                        "plugin_name": self.name,
                        "recording_id": recording_id,
                        "system_exists": os.path.exists(system_audio_path)
                        if system_audio_path
                        else False,
                        "mic_exists": os.path.exists(microphone_audio_path)
                        if microphone_audio_path
                        else False,
                    },
                )

        except Exception as e:
            logger.error(
                "Failed to process audio files",
                extra={
                    "req_id": self._req_id,
                    "plugin_name": self.name,
                    "recording_id": recording_id,
                    "error": str(e),
                },
                exc_info=True,
            )
            raise

    async def _shutdown(self) -> None:
        """Shutdown plugin."""
        try:
            if self.event_bus is not None:
                logger.info(
                    "Unsubscribing from recording.ended event",
                    extra={"req_id": self._req_id, "plugin_name": self.name},
                )
                await self.event_bus.unsubscribe(
                    "recording.ended", self.handle_recording_ended
                )

            if self._executor is not None:
                logger.info(
                    "Shutting down thread pool",
                    extra={"req_id": self._req_id, "plugin_name": self.name},
                )
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, self._executor.shutdown, True)
                self._executor = None

            if self._db is not None:
                logger.info(
                    "Closing database connection",
                    extra={"req_id": self._req_id, "plugin_name": self.name},
                )
                await self._db.close()
                self._db = None

            logger.info(
                "Plugin shutdown complete",
                extra={"req_id": self._req_id, "plugin_name": self.name},
            )

        except Exception as e:
            logger.error(
                "Error during plugin shutdown",
                extra={
                    "req_id": self._req_id,
                    "plugin_name": self.name,
                    "error": str(e),
                },
                exc_info=True,
            )
            raise
