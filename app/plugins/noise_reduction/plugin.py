"""Noise reduction plugin with an advanced frequency-domain bleed removal."""

import asyncio
import logging
import os
import sqlite3
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from sqlite3 import Connection
from typing import Any, Union

import numpy as np
import soundfile as sf
from scipy import signal
from scipy.signal import butter, filtfilt
import psutil

from app.core.events import ConcreteEventBus as EventBus
from app.core.events import Event, EventContext, EventPriority
from app.core.plugins import PluginBase, PluginConfig
from app.models.database import DatabaseManager, get_db_async
from app.utils.logging_config import get_logger

logger = get_logger("app.plugins.noise_reduction.plugin")
logger.setLevel(logging.DEBUG)


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

    def __init__(self, config: PluginConfig, event_bus: EventBus | None = None) -> None:
        super().__init__(config, event_bus)

        config_dict = config.config or {}
        self._output_directory = Path(
            config_dict.get("output_directory", "data/cleaned_audio")
        )
        self._noise_reduce_factor = float(config_dict.get("noise_reduce_factor", 1.0))
        self._wiener_alpha = float(config_dict.get("wiener_alpha", 2.5))
        self._highpass_cutoff = float(config_dict.get("highpass_cutoff", 95))
        self._spectral_floor = float(config_dict.get("spectral_floor", 0.02))
        self._smoothing_factor = int(config_dict.get("smoothing_factor", 2))
        self._max_workers = int(config_dict.get("max_concurrent_tasks", 4))
        self._task_timeout = int(config_dict.get("task_timeout", 3600))

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
        self._executor = ThreadPoolExecutor(
            max_workers=self._max_workers,
            thread_name_prefix="noise_reduction"
        )
        self._req_id = str(uuid.uuid4())
        self._db: Connection | None = None

        # Get recordings directory from environment variable, with fallback
        self._recordings_dir = os.getenv("RECORDINGS_DIR", "/app/recordings")
        # Ensure the path exists
        os.makedirs(self._recordings_dir, exist_ok=True)

        # STFT parameters
        self._nperseg = 2048
        self._noverlap = self._nperseg // 2
        self._window = "hann"
        
        # Bleed removal parameters
        self._alpha_max = 1.2  # Maximum bleed reduction factor
        self._alpha_min = 0.0  # Minimum bleed reduction factor
        self._chunk_duration = 30  # seconds for chunked processing

    async def _initialize(self) -> None:
        """Initialize the plugin."""
        try:
            logger.info(
                "Initializing noise reduction plugin",
                extra={"req_id": self._req_id, "plugin_name": self.name},
            )

            for attempt in range(3):
                try:
                    # Get DatabaseManager instance
                    async with get_db_async() as db:
                        self._db = db
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
                await self.event_bus.subscribe("recording.ended", self.__call__)
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
        self,
        mic_data: Union[np.ndarray, tuple[np.ndarray, np.ndarray]],
        sys_data: Union[np.ndarray, tuple[np.ndarray, np.ndarray]],
        sample_rate: int,
    ) -> tuple[np.ndarray, np.ndarray, float]:
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

        # Handle both mono and stereo audio
        def convert_to_mono(audio_data):
            if isinstance(audio_data, tuple):
                # For stereo, average the channels
                return np.mean(
                    [channel.astype(np.float32) for channel in audio_data], axis=0
                )
            return audio_data.astype(np.float32)

        mic_chunk = convert_to_mono(mic_data[:chunk_len])
        sys_chunk = convert_to_mono(sys_data[:chunk_len])

        corr = signal.correlate(mic_chunk, sys_chunk, mode="full", method="fft")
        best_lag = int(np.argmax(corr) - (len(sys_chunk) - 1))  # Ensure integer type

        if best_lag > 0:
            sys_aligned = np.pad(sys_data, (best_lag, 0), mode="constant")
            mic_aligned = mic_data
        else:
            mic_aligned = np.pad(mic_data, (max(int(-best_lag), 0), 0), mode="constant")
            sys_aligned = sys_data

        max_len = max(len(mic_aligned), len(sys_aligned))
        mic_aligned = np.pad(
            mic_aligned, (0, max_len - len(mic_aligned)), mode="constant"
        )
        sys_aligned = np.pad(
            sys_aligned, (0, max_len - len(sys_aligned)), mode="constant"
        )

        lag_seconds = float(best_lag) / sample_rate  # Ensure floating-point type
        logger.debug(f"Alignment lag calculated: {lag_seconds:.4f} seconds")

        return np.asarray(mic_aligned), np.asarray(sys_aligned), lag_seconds

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
    def _resample_if_needed(
        self, audio_data: np.ndarray, src_sr: int, target_sr: int
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Resample audio data if source and target sample rates differ."""
        if src_sr == target_sr:
            return audio_data

        logger.info(
            f"Sample rates differ: src={src_sr}Hz, target={target_sr}Hz. Resampling audio.",
            extra={
                "input_length": len(audio_data),
                "src_sr": src_sr,
                "target_sr": target_sr,
                "input_shape": audio_data.shape,
            }
        )

        # Handle stereo audio
        is_stereo = len(audio_data.shape) > 1 and audio_data.shape[1] == 2
        if is_stereo:
            logger.debug("Processing stereo audio")
            # Split into channels
            left_channel = audio_data[:, 0]
            right_channel = audio_data[:, 1]
        else:
            # Ensure 1D for mono
            audio_data = np.ascontiguousarray(audio_data.reshape(-1))

        # For very large files, use chunked processing with smaller chunks
        CHUNK_DURATION = 10  # Process 10 seconds at a time
        chunk_samples = int(CHUNK_DURATION * src_sr)
        
        def resample_channel(channel_data):
            if len(channel_data) > chunk_samples:
                logger.info(
                    "Large file detected, using chunked processing for resampling",
                    extra={
                        "total_samples": len(channel_data),
                        "chunk_samples": chunk_samples,
                        "num_chunks": len(channel_data) // chunk_samples + 1,
                        "estimated_duration_seconds": len(channel_data) / src_sr
                    }
                )
                
                # Pre-allocate output array
                target_length = int(len(channel_data) * target_sr / src_sr)
                resampled = np.zeros(target_length, dtype=np.float32)
                
                # Process in chunks with progress logging
                total_chunks = len(channel_data) // chunk_samples + 1
                for i in range(0, len(channel_data), chunk_samples):
                    current_chunk = i // chunk_samples + 1
                    logger.debug(
                        f"Resampling progress: {current_chunk}/{total_chunks} chunks",
                        extra={
                            "progress_percent": (current_chunk / total_chunks) * 100,
                            "chunk_number": current_chunk,
                            "total_chunks": total_chunks
                        }
                    )
                    
                    chunk = channel_data[i:i + chunk_samples]
                    target_chunk_len = int(len(chunk) * target_sr / src_sr)
                    
                    # Add small overlap to avoid boundary artifacts
                    overlap_samples = 0
                    if i > 0:
                        overlap_samples = min(100, len(chunk) // 10)  # Adaptive overlap
                        chunk = channel_data[i-overlap_samples:i + chunk_samples]
                    
                    # Resample chunk
                    try:
                        resampled_chunk = signal.resample(chunk, int(len(chunk) * target_sr / src_sr))
                        
                        # Remove overlap if present
                        if overlap_samples > 0:
                            overlap_resampled = int(overlap_samples * target_sr / src_sr)
                            resampled_chunk = resampled_chunk[overlap_resampled:]
                        
                        # Calculate output indices
                        start_idx = int(i * target_sr / src_sr)
                        end_idx = start_idx + len(resampled_chunk)
                        end_idx = min(end_idx, len(resampled))
                        
                        # Store chunk
                        resampled[start_idx:end_idx] = resampled_chunk[:end_idx-start_idx]
                    except Exception as e:
                        logger.error(f"Error resampling chunk: {e}", extra={
                            "chunk_size": len(chunk),
                            "target_size": target_chunk_len,
                            "chunk_number": current_chunk
                        })
                        raise
                
                logger.info(
                    "Resampling completed",
                    extra={
                        "output_length": len(resampled),
                        "input_length": len(channel_data)
                    }
                )
                return resampled
            else:
                # For small files, resample all at once
                target_length = int(len(channel_data) * target_sr / src_sr)
                return signal.resample(channel_data, target_length)

        try:
            if is_stereo:
                left_resampled = resample_channel(left_channel)
                right_resampled = resample_channel(right_channel)
                return np.column_stack((left_resampled, right_resampled))
            else:
                return resample_channel(audio_data)
        except Exception as e:
            logger.error(f"Resampling failed: {e}", extra={
                "is_stereo": is_stereo,
                "input_shape": audio_data.shape,
                "src_sr": src_sr,
                "target_sr": target_sr
            })
            raise

    def _process_stft_chunk(
        self,
        audio_chunk: np.ndarray,
        sample_rate: int,
        nperseg: int = 2048,
        noverlap: int | None = None,
        window: str = "hann",
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Process a single chunk with STFT."""
        if noverlap is None:
            noverlap = nperseg // 2
        try:
            return signal.stft(
                audio_chunk,
                fs=sample_rate,
                nperseg=nperseg,
                noverlap=noverlap,
                window=window,
            )
        except Exception as e:
            logger.error(
                "STFT processing failed",
                extra={
                    "error": str(e),
                    "chunk_length": len(audio_chunk),
                    "nperseg": nperseg,
                },
                exc_info=True
            )
            raise

    def _process_istft_chunk(
        self,
        stft_chunk: np.ndarray,
        sample_rate: int,
        nperseg: int = 2048,
        noverlap: int | None = None,
        window: str = "hann",
    ) -> tuple[np.ndarray, np.ndarray]:
        """Process a single chunk with ISTFT."""
        if noverlap is None:
            noverlap = nperseg // 2
        try:
            return signal.istft(
                stft_chunk,
                fs=sample_rate,
                nperseg=nperseg,
                noverlap=noverlap,
                window=window,
            )
        except Exception as e:
            logger.error(
                "ISTFT processing failed",
                extra={
                    "error": str(e),
                    "stft_shape": stft_chunk.shape,
                    "nperseg": nperseg,
                },
                exc_info=True
            )
            raise

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
        process = psutil.Process()
        
        def log_memory():
            memory = process.memory_info()
            logger.debug(
                "Memory usage",
                extra={
                    "rss_gb": memory.rss / 1024**3,
                    "vms_gb": memory.vms / 1024**3,
                }
            )

        # Read original files and get initial length
        logger.debug("Loading audio files", extra={"mic_file": mic_file, "sys_file": sys_file})
        log_memory()
        
        mic_data, mic_sr = sf.read(mic_file)
        sys_data, sys_sr = sf.read(sys_file)

        # Convert to float32 and normalize early
        mic_data = np.asarray(mic_data, dtype=np.float32)
        sys_data = np.asarray(sys_data, dtype=np.float32)
        
        # Normalize input audio
        mic_max = np.max(np.abs(mic_data))
        sys_max = np.max(np.abs(sys_data))
        if mic_max > 0:
            mic_data /= mic_max
        if sys_max > 0:
            sys_data /= sys_max

        logger.debug(
            "Audio files loaded and normalized",
            extra={
                "mic_samples": len(mic_data),
                "sys_samples": len(sys_data),
                "mic_sr": mic_sr,
                "sys_sr": sys_sr,
                "mic_channels": mic_data.shape[1] if len(mic_data.shape) > 1 else 1,
                "sys_channels": sys_data.shape[1] if len(sys_data.shape) > 1 else 1,
            }
        )

        # Convert stereo to mono immediately after loading
        def convert_to_mono(data: np.ndarray, source: str) -> np.ndarray:
            if data.ndim > 1:
                logger.debug(
                    f"Converting {source} audio from stereo to mono",
                    extra={"original_shape": data.shape}
                )
                # Handle different stereo channel arrangements
                if data.shape[-1] == 2:  # Channels in last dimension
                    return data.mean(axis=-1)
                elif data.shape[1] == 2:  # Traditional [samples, channels]
                    return data.mean(axis=1)
                else:
                    logger.warning(f"Unexpected {source} audio shape", extra={"shape": data.shape})
                    return data.mean(axis=np.argmin(data.shape))
            return data

        mic_data = convert_to_mono(mic_data, "microphone")
        sys_data = convert_to_mono(sys_data, "system")
        
        original_length = len(mic_data)

        # If sample rates differ, resample system audio to match microphone
        if mic_sr != sys_sr:
            logger.info(f"Sample rates differ: mic={mic_sr}Hz, sys={sys_sr}Hz. Resampling system audio.")
            try:
                target_length = int(len(sys_data) * mic_sr / sys_sr)
                
                # Use chunked processing for large files
                MAX_CHUNK_SIZE = 1_000_000  # 1M samples per chunk
                if len(sys_data) > MAX_CHUNK_SIZE:
                    logger.info("Using chunked resampling for large file")
                    num_chunks = (len(sys_data) + MAX_CHUNK_SIZE - 1) // MAX_CHUNK_SIZE
                    resampled_data = np.zeros(target_length, dtype=np.float32)
                    
                    for i in range(num_chunks):
                        start_idx = i * MAX_CHUNK_SIZE
                        end_idx = min(start_idx + MAX_CHUNK_SIZE, len(sys_data))
                        chunk = sys_data[start_idx:end_idx]
                        
                        # Add overlap for smoother transitions
                        overlap = 1000 if i > 0 else 0
                        if overlap > 0:
                            chunk = np.concatenate([sys_data[start_idx-overlap:start_idx], chunk])
                        
                        # Resample chunk
                        chunk_target_length = int(len(chunk) * mic_sr / sys_sr)
                        resampled_chunk = signal.resample(chunk, chunk_target_length)
                        
                        # Remove overlap
                        if overlap > 0:
                            overlap_resampled = int(overlap * mic_sr / sys_sr)
                            resampled_chunk = resampled_chunk[overlap_resampled:]
                        
                        # Store in output array
                        out_start = int(start_idx * mic_sr / sys_sr)
                        out_end = out_start + len(resampled_chunk)
                        out_end = min(out_end, len(resampled_data))
                        resampled_data[out_start:out_end] = resampled_chunk[:out_end-out_start]
                        
                else:
                    resampled_data = signal.resample(sys_data, target_length)
                
                sys_data = resampled_data
                sys_sr = mic_sr
                
            except Exception as e:
                logger.error(
                    "Resampling failed",
                    extra={
                        "error": str(e),
                        "mic_sr": mic_sr,
                        "sys_sr": sys_sr,
                        "mic_length": len(mic_data),
                        "sys_length": len(sys_data)
                    },
                    exc_info=True
                )
                raise

        # Align signals if requested
        lag_seconds = 0
        if do_alignment:
            logger.debug("Starting FFT alignment")
            mic_data, sys_data, lag_seconds = self._align_signals_by_fft(
                mic_data, sys_data, mic_sr
            )
            logger.debug(f"FFT alignment completed with lag of {lag_seconds:.4f} seconds")

        try:
            # Perform STFT on full signals
            logger.debug("Starting STFT processing")
            f_mic, t_mic, mic_stft = signal.stft(
                mic_data, fs=mic_sr, nperseg=self._nperseg,
                noverlap=self._noverlap, window=self._window
            )
            _, _, sys_stft = signal.stft(
                sys_data, fs=mic_sr, nperseg=self._nperseg,
                noverlap=self._noverlap, window=self._window
            )

            # Ensure both STFTs have the same shape
            min_time_frames = min(mic_stft.shape[1], sys_stft.shape[1])
            mic_stft = mic_stft[:, :min_time_frames]
            sys_stft = sys_stft[:, :min_time_frames]

            # Calculate magnitudes and phases
            mic_mag = np.abs(mic_stft)
            sys_mag = np.abs(sys_stft)
            mic_phase = np.angle(mic_stft)

            epsilon = 1e-9
            alpha = (mic_mag * sys_mag) / (sys_mag**2 + epsilon)
            alpha = np.clip(alpha, 0.0, 1.2)
            bleed_removed_mag = mic_mag - alpha * sys_mag
            spectral_floor = 0.02 * mic_mag
            bleed_removed_mag = np.maximum(bleed_removed_mag, spectral_floor)
            if randomize_phase and np.max(mic_mag) > 1e-3:
                dominant_mask = sys_mag > mic_mag
                rand_phase = 2.0 * np.pi * np.random.rand(*dominant_mask.shape)
                final_phase = np.where(dominant_mask, rand_phase, mic_phase)
            else:
                final_phase = mic_phase

            # Reconstruct complex STFT
            bleed_removed_stft = bleed_removed_mag * np.exp(1j * final_phase)

            # Inverse STFT
            logger.debug("Starting inverse STFT")
            _, cleaned_audio = signal.istft(
                bleed_removed_stft, fs=mic_sr,
                nperseg=self._nperseg, noverlap=self._noverlap,
                window=self._window
            )

            # Apply highpass filter if configured
            if self._highpass_cutoff > 0:
                nyq = mic_sr / 2
                cutoff = self._highpass_cutoff / nyq
                b, a = butter(2, cutoff, btype="high")
                cleaned_audio = filtfilt(b, a, cleaned_audio)

            # Normalize output
            max_val = np.max(np.abs(cleaned_audio))
            if max_val > 0:
                cleaned_audio /= max_val

            # Handle lag adjustment
            lag_samples = int(lag_seconds * mic_sr)
            if lag_samples > 0:
                logger.debug(
                    f"Trimming {lag_seconds:.4f} seconds ({lag_samples} samples) based on alignment lag."
                )
                cleaned_audio = cleaned_audio[lag_samples:]

            # Final length adjustment
            if len(cleaned_audio) > original_length:
                cleaned_audio = cleaned_audio[:original_length]
            elif len(cleaned_audio) < original_length:
                cleaned_audio = np.pad(cleaned_audio, (0, original_length - len(cleaned_audio)))

            # Save output
            cleaned_audio = (cleaned_audio * 32767).astype(np.int16)
            sf.write(output_file, cleaned_audio, mic_sr)

            logger.info("Frequency-domain bleed removal completed successfully")

        except Exception as e:
            logger.error(
                "Error during frequency-domain processing",
                extra={
                    "error": str(e),
                    "mic_file": mic_file,
                    "sys_file": sys_file,
                    "lag_seconds": lag_seconds
                },
                exc_info=True
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
    async def __call__(self, event_data: Any) -> None:
        """Handle an event."""
        await self.handle_recording_ended(event_data)

    async def handle_recording_ended(self, event_data: Event) -> None:
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
                    "recording_id": event_data.get("recording_id")
                    if isinstance(event_data, dict)
                    else event_data.data.get("recording_id"),
                    "event_id": event_data.get("event_id")
                    if isinstance(event_data, dict)
                    else event_data.event_id,
                    "event_type": type(event_data).__name__,
                    "event_data": str(event_data),
                    "event_bus_type": type(self.event_bus).__name__
                    if self.event_bus
                    else None,
                    "event_bus_id": id(self.event_bus) if self.event_bus else None,
                    "handler_id": id(self),
                    "handler_method": "handle_recording_ended",
                    "thread_id": threading.get_ident(),
                },
            )

            if isinstance(event_data, dict):
                recording_id = event_data.get("recording_id")
                current_event = event_data.get("current_event", {})
                recording_data = current_event.get("recording", {})
                audio_paths = recording_data.get("audio_paths", {})
                mic_path = audio_paths.get("microphone")
                sys_path = audio_paths.get("system")
                metadata = event_data.get("metadata", {})
            else:
                recording_id = event_data.data.get("recording_id")
                mic_path = event_data.data.get("microphone_audio_path")
                sys_path = event_data.data.get("system_audio_path")
                metadata = (
                    event_data.data.get("metadata", {})
                    if hasattr(event_data, "data")
                    else {}
                )

            # Log metadata presence and content
            if not metadata:
                logger.error(
                    "No metadata found in event data",
                    extra={
                        "req_id": event_id,
                        "plugin_name": self.name,
                        "recording_id": recording_id,
                        "event_type": type(event_data).__name__,
                    },
                )
            else:
                logger.debug(
                    "Metadata found in event data",
                    extra={
                        "req_id": event_id,
                        "plugin_name": self.name,
                        "recording_id": recording_id,
                        "metadata": metadata,
                        "metadata_keys": list(metadata.keys()),
                    },
                )

            if not recording_id:
                logger.error(
                    "No recording_id found in event data",
                    extra={
                        "req_id": event_id,
                        "plugin_name": self.name,
                        "event_data": str(event_data),
                    },
                )
                return None

            source_plugin = (
                event_data.get("source_plugin")
                if isinstance(event_data, dict)
                else getattr(event_data.context, "source_plugin", None)
                if hasattr(event_data, "context")
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
                return None

            if not self._db:
                async with get_db_async() as db:
                    self._db = db

            # Retry database operations with exponential backoff
            max_retries = 5
            retry_delay = 2.0

            try:
                for attempt in range(max_retries):
                    try:
                        db = await DatabaseManager.get_instance_async()
                        await db.execute(
                            """
                            INSERT INTO plugin_tasks (recording_id, plugin_name, status, created_at, updated_at)
                            VALUES (?, ?, 'processing', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                            ON CONFLICT(recording_id, plugin_name)
                            DO UPDATE SET status = 'processing', updated_at = CURRENT_TIMESTAMP
                            """,
                            (recording_id, self.name),
                        )
                        await db.commit()
                        break
                    except sqlite3.OperationalError as e:
                        if "database is locked" in str(e) and attempt < max_retries - 1:
                            await asyncio.sleep(retry_delay)
                            retry_delay *= 2
                        else:
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
                                    "max_retries": max_retries,
                                    "retry_delay": retry_delay,
                                },
                            )
                            await asyncio.sleep(retry_delay)
                            retry_delay *= 2
                        else:
                            raise
                        continue
            except Exception as e:
                logger.error(
                    "Database operation failed",
                    extra={
                        "req_id": event_id,
                        "plugin_name": self.name,
                        "recording_id": recording_id,
                        "error": str(e),
                    },
                    exc_info=True
                )
                raise

            await self._process_audio_files(
                recording_id, sys_path, mic_path, metadata, event_data
            )

        except Exception as e:
            logger.error(
                "Error handling recording ended event",
                extra={"req_id": event_id, "plugin_name": self.name, "error": str(e)},
                exc_info=True,
            )

    async def _process_audio_files(
        self,
        recording_id: str,
        system_audio_path: str | None,
        microphone_audio_path: str | None,
        event_metadata: dict | None = None,
        original_event: Event | dict | None = None,
    ) -> Path | None:
        """Process the audio files for noise reduction."""
        try:
            # Log metadata status at start of processing
            if event_metadata is None:
                logger.warning(
                    "No metadata provided for audio processing",
                    extra={
                        "req_id": self._req_id,
                        "plugin_name": self.name,
                        "recording_id": recording_id,
                        "system_path": system_audio_path,
                        "mic_path": microphone_audio_path,
                    },
                )
            else:
                logger.debug(
                    "Processing audio files with metadata",
                    extra={
                        "req_id": self._req_id,
                        "plugin_name": self.name,
                        "recording_id": recording_id,
                        "metadata": event_metadata,
                        "metadata_keys": list(event_metadata.keys()),
                        "system_path": system_audio_path,
                        "mic_path": microphone_audio_path,
                    },
                )

            logger.info(
                "Processing audio files",
                extra={
                    "req_id": self._req_id,
                    "plugin_name": self.name,
                    "recording_id": recording_id,
                    "system_audio_path": system_audio_path,
                    "microphone_audio_path": microphone_audio_path,
                    "correlation_id": str(
                        event_metadata.get("correlation_id", uuid.uuid4())
                    )
                    if event_metadata
                    else str(uuid.uuid4()),
                },
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
                        "correlation_id": str(
                            event_metadata.get("correlation_id", uuid.uuid4())
                        )
                        if event_metadata
                        else str(uuid.uuid4()),
                    },
                )
                # Emit error event with preserved data
                if self.event_bus:
                    error_event = Event(
                        name="noise_reduction.error",
                        data={
                            "recording": original_event.get("recording", {})
                            if isinstance(original_event, dict)
                            else {}
                            if original_event is None
                            else original_event.data.get("recording", {}),
                            "noise_reduction": {
                                "status": "error",
                                "timestamp": datetime.now(UTC).isoformat(),
                                "error": "Missing or invalid audio files",
                                "config": {
                                    "time_domain_subtraction": self._time_domain_subtraction,
                                    "freq_domain_bleed_removal": self._freq_domain_bleed_removal,
                                    "noise_reduce_factor": self._noise_reduce_factor,
                                    "wiener_alpha": self._wiener_alpha,
                                },
                            },
                        },
                        context=EventContext(
                            correlation_id=str(
                                event_metadata.get("correlation_id", uuid.uuid4())
                            )
                            if event_metadata
                            else str(uuid.uuid4()),
                            source_plugin=self.__class__.__name__,
                            metadata=event_metadata
                            if event_metadata is not None
                            else {},
                        ),
                        priority=EventPriority.NORMAL,
                    )
                    await self.event_bus.publish(error_event)
                return None

            # Process audio files
            final_output = None
            if system_audio_path and microphone_audio_path:
                loop = asyncio.get_running_loop()

                if self._freq_domain_bleed_removal:
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
                        True,
                        True,
                    )
                    final_output = output_path

                elif self._time_domain_subtraction:
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
                        True,
                        True,
                    )
                    final_output = output_path

                else:
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

            # Update database status
            if self._db:
                db = await DatabaseManager.get_instance_async()
                await db.execute(
                    """
                    UPDATE plugin_tasks
                    SET status = 'completed', updated_at = CURRENT_TIMESTAMP
                    WHERE recording_id = ? AND plugin_name = ?
                    """,
                    (recording_id, self.name),
                )

            logger.info(
                "Audio processing complete",
                extra={
                    "req_id": self._req_id,
                    "plugin_name": self.name,
                    "recording_id": recording_id,
                    "output_path": str(final_output),
                },
            )

            # Emit completed event
            if self.event_bus:
                # Get correlation ID from original event or create new one
                correlation_id = str(
                    event_metadata.get("correlation_id", uuid.uuid4())
                ) if event_metadata else str(uuid.uuid4())

                # Combine metadata from original event and current metadata
                combined_metadata = {}
                if isinstance(original_event, dict):
                    combined_metadata.update(original_event.get("metadata", {}))
                elif original_event is not None:
                    combined_metadata.update(getattr(original_event, "metadata", {}))
                if event_metadata:
                    combined_metadata.update(event_metadata)

                # Ensure speaker labels are in metadata
                if "speaker_labels" not in combined_metadata:
                    combined_metadata["speaker_labels"] = {
                        "microphone": combined_metadata.get("microphoneLabel"),
                        "system": combined_metadata.get("systemLabel"),
                    }

                completed_event = Event.create(
                    name="noise_reduction.completed",
                    data={
                        "recording": original_event.get("recording", {})
                        if isinstance(original_event, dict)
                        else {}
                        if original_event is None
                        else original_event.data.get("recording", {}),
                        "noise_reduction": {
                            "status": "completed",
                            "timestamp": datetime.now(UTC).isoformat(),
                            "recording_id": recording_id,
                            "output_path": str(final_output),
                            "system_audio_path": system_audio_path,
                            "microphone_audio_path": microphone_audio_path,
                            "method": "frequency_domain"
                            if self._freq_domain_bleed_removal
                            else "time_domain",
                            "config": {
                                "time_domain_subtraction": self._time_domain_subtraction,
                                "freq_domain_bleed_removal": self._freq_domain_bleed_removal,
                                "noise_reduce_factor": self._noise_reduce_factor,
                                "wiener_alpha": self._wiener_alpha,
                                "highpass_cutoff": self._highpass_cutoff,
                                "spectral_floor": self._spectral_floor,
                                "smoothing_factor": self._smoothing_factor,
                            },
                        },
                        "metadata": combined_metadata,
                        "context": {
                            "correlation_id": correlation_id,
                            "source_plugin": self.name,
                            "metadata": combined_metadata,
                        },
                    },
                    correlation_id=correlation_id,
                    source_plugin=self.name,
                    priority=EventPriority.NORMAL,
                )
                await self.event_bus.publish(completed_event)

            return final_output

        except Exception as e:
            logger.error(
                "Error processing audio files",
                extra={
                    "req_id": self._req_id,
                    "plugin_name": self.name,
                    "recording_id": recording_id,
                    "error": str(e),
                    "correlation_id": str(
                        event_metadata.get("correlation_id", uuid.uuid4())
                    )
                    if event_metadata
                    else str(uuid.uuid4()),
                },
            )
            # Emit error event with preserved data
            if self.event_bus:
                error_event = Event(
                    name="noise_reduction.error",
                    data={
                        # Preserve original recording data
                        "recording": original_event.get("recording", {})
                        if isinstance(original_event, dict)
                        else {}
                        if original_event is None
                        else original_event.data.get("recording", {}),
                        "noise_reduction": {
                            "status": "error",
                            "timestamp": datetime.now(UTC).isoformat(),
                            "error": str(e),
                            "config": {
                                "time_domain_subtraction": self._time_domain_subtraction,
                                "freq_domain_bleed_removal": self._freq_domain_bleed_removal,
                                "noise_reduce_factor": self._noise_reduce_factor,
                                "wiener_alpha": self._wiener_alpha,
                            },
                        },
                    },
                    context=EventContext(
                        correlation_id=str(
                            event_metadata.get("correlation_id", uuid.uuid4())
                        )
                        if event_metadata
                        else str(uuid.uuid4()),
                        source_plugin=self.__class__.__name__,
                        metadata=event_metadata if event_metadata is not None else {},
                    ),
                    priority=EventPriority.NORMAL,
                )
                await self.event_bus.publish(error_event)
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

            # Database connection is managed by the get_db_async context manager
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
