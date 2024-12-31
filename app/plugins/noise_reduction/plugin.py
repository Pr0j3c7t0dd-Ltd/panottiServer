"""Noise reduction plugin optimized for speech audio."""

import logging
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import NamedTuple

import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, filtfilt, istft, stft

from app.models.database import DatabaseManager
from app.plugins.base import EventType, PluginBase, PluginConfig
from app.plugins.events.bus import EventBus as PluginEventBus

logger = logging.getLogger(__name__)


class AudioPaths(NamedTuple):
    """Container for audio file paths."""

    recording_id: str
    system_audio: str | None
    mic_audio: str | None


class NoiseReductionPlugin(PluginBase):
    """Plugin for reducing noise in speech audio recordings."""

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
        self._stft_window_size = config_dict.get("stft_window_size", 2048)
        self._stft_overlap_percent = config_dict.get("stft_overlap_percent", 75)
        self._noise_reduce_factor = config_dict.get("noise_reduce_factor", 0.3)
        self._wiener_alpha = config_dict.get("wiener_alpha", 0.8)
        self._min_magnitude_threshold = config_dict.get("spectral_floor", 0.1)
        self._noise_smooth_factor = config_dict.get("smoothing_factor", 4)

        # Create output directory
        self._output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Speech-optimized noise reduction plugin initialized",
            extra={
                "plugin_name": self._plugin_name,
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

    def _apply_highpass_filter(
        self, data: np.ndarray, cutoff_freq: float, sample_rate: int, order: int = 3
    ) -> np.ndarray:
        """Apply a Butterworth highpass filter to the audio data."""
        nyquist = sample_rate * 0.5
        normalized_cutoff = cutoff_freq / nyquist
        b, a = butter(order, normalized_cutoff, btype="high", analog=False)
        return filtfilt(b, a, data)

    def _compute_noise_profile(
        self,
        noise_data: np.ndarray,
        sample_rate: int,
        nperseg: int,
        noverlap: int,
        smooth_factor: int,
    ) -> np.ndarray:
        """Compute smoothed noise profile from noise sample."""
        _, _, noise_spec = stft(
            noise_data,
            fs=sample_rate,
            nperseg=nperseg,
            noverlap=noverlap,
            boundary="even",
            window="hann",
        )

        # Compute power spectrum and apply smoothing
        noise_power = np.mean(np.abs(noise_spec) ** 2, axis=1)
        kernel = np.ones(smooth_factor) / smooth_factor
        noise_power = np.convolve(noise_power, kernel, mode="same")

        return noise_power[:, np.newaxis]

    def _wiener_filter(
        self, spec: np.ndarray, noise_power: np.ndarray, alpha: float = 0.8
    ) -> np.ndarray:
        """Apply classical Wiener filter.

        Args:
            spec: Input spectrogram
            noise_power: Noise power spectrum
            alpha: Wiener filter parameter (default: 0.8 for minimal filtering)

        Returns:
            Filtered spectrogram
        """
        # Compute signal power
        sig_power = np.abs(spec) ** 2

        # Classical Wiener filter formula
        wiener_gain = sig_power / (sig_power + alpha * noise_power)

        # Apply minimum gain floor of 0.5 to preserve more speech
        wiener_gain = np.maximum(wiener_gain, 0.5)

        # Apply gain to spectrogram
        return spec * wiener_gain

    def _reduce_noise_worker(
        self,
        mic_file: str,
        noise_file: str,
        output_file: str,
    ) -> None:
        """Worker function for speech-optimized noise reduction processing."""
        import warnings

        from scipy.io.wavfile import WavFileWarning

        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=WavFileWarning)
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

            # Apply gentle highpass filter
            mic_data = self._apply_highpass_filter(
                mic_data,
                self._highpass_cutoff,
                mic_rate,
                order=self._highpass_order,
            )
            noise_data = self._apply_highpass_filter(
                noise_data,
                self._highpass_cutoff,
                noise_rate,
                order=self._highpass_order,
            )

            # STFT parameters optimized for speech
            nperseg = self._stft_window_size
            noverlap = nperseg * self._stft_overlap_percent // 100

            # Speech-optimized noise profile computation
            noise_profile = self._compute_noise_profile(
                noise_data,
                noise_rate,
                nperseg=nperseg,
                noverlap=noverlap,
                smooth_factor=self._noise_smooth_factor,
            )

            # Compute STFT of microphone signal
            _, _, mic_spec = stft(
                mic_data,
                fs=mic_rate,
                nperseg=nperseg,
                noverlap=noverlap,
                boundary="even",
                window="hann",
            )

            # Apply gentler Wiener filter
            filtered_spec = self._wiener_filter(
                mic_spec,
                noise_profile,
                alpha=self._wiener_alpha,
            )

            # Apply speech-preserving spectral floor
            filtered_spec = np.maximum(
                filtered_spec,
                self._min_magnitude_threshold * np.max(np.abs(filtered_spec)),
            )

            # Inverse STFT with same parameters
            _, cleaned_data = istft(
                filtered_spec,
                fs=mic_rate,
                nperseg=nperseg,
                noverlap=noverlap,
                boundary="even",
                window="hann",
            )

            # Normalize while preserving dynamic range
            cleaned_data = cleaned_data / np.max(np.abs(cleaned_data))

            # Apply subtle compression to preserve speech
            cleaned_data = np.sign(cleaned_data) * np.power(np.abs(cleaned_data), 0.9)

            # Convert to int16 with dithering for better speech quality
            cleaned_data = (cleaned_data * 32767).astype(np.float32)
            cleaned_data += np.random.normal(0, 0.5, cleaned_data.shape)
            cleaned_data = np.clip(cleaned_data, -32767, 32767).astype(np.int16)

            # Write output file
            wavfile.write(output_file, mic_rate, cleaned_data)

        except Exception as e:
            logger.error(
                "Error in noise reduction worker",
                extra={
                    "error": str(e),
                    "mic_file": mic_file,
                    "noise_file": noise_file,
                    "output_file": output_file,
                },
                exc_info=True,
            )
            raise

    def _extract_paths(self, event: EventType) -> AudioPaths:
        """Extract audio paths from event.

        Args:
            event: Event containing audio paths

        Returns:
            AudioPaths containing recording_id and audio paths
        """
        recording_id = str(uuid.uuid4())

        if isinstance(event, dict):
            recording_id = event.get("recording_id", recording_id)
            system_audio = event.get("system_audio_path")
            mic_audio = event.get("microphone_audio_path")
        else:
            recording_id = event.recording_id
            system_audio = event.system_audio_path
            mic_audio = event.microphone_audio_path

        return AudioPaths(recording_id, system_audio, mic_audio)

    def _get_input_paths(self, paths: AudioPaths) -> list[str]:
        """Get list of valid input paths.

        Args:
            paths: AudioPaths containing potential input paths

        Returns:
            List of valid file paths as strings
        """
        input_paths: list[str] = []
        if paths.system_audio:
            input_paths.append(str(Path(paths.system_audio)))
        if paths.mic_audio:
            input_paths.append(str(Path(paths.mic_audio)))
        return input_paths

    async def _initialize(self) -> None:
        """Initialize plugin-specific resources and subscribe to events."""
        if self.event_bus is None:
            return

        # Initialize database connection
        self.db = DatabaseManager.get_instance()

        # Subscribe to relevant events
        self.logger.info("Subscribing to recording events")
        await self.subscribe("recording.ended", self._handle_recording_ended)

        self.logger.info(
            "Noise reduction plugin initialized",
            extra={
                "plugin": self._plugin_name,
                "config": {
                    "output_directory": str(self._output_dir),
                    "max_concurrent_tasks": self._thread_pool._max_workers,
                    "highpass_cutoff": self._highpass_cutoff,
                    "highpass_order": self._highpass_order,
                    "stft_window_size": self._stft_window_size,
                },
            },
        )

    async def _shutdown(self) -> None:
        """Clean up plugin resources."""
        self.logger.info("Shutting down noise reduction plugin")

        # Unsubscribe from events
        if self.event_bus is not None:
            await self.unsubscribe("recording.ended", self._handle_recording_ended)

        # Clean up database connection
        if self.db is not None:
            await self.db.close()

        # Shutdown thread pool
        self._thread_pool.shutdown(wait=True)

        self.logger.info("Noise reduction plugin shutdown complete")

    async def _handle_recording_ended(self, event: EventType) -> None:
        """Handle recording ended event by processing audio for noise reduction.

        Args:
            event: Recording ended event containing recording_id and file paths
        """
        # Extract paths from event
        paths = self._extract_paths(event)
        if not paths.recording_id:
            self.logger.error("No recording ID in event")
            return

        # Get input file paths
        input_paths = self._get_input_paths(paths)
        if not input_paths:
            self.logger.error(
                "No valid audio paths found",
                extra={"recording_id": paths.recording_id},
            )
            return

        # Create processing task
        task_id = await self._create_task(paths.recording_id, input_paths)
        if not task_id:
            self.logger.error(
                "Failed to create processing task",
                extra={"recording_id": paths.recording_id},
            )
            return

        # Process files asynchronously
        try:
            futures = []
            for input_path in input_paths:
                output_name = f"{paths.recording_id}_{Path(input_path).name}"
                output_path = str(self._output_dir / output_name)
                future = self._thread_pool.submit(
                    self._reduce_noise_worker,
                    str(input_path),
                    str(input_path),  # Use input file as noise file
                    output_path,
                )
                futures.append((future, output_path))

            # Wait for all processing to complete
            output_paths = []
            for future, output_path in futures:
                try:
                    future.result()  # This blocks until the worker is done
                    output_paths.append(output_path)
                except Exception as e:
                    self.logger.error(
                        "Error processing file",
                        extra={
                            "recording_id": paths.recording_id,
                            "error": str(e),
                            "output_path": output_path,
                        },
                    )

            # Update task status and emit completion event
            if output_paths:
                await self._update_task_status(task_id, "completed", output_paths)
                await self.emit_event(
                    "noise_reduction.completed",
                    {
                        "recording_id": paths.recording_id,
                        "input_paths": [str(p) for p in input_paths],
                        "output_paths": [str(p) for p in output_paths],
                    },
                )
            else:
                await self._update_task_status(
                    task_id,
                    "failed",
                    error_message="No files were successfully processed",
                )

        except Exception as e:
            self.logger.error(
                "Error during noise reduction",
                extra={
                    "recording_id": paths.recording_id,
                    "error": str(e),
                },
            )
            await self._update_task_status(
                task_id,
                "failed",
                error_message=str(e),
            )

    async def _create_task(
        self, recording_id: str, input_paths: list[str]
    ) -> str | None:
        """Create a new processing task in the database.

        Args:
            recording_id: ID of the recording being processed
            input_paths: List of input file paths to process

        Returns:
            Task ID if successful, None otherwise
        """
        if self.db is None:
            return None

        try:
            task_id = str(uuid.uuid4())
            sql = """
                INSERT INTO plugin_tasks (
                    id, plugin_name, recording_id, status, input_paths, created_at
                ) VALUES (?, ?, ?, ?, ?, datetime('now'))
            """
            await self.db.execute(
                sql,
                (
                    task_id,
                    self._plugin_name,
                    recording_id,
                    "processing",
                    ",".join(input_paths),
                ),
            )
            await self.db.commit()
            return task_id
        except Exception as e:
            self.logger.error(
                "Failed to create task",
                extra={
                    "recording_id": recording_id,
                    "error": str(e),
                },
            )
            return None

    async def _update_task_status(
        self,
        task_id: str,
        status: str,
        output_paths: list[str] | None = None,
        error_message: str | None = None,
    ) -> None:
        """Update the status of a processing task in the database.

        Args:
            task_id: ID of the task to update
            status: New status (completed, failed, etc.)
            output_paths: Optional list of processed file paths
            error_message: Optional error message if the task failed
        """
        if self.db is None:
            return

        try:
            await self.db.execute(
                """
                UPDATE plugin_tasks
                SET status = ?,
                    output_paths = ?,
                    error_message = ?,
                    updated_at = datetime('now')
                WHERE id = ?
                """,
                (
                    status,
                    ",".join(output_paths) if output_paths else None,
                    error_message,
                    task_id,
                ),
            )
            await self.db.commit()
        except Exception as e:
            self.logger.error(
                "Failed to update task status",
                extra={
                    "task_id": task_id,
                    "status": status,
                    "error": str(e),
                },
            )
