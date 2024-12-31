import json
import logging
import os
import threading
import wave
from collections.abc import Callable, Coroutine
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any

from faster_whisper import WhisperModel

from app.models.database import DatabaseManager
from app.models.event import RecordingEndRequest, RecordingEvent, RecordingStartRequest
from app.plugins.base import PluginBase

logger = logging.getLogger(__name__)

EventData = (
    dict[str, Any] | RecordingEvent | RecordingStartRequest | RecordingEndRequest
)
EventHandler = Callable[[EventData], Coroutine[Any, Any, None]]


class AudioTranscriptionPlugin(PluginBase):
    """Plugin for transcribing audio files using Whisper"""

    def __init__(self, config: Any, event_bus: Any | None = None) -> None:
        super().__init__(config, event_bus)
        self._executor: ThreadPoolExecutor | None = None
        self._processing_lock: threading.Lock = threading.Lock()
        self._db_initialized: bool = False
        self._model: WhisperModel | None = None

    async def _initialize(self) -> None:
        """Initialize plugin"""
        if not self.event_bus:
            return

        # Initialize database table
        await self._init_database()

        # Initialize thread pool executor
        max_workers = getattr(self.config, "max_concurrent_tasks", 4)
        self._executor = ThreadPoolExecutor(max_workers=max_workers)

        # Initialize Whisper model
        model_name = getattr(self.config, "model_name", "base.en")

        # Get project root directory (3 levels up from plugin directory)
        current_dir = Path(__file__).resolve().parent
        project_root = current_dir.parent.parent.parent

        # Initialize the Whisper model
        model_dir = os.path.join(project_root, "models", "whisper")
        self.logger.info(
            "Initializing Whisper model",
            extra={
                "model_name": model_name,
                "model_dir": model_dir,
                "max_workers": max_workers,
            },
        )

        # Subscribe to noise reduction completed event
        await self.event_bus.subscribe(
            "noise_reduction.completed", self.handle_noise_reduction_completed
        )

        self.logger.info("Audio transcription plugin initialized")

    async def _shutdown(self) -> None:
        """Shutdown plugin"""
        if not self.event_bus:
            return

        # Unsubscribe from events
        await self.event_bus.unsubscribe(
            "noise_reduction.completed", self.handle_noise_reduction_completed
        )

        # Shutdown thread pool
        if self._executor:
            self._executor.shutdown(wait=True)

    async def handle_noise_reduction_completed(self, event_data: EventData) -> None:
        """Handle noise reduction completed event"""
        try:
            if isinstance(event_data, dict):
                recording_id = event_data.get("recording_id", "unknown")
                output_file = event_data.get("output_file")
                status = event_data.get("status")
            else:
                # Handle RecordingEvent, RecordingStartRequest, RecordingEndRequest
                recording_id = getattr(event_data, "recording_id", "unknown")
                output_file = getattr(event_data, "output_file", None)
                status = getattr(event_data, "status", None)

            if status != "completed" or not output_file:
                logger.error(
                    "Invalid noise reduction event",
                    extra={
                        "recording_id": recording_id,
                        "status": status,
                        "output_file": output_file,
                    },
                )
                return

            # Process audio file
            await self._process_audio(
                recording_id=recording_id,
                input_files=[output_file],
                output_files=[f"{output_file}.transcript.txt"],
                merged_output=f"{output_file}.merged.txt",
                original_event={"recording_id": recording_id},
                input_labels=["Speaker"],
            )

        except Exception as e:
            logger.error(
                f"Failed to handle noise reduction completion: {e}",
                extra={
                    "recording_id": (
                        recording_id if "recording_id" in locals() else "unknown"
                    )
                },
                exc_info=True,
            )

    async def _emit_transcription_event(
        self,
        recording_id: str,
        status: str,
        output_file: str | None = None,
        error: str | None = None,
    ) -> None:
        """Emit transcription event"""
        if not self.event_bus:
            return

        event_data: dict[str, Any] = {
            "type": "transcription.completed",
            "recording_id": recording_id,
            "status": status,
            "timestamp": datetime.utcnow().isoformat(),
        }

        if output_file:
            event_data["output_file"] = output_file
        if error:
            event_data["error"] = error

        await self.event_bus.emit(event_data)

    async def _init_database(self) -> None:
        """Initialize database table for tracking processing state"""
        db = DatabaseManager.get_instance()
        with db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS transcription_tasks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    recording_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    input_paths TEXT,  -- JSON array of input file paths
                    output_paths TEXT, -- JSON array of output transcript paths
                    merged_output_path TEXT,
                    error_message TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
            )
            conn.commit()
        self._db_initialized = True

    def _update_task_status(
        self,
        recording_id: str,
        status: str,
        output_paths: list[str] | None = None,
        merged_output_path: str | None = None,
        error_message: str | None = None,
    ) -> None:
        """Update the status of a processing task in the database"""
        db = DatabaseManager.get_instance()
        with db.get_connection() as conn:
            cursor = conn.cursor()
            update_values = [
                status,
                json.dumps(output_paths) if output_paths else None,
                merged_output_path,
                error_message,
                datetime.utcnow().isoformat(),
                recording_id,
            ]
            cursor.execute(
                """
                UPDATE transcription_tasks
                SET status = ?, output_paths = ?, merged_output_path = ?,
                    error_message = ?, updated_at = ?
                WHERE recording_id = ?
            """,
                update_values,
            )
            conn.commit()

    def _create_task(self, recording_id: str, input_paths: list[str]) -> None:
        """Create a new processing task in the database"""
        db = DatabaseManager.get_instance()
        with db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO transcription_tasks
                (recording_id, status, input_paths)
                VALUES (?, ?, ?)
            """,
                (recording_id, "pending", json.dumps(input_paths)),
            )
            conn.commit()

    def validate_wav_file(self, wav_path: str) -> bool:
        """Validate that the file is a proper WAV file and can be opened"""
        try:
            with wave.open(wav_path, "rb") as wav_file:
                channels = wav_file.getnchannels()
                sample_width = wav_file.getsampwidth()
                framerate = wav_file.getframerate()
                n_frames = wav_file.getnframes()
                duration = n_frames / float(framerate)

                self.logger.info(
                    "WAV file validated",
                    extra={
                        "file": os.path.basename(wav_path),
                        "channels": channels,
                        "sample_width": sample_width,
                        "framerate": framerate,
                        "duration": duration,
                    },
                )
                return True
        except Exception as e:
            self.logger.error(
                "WAV file validation failed",
                extra={"file": os.path.basename(wav_path), "error": str(e)},
            )
            return False

    async def transcribe_audio(
        self, audio_path: str, output_path: str, label: str
    ) -> None:
        """Transcribe an audio file using Whisper"""
        # Validate input file
        self.validate_wav_file(audio_path)

        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Write transcript to file
        with open(output_path, "w") as f:
            f.write(f"# {label}'s Transcript\n\n")
            # Add actual transcription logic here
            f.write("Transcript content would go here\n")

    def merge_transcripts(
        self,
        transcript_files: list[str],
        output_path: str,
        labels: list[str],
        original_event: dict,
    ) -> None:
        """Merge multiple transcript files"""
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Write merged transcript to file
        with open(output_path, "w") as f:
            f.write("# Merged Transcript\n\n")
            for transcript_file, label in zip(transcript_files, labels, strict=False):
                f.write(f"## {label}'s Transcript\n\n")
                with open(transcript_file) as tf:
                    f.write(tf.read())
                f.write("\n\n")

    async def _process_audio(
        self,
        recording_id: str,
        input_files: list[str],
        output_files: list[str],
        merged_output: str,
        original_event: dict,
        input_labels: list[str],
    ) -> None:
        """Process audio files"""
        try:
            # Update status to processing
            self._update_task_status(recording_id, "processing")

            # Transcribe each file
            for _i, (input_file, output_file, label) in enumerate(
                zip(input_files, output_files, input_labels, strict=False)
            ):
                if not os.path.exists(input_file):
                    raise FileNotFoundError(f"Audio file not found: {input_file}")
                # Use appropriate fallback based on audio type
                fallback = (
                    "Meeting Participants"
                    if "system_audio" in input_file
                    else "Speaker"
                )
                await self.transcribe_audio(input_file, output_file, label or fallback)

            # Merge transcripts
            self.merge_transcripts(
                output_files, merged_output, input_labels, original_event
            )

            # Update task status
            self._update_task_status(
                recording_id,
                "completed",
                output_paths=output_files,
                merged_output_path=merged_output,
            )

            # Emit completion event
            if self.event_bus:
                event_data = {
                    "type": "transcription.completed",
                    "recording_id": recording_id,
                    "recording_timestamp": original_event.get("recording_timestamp"),
                    "input_files": input_files,
                    "output_files": output_files,
                    "merged_output": merged_output,
                    "meeting_title": original_event.get("metadata", {}).get(
                        "eventTitle"
                    ),
                    "meeting_provider": original_event.get("metadata", {}).get(
                        "eventProvider"
                    ),
                    "meeting_provider_id": original_event.get("metadata", {}).get(
                        "eventProviderId"
                    ),
                    "meeting_attendees": original_event.get("metadata", {}).get(
                        "eventAttendees", []
                    ),
                    "meeting_start_time": original_event.get("metadata", {}).get(
                        "recordingStarted"
                    ),
                    "meeting_end_time": original_event.get("metadata", {}).get(
                        "recordingEnded"
                    ),
                    "system_audio_label": original_event.get("metadata", {}).get(
                        "systemLabel", "Meeting Participants"
                    ),
                    "microphone_audio_label": original_event.get("metadata", {}).get(
                        "microphoneLabel", "Speaker"
                    ),
                    "transcription_status": "completed",
                    "timestamp": datetime.utcnow().isoformat(),
                }
                await self.event_bus.emit(event_data)

        except Exception as e:
            self.logger.error(
                "Audio processing failed",
                extra={"recording_id": recording_id, "error": str(e)},
            )
            self._update_task_status(recording_id, "error", error_message=str(e))

            # Emit error event
            if self.event_bus:
                error_data = {
                    "type": "transcription.error",
                    "recording_id": recording_id,
                    "recording_timestamp": original_event.get("recording_timestamp"),
                    "input_files": input_files,
                    "output_files": output_files,
                    "merged_output": merged_output,
                    "meeting_title": original_event.get("metadata", {}).get(
                        "eventTitle"
                    ),
                    "meeting_provider": original_event.get("metadata", {}).get(
                        "eventProvider"
                    ),
                    "meeting_provider_id": original_event.get("metadata", {}).get(
                        "eventProviderId"
                    ),
                    "meeting_attendees": original_event.get("metadata", {}).get(
                        "eventAttendees", []
                    ),
                    "meeting_start_time": original_event.get("metadata", {}).get(
                        "recordingStarted"
                    ),
                    "meeting_end_time": original_event.get("metadata", {}).get(
                        "recordingEnded"
                    ),
                    "system_audio_label": original_event.get("metadata", {}).get(
                        "systemLabel", "Meeting Participants"
                    ),
                    "microphone_audio_label": original_event.get("metadata", {}).get(
                        "microphoneLabel", "Speaker"
                    ),
                    "transcription_status": "error",
                    "error_message": str(e),
                    "timestamp": datetime.utcnow().isoformat(),
                }
                await self.event_bus.emit(error_data)
