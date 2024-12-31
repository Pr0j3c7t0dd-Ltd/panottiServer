import json
import logging
import os
import threading
import wave
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

from faster_whisper import WhisperModel

from app.models.database import DatabaseManager
from app.plugins.base import PluginBase
from app.plugins.events.models import Event, EventContext, EventPriority

logger = logging.getLogger(__name__)


class AudioTranscriptionPlugin(PluginBase):
    """Plugin for transcribing audio files using Whisper"""

    def __init__(self, config, event_bus=None):
        super().__init__(config, event_bus)
        self._executor = None
        self._processing_lock = threading.Lock()
        self._db_initialized = False
        self._model = None

    async def _initialize(self) -> None:
        """Initialize plugin"""
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
                "plugin": "audio_transcription",
                "model_name": model_name,
                "model_dir": model_dir,
            },
        )

        self._model = WhisperModel(
            model_size_or_path=model_dir,  # Use the model directory path directly
            device="cpu",
            compute_type="int8",
            local_files_only=True,  # Only use local files, no downloads
        )

        # Verify model exists
        config_path = os.path.join(model_dir, "config.json")
        if not os.path.exists(config_path):
            raise RuntimeError(
                f"Model files not found in {model_dir}. "
                "Please run the download script first."
            )
        else:
            self.logger.info(
                "Whisper model initialized successfully",
                extra={
                    "plugin": "audio_transcription",
                    "model_name": model_name,
                    "model_dir": model_dir,
                },
            )

        # Subscribe to noise reduction completed event
        self.event_bus.subscribe(
            "noise_reduction.completed", self.handle_noise_reduction_completed
        )

        self.logger.info(
            "AudioTranscriptionPlugin initialized successfully",
            extra={
                "plugin": "audio_transcription",
                "max_workers": max_workers,
                "model": model_name,
                "output_directory": getattr(
                    self.config, "output_directory", "data/transcripts"
                ),
                "db_initialized": self._db_initialized,
            },
        )

    async def _shutdown(self) -> None:
        """Shutdown plugin"""
        # Unsubscribe from events
        self.event_bus.unsubscribe(
            "noise_reduction.completed", self.handle_noise_reduction_completed
        )

        # Shutdown thread pool
        if self._executor:
            self._executor.shutdown(wait=True)

        self.logger.info("Audio transcription plugin shutdown")

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

    def transcribe_audio(
        self, audio_path: str, output_path: str, label: str
    ) -> str | None:
        """Transcribe an audio file using Whisper"""
        try:
            if not self.validate_wav_file(audio_path):
                raise ValueError(f"Invalid WAV file: {audio_path}")

            self.logger.info(f"Transcribing {os.path.basename(audio_path)}...")

            # Transcribe the audio file
            segments, info = self._model.transcribe(
                audio_path, beam_size=5, word_timestamps=True
            )

            # Generate markdown with timestamps and labels
            with open(output_path, "w") as f:
                f.write(f"# Transcript from {label}\n\n")
                for segment in segments:
                    timestamp = f"[{segment.start:.2f}s - {segment.end:.2f}s]"
                    f.write(f"{timestamp} ({label}) {segment.text}\n\n")

            self.logger.info(
                "Transcription completed",
                extra={
                    "file": os.path.basename(audio_path),
                    "output": os.path.basename(output_path),
                },
            )

            return output_path

        except Exception as e:
            self.logger.error(
                "Transcription failed",
                extra={"file": os.path.basename(audio_path), "error": str(e)},
            )
            return None

    def merge_transcripts(
        self,
        transcript_files: list[str],
        output_path: str,
        labels: list[str],
        original_event: dict,
    ) -> str:
        """Merge multiple transcript files based on timestamps"""
        try:
            segments = []
            file_labels = {}  # Map file paths to their labels

            # Create mapping of files to labels
            for file_path, label in zip(transcript_files, labels, strict=False):
                # Use appropriate fallback based on audio type
                fallback = (
                    "Meeting Participants" if "system_audio" in file_path else "Speaker"
                )
                file_labels[file_path] = label or fallback

            # Write merged transcript
            with open(output_path, "w") as f:
                # Write metadata header
                f.write("# Merged Transcript\n\n")
                f.write("## Recording Metadata\n")
                f.write("```json\n")
                metadata = original_event.get("metadata", {})
                metadata_output = {
                    "recording_id": original_event.get("recording_id"),
                    "recording_timestamp": original_event.get("recording_timestamp"),
                    "event_title": metadata.get("eventTitle"),
                    "event_provider": metadata.get("eventProvider"),
                    "event_provider_id": metadata.get("eventProviderId"),
                    "event_attendees": metadata.get("eventAttendees", []),
                    "system_label": metadata.get("systemLabel", "Meeting Participants"),
                    "microphone_label": metadata.get("microphoneLabel", "Speaker"),
                    "recording_started": metadata.get("recordingStarted"),
                    "recording_ended": metadata.get("recordingEnded"),
                    "system_audio_path": original_event.get("system_audio_path"),
                    "microphone_audio_path": original_event.get(
                        "microphone_audio_path"
                    ),
                }
                f.write(json.dumps(metadata_output, indent=2))
                f.write("\n```\n\n")
                f.write("## Transcript\n\n")

                # Read all transcripts and parse segments
                for file_path in transcript_files:
                    label = file_labels[file_path]
                    with open(file_path) as tf:
                        for raw_line in tf:
                            line = raw_line.strip()
                            if line and not line.startswith("#"):  # Skip header
                                # Parse timestamp and text
                                timestamp_end = line.find("]")
                                if timestamp_end != -1:
                                    timestamp = line[1:timestamp_end]

                                    # Extract text after timestamp
                                    text = line[timestamp_end + 2 :]

                                    # Parse start and end times
                                    times = timestamp.split(" - ")
                                    start = float(times[0].replace("s", ""))
                                    end = float(times[1].replace("s", ""))

                                    segments.append((start, end, text, label))

                # Sort segments by start time
                segments.sort(key=lambda x: x[0])

                # Write transcript content
                for start, end, text, label in segments:
                    f.write(f"[{start:.2f}s - {end:.2f}s] ({label}) {text}\n\n")

            return output_path

        except Exception as e:
            self.logger.error(
                "Transcript merge failed",
                extra={"output_file": os.path.basename(output_path), "error": str(e)},
            )
            raise

    async def handle_noise_reduction_completed(self, event: Event) -> None:
        """Handle noise reduction completion events"""
        try:
            # Get recording ID and paths from flattened event payload
            recording_id = event.payload["recording_id"]
            system_audio = event.payload["raw_system_audio_path"]
            microphone_audio = event.payload["noise_reduced_audio_path"]

            if not system_audio or not microphone_audio:
                raise ValueError(
                    "Missing audio paths: "
                    f"system_audio={system_audio}, "
                    f"microphone_audio={microphone_audio}"
                )

            # Create output paths in the transcripts directory
            transcripts_dir = getattr(
                self.config, "transcripts_directory", "data/transcripts"
            )
            os.makedirs(transcripts_dir, exist_ok=True)

            system_transcript = os.path.join(
                transcripts_dir, f"{recording_id}_system_audio_transcript.md"
            )
            microphone_transcript = os.path.join(
                transcripts_dir, f"{recording_id}_microphone_transcript.md"
            )
            merged_transcript = os.path.join(
                transcripts_dir, f"{recording_id}_transcript.md"
            )

            # Get labels from flattened event payload
            system_label = event.payload.get(
                "system_audio_label", "Meeting Participants"
            )
            microphone_label = event.payload.get("microphone_audio_label", "Speaker")

            # Create metadata dictionary from flattened fields
            metadata = {
                "eventTitle": event.payload.get("meeting_title"),
                "eventProvider": event.payload.get("meeting_provider"),
                "eventProviderId": event.payload.get("meeting_provider_id"),
                "eventAttendees": event.payload.get("meeting_attendees", []),
                "systemLabel": system_label,
                "microphoneLabel": microphone_label,
                "recordingStarted": event.payload.get("meeting_start_time"),
                "recordingEnded": event.payload.get("meeting_end_time"),
            }

            self.logger.info(
                "Starting audio transcription",
                extra={
                    "recording_id": recording_id,
                    "system_audio": system_audio,
                    "microphone_audio": microphone_audio,
                    "system_transcript": system_transcript,
                    "microphone_transcript": microphone_transcript,
                    "merged_transcript": merged_transcript,
                    "system_label": system_label,
                    "microphone_label": microphone_label,
                    "transcripts_dir": transcripts_dir,
                    "metadata": metadata,
                },
            )

            # Create task entry
            self._create_task(recording_id, [system_audio, microphone_audio])

            # Create original event format for backward compatibility
            original_event = {
                "recording_id": recording_id,
                "recording_timestamp": event.payload.get("recording_timestamp"),
                "system_audio_path": system_audio,
                "microphone_audio_path": microphone_audio,
                "metadata": metadata,
            }

            # Process audio files
            await self._process_audio(
                recording_id=recording_id,
                input_files=[system_audio, microphone_audio],
                output_files=[system_transcript, microphone_transcript],
                merged_output=merged_transcript,
                original_event=original_event,
                input_labels=[system_label, microphone_label],
            )

        except Exception as e:
            self.logger.error(
                "Error processing noise reduction completion",
                extra={
                    "recording_id": (
                        recording_id if "recording_id" in locals() else None
                    ),
                    "error": str(e),
                    "config": str(self.config) if hasattr(self, "config") else None,
                    "event_payload": event.payload,
                },
            )
            raise

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
                self.transcribe_audio(input_file, output_file, label or fallback)

            # Merge transcripts
            if len(output_files) > 1:
                self.merge_transcripts(
                    output_files, merged_output, input_labels, original_event
                )
            else:
                # If only one transcript, just copy it and add metadata
                with open(output_files[0]) as src:
                    content = src.read()
                with open(merged_output, "w") as dest:
                    # Write metadata
                    dest.write("# Merged Transcript\n\n")
                    dest.write("## Recording Metadata\n")
                    dest.write("```json\n")
                    metadata = {
                        "recording_id": recording_id,
                        "recording_timestamp": original_event.get(
                            "recording_timestamp"
                        ),
                        "event_title": original_event.get("metadata", {}).get(
                            "eventTitle"
                        ),
                        "event_provider": original_event.get("metadata", {}).get(
                            "eventProvider"
                        ),
                        "event_provider_id": original_event.get("metadata", {}).get(
                            "eventProviderId"
                        ),
                        "event_attendees": original_event.get("metadata", {}).get(
                            "eventAttendees", []
                        ),
                        "system_label": original_event.get("metadata", {}).get(
                            "systemLabel", "Meeting Participants"
                        ),
                        "microphone_label": original_event.get("metadata", {}).get(
                            "microphoneLabel", "Speaker"
                        ),
                        "recording_started": original_event.get("metadata", {}).get(
                            "recordingStarted"
                        ),
                        "recording_ended": original_event.get("metadata", {}).get(
                            "recordingEnded"
                        ),
                        "system_audio_path": original_event.get("system_audio_path"),
                        "microphone_audio_path": original_event.get(
                            "microphone_audio_path"
                        ),
                    }
                    dest.write(json.dumps(metadata, indent=2))
                    dest.write("\n```\n\n")
                    dest.write("## Transcript\n\n")
                    dest.write(content)

            # Update status and publish completion event
            self._update_task_status(
                recording_id, "completed", output_files, merged_output
            )

            # Create and publish transcription completed event
            event = Event(
                name="transcription.completed",
                payload={
                    # Recording identifiers
                    "recording_id": recording_id,
                    "recording_timestamp": original_event.get("recording_timestamp"),
                    # Audio file paths
                    "raw_system_audio_path": original_event.get("system_audio_path"),
                    "raw_microphone_audio_path": original_event.get(
                        "microphone_audio_path"
                    ),
                    "noise_reduced_audio_path": original_event.get(
                        "microphone_audio_path"
                    ),
                    # Transcript file paths
                    "system_transcript_path": (
                        output_files[0] if len(output_files) > 0 else None
                    ),
                    "microphone_transcript_path": (
                        output_files[1] if len(output_files) > 1 else None
                    ),
                    "merged_transcript_path": merged_output,
                    # Meeting metadata
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
                    # Audio source labels
                    "system_audio_label": original_event.get("metadata", {}).get(
                        "systemLabel", "Meeting Participants"
                    ),
                    "microphone_audio_label": original_event.get("metadata", {}).get(
                        "microphoneLabel", "Speaker"
                    ),
                    # Processing status
                    "transcription_status": "completed",
                },
                context=EventContext(
                    correlation_id=recording_id, source_plugin=self.name
                ),
                priority=EventPriority.LOW,
            )

            await self.event_bus.publish(event)

        except Exception as e:
            self.logger.error(
                "Audio processing failed",
                extra={"recording_id": recording_id, "error": str(e)},
            )
            self._update_task_status(recording_id, "error", error_message=str(e))

            # Emit error event
            error_event = Event(
                name="transcription.completed",
                payload={
                    # Recording identifiers
                    "recording_id": recording_id,
                    "recording_timestamp": original_event.get("recording_timestamp"),
                    # Audio file paths
                    "raw_system_audio_path": original_event.get("system_audio_path"),
                    "raw_microphone_audio_path": original_event.get(
                        "microphone_audio_path"
                    ),
                    "noise_reduced_audio_path": original_event.get(
                        "microphone_audio_path"
                    ),
                    # Meeting metadata
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
                    # Audio source labels
                    "system_audio_label": original_event.get("metadata", {}).get(
                        "systemLabel", "Meeting Participants"
                    ),
                    "microphone_audio_label": original_event.get("metadata", {}).get(
                        "microphoneLabel", "Speaker"
                    ),
                    # Processing status and error
                    "transcription_status": "error",
                    "error_message": str(e),
                },
                context=EventContext(
                    correlation_id=(
                        original_event.get("correlation_id") if original_event else None
                    ),
                    source_plugin=self.name,
                ),
                priority=EventPriority.LOW,
            )

            await self.event_bus.publish(error_event)
