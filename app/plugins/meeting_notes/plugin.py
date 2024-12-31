import asyncio
import logging
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

from app.models.database import DatabaseManager
from app.plugins.base import PluginBase
from app.plugins.events.models import Event, EventContext, EventPriority

logger = logging.getLogger(__name__)


class MeetingNotesPlugin(PluginBase):
    """Plugin for generating meeting notes from transcripts using Ollama LLM"""

    def __init__(self, config, event_bus=None):
        super().__init__(config, event_bus)
        self._executor = None
        self._processing_lock = threading.Lock()
        self._db_initialized = False

    async def _initialize(self) -> None:
        """Initialize plugin"""
        # Initialize database table
        self._init_database()

        # Initialize thread pool executor
        max_workers = self.get_config("max_concurrent_tasks", 4)
        self._executor = ThreadPoolExecutor(max_workers=max_workers)

        # Subscribe to transcription completed event
        self.event_bus.subscribe(
            "transcription.completed", self.handle_transcription_completed
        )

        self.logger.info(
            "MeetingNotesPlugin initialized successfully",
            extra={
                "plugin": "meeting_notes",
                "max_workers": max_workers,
                "output_directory": self.get_config(
                    "output_directory", "data/meeting_notes"
                ),
                "ollama_url": self.get_config(
                    "ollama_url", "http://localhost:11434/api/generate"
                ),
                "model_name": self.get_config("model_name", "llama3.1:latest"),
                "num_ctx": self.get_config("num_ctx", 128000),
            },
        )

    def _init_database(self) -> None:
        """Initialize the database table for meeting notes tasks"""
        db = DatabaseManager.get_instance()
        with db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS meeting_notes_tasks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    recording_id TEXT NOT NULL,
                    input_path TEXT NOT NULL,
                    output_path TEXT,
                    status TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    completed_at TIMESTAMP,
                    error TEXT
                )
            """
            )
            conn.commit()
            self._db_initialized = True

    async def handle_transcription_completed(self, event: Event) -> None:
        """Handle transcription completed event"""
        try:
            # Extract data from flattened event payload
            recording_id = event.payload["recording_id"]
            merged_transcript_path = event.payload["merged_transcript_path"]
            transcription_status = event.payload["transcription_status"]

            if transcription_status == "error":
                error_message = event.payload.get(
                    "error_message", "Unknown error in transcription"
                )
                self.logger.error(
                    "Transcription failed, skipping meeting notes generation",
                    extra={"recording_id": recording_id, "error": error_message},
                )
                return

            if not merged_transcript_path or not os.path.exists(merged_transcript_path):
                raise ValueError(
                    f"Invalid merged transcript path: {merged_transcript_path}"
                )

            self.logger.info(
                "Processing transcript",
                extra={
                    "recording_id": recording_id,
                    "correlation_id": event.context.correlation_id,
                    "input_path": merged_transcript_path,
                    "meeting_title": event.payload.get("meeting_title"),
                    "meeting_provider": event.payload.get("meeting_provider"),
                },
            )

            # Create task record
            self._update_task_created(recording_id, merged_transcript_path)

            # Process in thread pool
            self._executor.submit(
                self._process_transcript, recording_id, merged_transcript_path, event
            )

        except Exception as e:
            self.logger.error(
                "Failed to handle transcription completion",
                extra={
                    "recording_id": (
                        recording_id if "recording_id" in locals() else None
                    ),
                    "error": str(e),
                    "event_payload": event.payload,
                },
            )
            raise

    def _update_task_created(self, recording_id: str, input_path: str) -> None:
        """Create task record in database"""
        db = DatabaseManager.get_instance()
        with db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO meeting_notes_tasks
                (recording_id, input_path, status)
                VALUES (?, ?, ?)
                """,
                (recording_id, input_path, "pending"),
            )
            conn.commit()

    def _process_transcript(
        self, recording_id: str, input_path: str, event: Event
    ) -> None:
        """Process transcript and generate meeting notes"""
        try:
            # Read transcript
            with open(input_path) as f:
                transcript_text = f.read()

            # Generate output path
            output_dir = Path(self.get_config("output_directory", "data/meeting_notes"))
            output_dir.mkdir(parents=True, exist_ok=True)

            output_filename = f"{recording_id}_meeting_notes.md"
            output_path = str(output_dir / output_filename)

            # Generate meeting notes
            meeting_notes = self._generate_meeting_notes(transcript_text)

            # Save meeting notes
            with open(output_path, "w") as f:
                f.write(meeting_notes)

            # Update database
            self._update_task_completed(recording_id, output_path)

            # Create completion event
            completion_event = Event(
                type="meeting_notes.completed",
                priority=EventPriority.NORMAL,
                context=EventContext(
                    recording_id=recording_id,
                    correlation_id=event.context.correlation_id,
                    source_plugin="meeting_notes",
                ),
                payload={
                    # Recording identifiers
                    "recording_id": recording_id,
                    "recording_timestamp": event.payload.get("recording_timestamp"),
                    # Audio file paths
                    "raw_system_audio_path": event.payload.get("raw_system_audio_path"),
                    "raw_microphone_audio_path": event.payload.get(
                        "raw_microphone_audio_path"
                    ),
                    "noise_reduced_audio_path": event.payload.get(
                        "noise_reduced_audio_path"
                    ),
                    # Transcript file paths
                    "system_transcript_path": event.payload.get(
                        "system_transcript_path"
                    ),
                    "microphone_transcript_path": event.payload.get(
                        "microphone_transcript_path"
                    ),
                    "merged_transcript_path": event.payload.get(
                        "merged_transcript_path"
                    ),
                    # Meeting notes path
                    "meeting_notes_path": output_path,
                    # Meeting metadata
                    "meeting_title": event.payload.get("meeting_title"),
                    "meeting_provider": event.payload.get("meeting_provider"),
                    "meeting_provider_id": event.payload.get("meeting_provider_id"),
                    "meeting_attendees": event.payload.get("meeting_attendees", []),
                    "meeting_start_time": event.payload.get("meeting_start_time"),
                    "meeting_end_time": event.payload.get("meeting_end_time"),
                    # Audio source labels
                    "system_audio_label": event.payload.get("system_audio_label"),
                    "microphone_audio_label": event.payload.get(
                        "microphone_audio_label"
                    ),
                    # Processing status
                    "meeting_notes_status": "completed",
                },
            )

            asyncio.run(self.event_bus.emit(completion_event))

        except Exception as e:
            error_event = Event(
                type="meeting_notes.error",
                priority=EventPriority.HIGH,
                context=EventContext(
                    recording_id=recording_id,
                    correlation_id=event.context.correlation_id,
                    source_plugin="meeting_notes",
                ),
                payload={
                    # Recording identifiers
                    "recording_id": recording_id,
                    "recording_timestamp": event.payload.get("recording_timestamp"),
                    # Meeting metadata
                    "meeting_title": event.payload.get("meeting_title"),
                    "meeting_provider": event.payload.get("meeting_provider"),
                    "meeting_provider_id": event.payload.get("meeting_provider_id"),
                    # Error details
                    "meeting_notes_status": "error",
                    "error_message": str(e),
                },
            )

            asyncio.run(self.event_bus.emit(error_event))

            self.logger.error(
                "Error processing transcript",
                extra={
                    "plugin": "meeting_notes",
                    "recording_id": recording_id,
                    "error": str(e),
                    "correlation_id": event.context.correlation_id,
                },
                exc_info=True,
            )
            self._update_task_error(recording_id, str(e))

    def _update_task_completed(self, recording_id: str, output_path: str) -> None:
        """Update task as completed in database"""
        db = DatabaseManager.get_instance()
        with db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE meeting_notes_tasks
                SET status = ?, output_path = ?, completed_at = CURRENT_TIMESTAMP
                WHERE recording_id = ?
                """,
                ("completed", output_path, recording_id),
            )
            conn.commit()

    def _update_task_error(self, recording_id: str, error: str) -> None:
        """Update task as failed in database"""
        db = DatabaseManager.get_instance()
        with db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE meeting_notes_tasks
                SET status = ?, error = ?, completed_at = CURRENT_TIMESTAMP
                WHERE recording_id = ?
                """,
                ("failed", error, recording_id),
            )
            conn.commit()

    def _extract_transcript_lines(self, transcript_text: str) -> list:
        """Extract transcript lines from markdown text"""
        try:
            # Find the transcript section
            if "## Transcript" not in transcript_text:
                self.logger.error("Could not find transcript section")
                return []

            transcript_section = transcript_text.split("## Transcript")[1].strip()
            lines = []

            # Process each line
            current_chunk = []
            for line in transcript_section.split("\n"):
                line = line.strip()
                if not line:
                    continue

                # Check if this is a new timestamp line
                match = re.match(r"\[([\d.]+)s - ([\d.]+)s\] \((.*?)\) (.*)", line)
                if match:
                    # If we have a previous chunk, add it
                    if current_chunk:
                        # Get the first chunk's metadata (they're all the same within a chunk)
                        first_chunk = current_chunk[0]
                        content = " ".join(
                            chunk_line for _, _, _, chunk_line in current_chunk
                        )
                        lines.append(
                            {
                                "start_time": float(first_chunk[0]),
                                "end_time": float(first_chunk[1]),
                                "speaker": first_chunk[2],
                                "content": content.strip(),
                            }
                        )
                        current_chunk = []

                    # Start new chunk with all 4 values
                    start_time, end_time, speaker, content = match.groups()
                    current_chunk.append((start_time, end_time, speaker, content))
                # If this is a continuation line, append to current chunk
                elif current_chunk:
                    # Use the same metadata as the previous line
                    prev = current_chunk[-1]
                    current_chunk.append((prev[0], prev[1], prev[2], line))

            # Add the last chunk if any
            if current_chunk:
                first_chunk = current_chunk[0]
                content = " ".join(chunk_line for _, _, _, chunk_line in current_chunk)
                lines.append(
                    {
                        "start_time": float(first_chunk[0]),
                        "end_time": float(first_chunk[1]),
                        "speaker": first_chunk[2],
                        "content": content.strip(),
                    }
                )

            if not lines:
                self.logger.warning(
                    "No transcript lines extracted",
                    extra={"transcript_length": len(transcript_text)},
                )

            return lines

        except Exception as e:
            self.logger.error(
                "Error extracting transcript lines",
                extra={
                    "error": str(e),
                    "transcript_length": len(transcript_text) if transcript_text else 0,
                },
                exc_info=True,
            )
            return []

    def _extract_metadata(self, transcript_text: str) -> str | None:
        """Extract metadata section from transcript"""
        try:
            metadata_section = (
                transcript_text.split("## Recording Metadata")[1]
                .split("## Transcript")[0]
                .strip()
            )
            return metadata_section
        except Exception:
            return None

    def _generate_meeting_notes(self, transcript_text: str) -> str:
        """Generate meeting notes using Ollama LLM"""
        transcript_lines = self._extract_transcript_lines(transcript_text)
        metadata = self._extract_metadata(transcript_text)

        if not transcript_lines:
            raise ValueError("Could not extract transcript lines")

        # Format transcript lines as text
        formatted_transcript = "\n".join(
            f"[{line['start_time']}s - {line['end_time']}s] ({line['speaker']}) {line['content']}"
            for line in transcript_lines
        )

        # Prepare prompt
        prompt = f"""Please analyze the meeting metadata and transcript below and create comprehensive meeting notes in markdown format. Please ensure the notes are clear and concise.

---

Meeting Metadata:

{metadata}

---

Meeting Transcript:

{formatted_transcript}

---

The meeting notes should include the following sections:

# Event Title: [event_title from the Meeting Metadata, or if not available, create a meeting title]

## Meeting Information
Date: [Convert Meeting Metadata meeting date timestamp into a human readable format, just the date/time, no additional comments]
Duration: [Convert the number of seconds from the transcript into a human-readable format]
Location: [Event provider from the Meeting Metadata, if available]

## Attendees
[List each attendee from the <Metadata> event_attendees as a bullet point for each attendee.  If no event_attendees are available, use "Unknown".]

## Executive Summary
[Provide a brief, high-level overview of the meeting's purpose and key outcomes]

## Agenda Items Discussed
[List and elaborate on the main topics discussed]

## Notes and Additional Information
[Include any other relevant information, clarifications, or important context]

## Key Decisions
[List all decisions made during the meeting]

## Risks and Issues
[Document any risks, blockers, or issues raised during the meeting]

## Open Questions
[List any questions that remained unanswered or need further discussion]

## Action Items
[
List action items in format: "- [Owner/Responsible Person] Action description".
NOTE: If no clear owner is mentioned, use "UNASSIGNED". Example:
- [<Owner Name>] Create project timeline by Friday
- [UNASSIGNED] Review security documentation
]

## Next Steps
[Outline immediate next steps and upcoming milestones]

## Next Meeting
[If discussed, include details about the next meeting]
"""

        # Call Ollama API
        try:
            response = requests.post(
                self.get_config("ollama_url", "http://localhost:11434/api/generate"),
                json={
                    "model": self.get_config("model_name", "llama3.1:latest"),
                    "prompt": prompt,
                    "stream": False,  # Don't stream to get complete response
                    "num_ctx": self.get_config("num_ctx", 128000),
                },
                stream=False,  # Don't stream at requests level either
                timeout=30  # Add 30 second timeout
            )
            response.raise_for_status()

            # Extract generated text from response
            response_json = response.json()
            if isinstance(response_json, dict):
                meeting_notes = response_json.get("response", "")
            else:
                self.logger.error(
                    "Unexpected response format from Ollama API",
                    extra={"response": str(response_json)},
                )
                raise ValueError("Unexpected response format from Ollama API")

            if not meeting_notes:
                raise ValueError("Empty response from Ollama API")

            return meeting_notes

        except requests.exceptions.JSONDecodeError as e:
            self.logger.error(
                "Failed to decode Ollama API response",
                extra={
                    "error": str(e),
                    "response_text": response.text[
                        :1000
                    ],  # Log first 1000 chars of response
                },
                exc_info=True,
            )
            raise
        except Exception as e:
            self.logger.error(
                "Error calling Ollama API", extra={"error": str(e)}, exc_info=True
            )
            raise

    async def _shutdown(self) -> None:
        """Shutdown plugin"""
        # Unsubscribe from events
        self.event_bus.unsubscribe(
            "transcription.completed", self.handle_transcription_completed
        )

        # Shutdown thread pool
        if self._executor:
            self._executor.shutdown(wait=True)

        self.logger.info("Meeting notes plugin shutdown")
