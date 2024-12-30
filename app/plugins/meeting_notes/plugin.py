import asyncio
import json
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional
import logging
import requests

from app.plugins.base import PluginBase
from app.plugins.events.models import Event, EventContext, EventPriority
from app.models.database import DatabaseManager

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
        self.event_bus.subscribe("transcription.completed", self.handle_transcription_completed)
        
        self.logger.info(
            "MeetingNotesPlugin initialized successfully",
            extra={
                "plugin": "meeting_notes",
                "max_workers": max_workers,
                "output_directory": self.get_config("output_directory", "data/meeting_notes"),
                "ollama_url": self.get_config("ollama_url", "http://localhost:11434/api/generate"),
                "model_name": self.get_config("model_name", "llama3.1:latest"),
                "num_ctx": self.get_config("num_ctx", 128000)
            }
        )

    def _init_database(self) -> None:
        """Initialize the database table for meeting notes tasks"""
        db = DatabaseManager.get_instance()
        with db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
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
            """)
            conn.commit()
            self._db_initialized = True

    async def handle_transcription_completed(self, event: Event) -> None:
        """Handle transcription completed event"""
        try:
            recording_id = event.context.recording_id
            input_path = event.data.get("output_path")
            
            if not input_path or not os.path.exists(input_path):
                raise ValueError(f"Invalid input path: {input_path}")

            self.logger.info(
                "Processing transcript",
                extra={
                    "recording_id": recording_id,
                    "correlation_id": event.context.correlation_id,
                    "input_path": input_path
                }
            )

            # Create task record
            self._update_task_created(recording_id, input_path)

            # Process in thread pool
            self._executor.submit(
                self._process_transcript,
                recording_id,
                input_path,
                event
            )

        except Exception as e:
            self.logger.error(
                "Error handling transcription completed event",
                extra={
                    "plugin": "meeting_notes",
                    "error": str(e),
                    "event_id": event.id,
                    "correlation_id": event.context.correlation_id
                },
                exc_info=True
            )

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
                (recording_id, input_path, "pending")
            )
            conn.commit()

    def _process_transcript(self, recording_id: str, input_path: str, event: Event) -> None:
        """Process transcript and generate meeting notes"""
        try:
            # Read transcript
            with open(input_path, 'r') as f:
                transcript_text = f.read()

            # Generate output path
            output_dir = Path(self.get_config("output_directory", "data/meeting_notes"))
            output_dir.mkdir(parents=True, exist_ok=True)
            
            output_filename = f"{recording_id}_meeting_notes.md"
            output_path = str(output_dir / output_filename)

            # Generate meeting notes
            meeting_notes = self._generate_meeting_notes(transcript_text)

            # Save meeting notes
            with open(output_path, 'w') as f:
                f.write(meeting_notes)

            # Update database
            self._update_task_completed(recording_id, output_path)

            # Create completion event
            event_data = {
                **event.data,
                "meeting_notes_path": output_path
            }
            
            completion_event = Event(
                type="meeting_notes.completed",
                priority=EventPriority.NORMAL,
                context=EventContext(
                    recording_id=recording_id,
                    correlation_id=event.context.correlation_id
                ),
                data=event_data
            )
            
            asyncio.run(self.event_bus.emit(completion_event))

        except Exception as e:
            self.logger.error(
                "Error processing transcript",
                extra={
                    "plugin": "meeting_notes",
                    "recording_id": recording_id,
                    "error": str(e),
                    "correlation_id": event.context.correlation_id
                },
                exc_info=True
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
                ("completed", output_path, recording_id)
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
                ("failed", error, recording_id)
            )
            conn.commit()

    def _extract_transcript_lines(self, transcript_text: str) -> list:
        """Extract transcript lines from markdown text"""
        try:
            transcript_section = transcript_text.split("## Transcript")[1].strip()
            lines = []
            for line in transcript_section.split('\n'):
                if line.strip():
                    match = re.match(r'\[([\d.]+)s - ([\d.]+)s\] \((.*?)\) (.*)', line.strip())
                    if match:
                        start_time, end_time, speaker, content = match.groups()
                        lines.append({
                            'start_time': float(start_time),
                            'end_time': float(end_time),
                            'speaker': speaker,
                            'content': content.strip()
                        })
            return lines
        except Exception as e:
            self.logger.error(
                "Error extracting transcript lines",
                extra={"error": str(e)},
                exc_info=True
            )
            return []

    def _extract_metadata(self, transcript_text: str) -> Optional[str]:
        """Extract metadata section from transcript"""
        try:
            metadata_section = transcript_text.split("## Recording Metadata")[1].split("## Transcript")[0].strip()
            return metadata_section
        except Exception:
            return None

    def _generate_meeting_notes(self, transcript_text: str) -> str:
        """Generate meeting notes using Ollama LLM"""
        transcript_lines = self._extract_transcript_lines(transcript_text)
        metadata = self._extract_metadata(transcript_text)

        if not transcript_lines:
            raise ValueError("Could not extract transcript lines")

        # Prepare prompt
        prompt = f"""Please analyze the meeting metadata and transcript below and create comprehensive meeting notes in markdown format. Please ensure the notes are clear and concise.

Metadata:
{metadata if metadata else 'No metadata available'}

Transcript:
{json.dumps(transcript_lines, indent=2)}

Please format the notes with the following sections:
## Summary
[Brief overview of the meeting]

## Key Points
[List main topics discussed]

## Action Items
[List tasks, assignments, and deadlines]

## Decisions
[List all decisions made during the meeting]

## Risks and Issues
[List any risks or issues discussed]
"""

        # Call Ollama API
        try:
            response = requests.post(
                self.get_config("ollama_url", "http://localhost:11434/api/generate"),
                json={
                    "model": self.get_config("model_name", "llama3.1:latest"),
                    "prompt": prompt,
                    "context_window": self.get_config("num_ctx", 128000)
                }
            )
            response.raise_for_status()
            
            # Extract generated text from response
            meeting_notes = response.json().get("response", "")
            
            if not meeting_notes:
                raise ValueError("Empty response from Ollama API")
                
            return meeting_notes

        except Exception as e:
            self.logger.error(
                "Error calling Ollama API",
                extra={"error": str(e)},
                exc_info=True
            )
            raise

    async def _shutdown(self) -> None:
        """Shutdown plugin"""
        # Unsubscribe from events
        self.event_bus.unsubscribe("transcription.completed", self.handle_transcription_completed)
        
        # Shutdown thread pool
        if self._executor:
            self._executor.shutdown(wait=True)
            
        self.logger.info("Meeting notes plugin shutdown")
