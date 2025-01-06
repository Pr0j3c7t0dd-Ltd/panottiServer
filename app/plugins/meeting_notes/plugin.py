import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, TypedDict, cast
import uuid
from datetime import datetime

import requests

from app.plugins.base import PluginBase
from app.plugins.events.bus import EventBus
from app.plugins.events.models import Event
from app.utils.logging_config import get_logger
from app.models.recording.events import RecordingEvent, EventContext

EventData = dict[str, Any] | RecordingEvent

logger = get_logger(__name__)


class TranscriptLine(TypedDict):
    """Type for a transcript line"""

    start_time: float
    end_time: float
    speaker: str
    content: str


class MeetingNotesPlugin(PluginBase):
    """Plugin for generating meeting notes from transcripts using Ollama LLM"""

    def __init__(self, config: Any, event_bus: EventBus | None = None) -> None:
        """Initialize the plugin"""
        super().__init__(config, event_bus)
        self._req_id = str(uuid.uuid4())
        self.ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
        self.model = "mistral"  # Default model
        self.output_dir = Path("data/meeting_notes")  # Default output directory

        if config and hasattr(config, "config"):
            config_dict = config.config
            if isinstance(config_dict, dict):
                self.model = config_dict.get("model", self.model)
                self.output_dir = Path(
                    config_dict.get("output_dir", str(self.output_dir))
                )

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._executor: ThreadPoolExecutor | None = None
        self._processing_lock = threading.Lock()

    async def _initialize(self) -> None:
        """Initialize plugin"""
        if not self.event_bus:
            logger.warning(
                "No event bus available for plugin",
                extra={
                    "req_id": self._req_id,
                    "plugin_name": self.name
                }
            )
            return

        try:
            max_workers = 4  # Default value

            if hasattr(self, "config") and self.config and hasattr(self.config, "config"):
                config_dict = self.config.config
                if isinstance(config_dict, dict):
                    max_workers = config_dict.get("max_concurrent_tasks", max_workers)

            logger.debug(
                "Initializing meeting notes plugin",
                extra={
                    "req_id": self._req_id,
                    "plugin_name": self.name,
                    "max_workers": max_workers,
                    "model": self.model,
                    "output_dir": str(self.output_dir)
                }
            )

            self._executor = ThreadPoolExecutor(max_workers=max_workers)
            
            # Subscribe to transcription completed event
            await self.event_bus.subscribe(
                "transcription.completed",
                self.handle_transcription_completed
            )
            
            logger.info(
                "Meeting notes plugin initialized",
                extra={
                    "req_id": self._req_id,
                    "plugin_name": self.name,
                    "max_workers": max_workers,
                    "model": self.model,
                    "event": "transcription.completed"
                }
            )

        except Exception as e:
            logger.error(
                "Failed to initialize plugin",
                extra={
                    "req_id": self._req_id,
                    "plugin_name": self.name,
                    "error": str(e)
                }
            )
            raise

    def _extract_transcript_lines(self, transcript_text: str) -> list[TranscriptLine]:
        """Extract transcript lines from markdown text"""
        try:
            # Find the transcript section
            if "## Transcript" not in transcript_text:
                return []

            transcript_section = transcript_text.split("## Transcript")[1].strip()
            lines: list[TranscriptLine] = []

            # Process each line
            for raw_line in transcript_section.split("\n"):
                line = raw_line.strip()
                if not line:
                    continue

                # Parse timestamp and speaker
                match = re.match(
                    r"\[(\d+\.\d+)s - (\d+\.\d+)s\] \(([^)]+)\) (.+)", line
                )
                if match:
                    start_time = float(match.group(1))
                    end_time = float(match.group(2))
                    speaker = match.group(3)
                    content = match.group(4)

                    lines.append(
                        TranscriptLine(
                            start_time=start_time,
                            end_time=end_time,
                            speaker=speaker,
                            content=content,
                        )
                    )

            return lines

        except Exception as e:
            logger.error("Failed to extract transcript lines: %s", str(e), exc_info=True)
            return []

    def _generate_meeting_notes_from_text(self, transcript_text: str) -> str:
        """Generate meeting notes using Ollama LLM"""
        transcript_lines = self._extract_transcript_lines(transcript_text)

        if not transcript_lines:
            return "No transcript lines found to generate notes from."

        # Format transcript lines as text
        lines = [
            f"[{line['start_time']}s - {line['end_time']}s] "
            f"({line['speaker']}) {line['content']}"
            for line in transcript_lines
        ]
        formatted_text = "\n".join(lines)

        # Prepare prompt
        prompt = f"""Please analyze the following transcript and create
comprehensive meeting notes in markdown format. Please ensure the notes are
clear and concise.

---

Meeting Transcript:

{formatted_text}

The meeting notes should include the following sections:

# Event Title: [Create a descriptive title based on the content]

## Meeting Information
Date: [Extract from transcript timestamps]
Duration: [Calculate from first and last timestamp]
Participants: [Extract speaker names from transcript]

## Executive Summary
[Provide a brief, high-level overview of the meeting's purpose and key outcomes]

## Key Discussion Points
[List the main topics discussed]

## Action Items
[List any tasks, assignments, or follow-up items mentioned]

## Decisions Made
[List any decisions or conclusions reached]

## Next Steps
[Outline any planned next steps or future actions]
"""

        # Call Ollama API
        try:
            response = requests.post(
                self.ollama_url,
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                },
                timeout=30,
            )
            response.raise_for_status()
            return cast(str, response.json().get("response", ""))

        except Exception as e:
            logger.error("Failed to generate meeting notes: %s", str(e), exc_info=True)
            return f"Error generating meeting notes: {e}"

    async def _process_transcript(
        self, recording_id: str, transcript_text: str, original_event: Event
    ) -> None:
        """Process transcript and generate meeting notes"""
        try:
            # Generate meeting notes synchronously since the API call is blocking
            meeting_notes = self._generate_meeting_notes_from_text(transcript_text)

            # Save to file
            output_file = self.output_dir / f"{recording_id}_notes.txt"
            output_file.write_text(meeting_notes)

            # Emit completion event
            if self.event_bus:
                from datetime import datetime
                from app.models.recording.events import RecordingEvent

                event_data = {
                    "recording_id": recording_id,
                    "event_type": "meeting_notes.completed",
                    "current_event": {
                        "meeting_notes": {
                            "status": "completed",
                            "timestamp": datetime.utcnow().isoformat(),
                            "output_paths": {
                                "notes": str(output_file)
                            }
                        }
                    },
                    "event_history": {
                        "transcription": original_event.data.get("current_event", {}),
                        "noise_reduction": original_event.data.get("event_history", {}).get("noise_reduction", {}),
                        "recording": original_event.data.get("event_history", {}).get("recording", {})
                    }
                }

                event = RecordingEvent(
                    recording_timestamp=datetime.utcnow().isoformat(),
                    recording_id=recording_id,
                    event="meeting_notes.completed",
                    data=event_data,
                    context=EventContext(
                        correlation_id=str(uuid.uuid4()),
                        source_plugin=self.name
                    )
                )
                await self.event_bus.publish(event)
                
                logger.info(
                    "Meeting notes generation completed",
                    extra={
                        "req_id": self._req_id,
                        "plugin_name": self.name,
                        "recording_id": recording_id,
                        "event": "meeting_notes.completed",
                        "output_file": str(output_file)
                    }
                )

        except Exception as e:
            error_msg = f"Failed to process transcript: {str(e)}"
            logger.error(
                error_msg,
                extra={
                    "req_id": self._req_id,
                    "plugin_name": self.name,
                    "recording_id": recording_id,
                    "error": str(e)
                },
                exc_info=True
            )
            
            if self.event_bus:
                # Emit error event with preserved chain
                event = RecordingEvent(
                    recording_timestamp=datetime.utcnow().isoformat(),
                    recording_id=recording_id,
                    event="meeting_notes.error",
                    data={
                        "recording_id": recording_id,
                        "meeting_notes": {
                            "status": "error",
                            "timestamp": datetime.utcnow().isoformat(),
                            "error": str(e)
                        },
                        # Preserve previous event data
                        "transcription": original_event.data.get("transcription", {}),
                        "noise_reduction": original_event.data.get("noise_reduction", {}),
                        "recording": original_event.data.get("recording", {})
                    },
                    context=original_event.context
                )
                await self.event_bus.publish(event)

    async def handle_event(self, event: Event) -> None:
        """Handle an event"""
        if not isinstance(event, Event) or event.name != "transcript_ready":
            return

        transcript_text = (
            event.payload.get("transcript_text") if event.payload else None
        )
        if not transcript_text:
            logger.warning("No transcript text in event")
            return

        recording_id = (
            event.payload.get("recording_id", "unknown") if event.payload else "unknown"
        )

        try:
            if self._executor:
                await self._process_transcript(recording_id, transcript_text, event)
        except Exception as e:
            logger.error("Failed to handle transcript event: %s", str(e), exc_info=True)

    async def handle_transcription_completed(self, event_data: EventData) -> None:
        """Handle transcription completed event"""
        try:
            event_id = str(uuid.uuid4())  # Generate new event ID
            
            logger.info(
                "Processing transcription completed event",
                extra={
                    "plugin_name": self.name,
                    "event_id": getattr(event_data, "event_id", None)
                }
            )

            # Debug logging for event data
            logger.debug(
                "Transcription event data",
                extra={
                    "plugin_name": self.name,
                    "event_data": str(event_data),
                    "event_data_type": type(event_data).__name__,
                    "has_transcript_path": "transcript_path" in event_data if isinstance(event_data, dict) else hasattr(event_data, "transcript_path"),
                    "has_transcript_paths": "transcript_paths" in event_data if isinstance(event_data, dict) else hasattr(event_data, "transcript_paths"),
                    "transcript_path": event_data.get("transcript_path") if isinstance(event_data, dict) else getattr(event_data, "transcript_path", None),
                    "transcript_paths": event_data.get("transcript_paths") if isinstance(event_data, dict) else getattr(event_data, "transcript_paths", None)
                }
            )

            # Get transcript path
            transcript_path = None
            if isinstance(event_data, dict):
                transcript_path = event_data.get("transcript_path") or \
                                (event_data.get("transcript_paths", {}).get("merged") if event_data.get("transcript_paths") else None)
            else:
                transcript_path = getattr(event_data, "transcript_path", None) or \
                                (getattr(event_data, "transcript_paths", {}).get("merged") if hasattr(event_data, "transcript_paths") else None)

            if not transcript_path:
                logger.warning(
                    "No transcript path found in event",
                    extra={
                        "plugin_name": self.name,
                        "event_id": getattr(event_data, "event_id", None)
                    }
                )
                return

            # Convert transcript path to Path object
            transcript_path = Path(transcript_path)

            # Get recording ID
            recording_id = event_data.get("recording_id") if isinstance(event_data, dict) else getattr(event_data, "recording_id", None)

            # Generate meeting notes
            output_path = await self._generate_meeting_notes(
                transcript_path,
                event_id,
                recording_id
            )

            if output_path:
                logger.info(
                    "Meeting notes generated successfully",
                    extra={
                        "req_id": event_id,
                        "plugin_name": self.name,
                        "output_path": str(output_path),
                        "recording_id": recording_id
                    }
                )

                # Emit completion event
                await self.emit_event(
                    "meeting_notes.completed",
                    {
                        "recording_id": recording_id,
                        "notes_path": str(output_path)
                    }
                )
            else:
                logger.error(
                    "Failed to generate meeting notes",
                    extra={
                        "plugin_name": self.name,
                        "transcript_path": str(transcript_path)
                    }
                )

        except Exception as e:
            logger.error(
                "Error processing transcription event",
                extra={
                    "plugin_name": self.name,
                    "error": str(e)
                }
            )
            raise

    def _get_transcript_path(self, event: Event | RecordingEvent) -> Path | None:
        """Extract transcript path from event."""
        # Handle RecordingEvent type
        if isinstance(event, RecordingEvent):
            if event.output_file:
                return Path(event.output_file)
            return None
            
        # Handle generic Event type
        if hasattr(event, 'data') and isinstance(event.data, dict):
            transcript_path = event.data.get('transcript_path') or event.data.get('output_file')
            if transcript_path:
                return Path(transcript_path)
        
        return None

    def _read_transcript(self, transcript_path: str | Path) -> str:
        """Read transcript file contents"""
        try:
            # Convert string path to Path object if needed
            path = Path(transcript_path) if isinstance(transcript_path, str) else transcript_path
            return path.read_text()
        except Exception as e:
            logger.error(
                "Failed to read transcript",
                extra={
                    "plugin_name": self.name,
                    "transcript_path": str(transcript_path)
                }
            )
            raise

    async def _generate_notes_with_llm(self, transcript: str, event_id: str) -> str | None:
        """Generate notes using LLM from transcript text."""
        try:
            return self._generate_meeting_notes_from_text(transcript)
        except Exception as e:
            logger.error("Failed to generate notes with LLM: %s", str(e), exc_info=True)
            return None

    def _get_output_path(self, transcript_path: Path) -> Path:
        """Get output path for meeting notes file."""
        logger.debug(
            "Generating output path",
            extra={
                "plugin_name": self.name,
                "transcript_path": str(transcript_path),
                "transcript_path_type": type(transcript_path).__name__,
                "output_dir": str(self.output_dir)
            }
        )
        output_path = self.output_dir / f"{transcript_path.stem}_notes.md"
        logger.debug(
            "Generated output path",
            extra={
                "plugin_name": self.name,
                "output_path": str(output_path)
            }
        )
        return output_path

    async def _generate_meeting_notes(
        self,
        transcript_path: Path,
        event_id: str,
        recording_id: str | None = None
    ) -> Path | None:
        """Generate meeting notes from transcript."""
        try:
            logger.debug(
                "Starting meeting notes generation",
                extra={
                    "req_id": event_id,
                    "plugin_name": self.name,
                    "transcript_path": str(transcript_path),
                    "recording_id": recording_id
                }
            )

            # Read transcript
            transcript = self._read_transcript(transcript_path)
            if not transcript:
                logger.error(
                    "Failed to read transcript",
                    extra={
                        "req_id": event_id,
                        "plugin_name": self.name,
                        "transcript_path": str(transcript_path)
                    }
                )
                return None

            # Generate notes using LLM
            notes = await self._generate_notes_with_llm(transcript, event_id)
            if not notes:
                logger.error(
                    "Failed to generate notes with LLM",
                    extra={
                        "req_id": event_id,
                        "plugin_name": self.name,
                        "transcript_length": len(transcript)
                    }
                )
                return None

            # Save notes to file
            output_path = self._get_output_path(transcript_path)
            output_path.write_text(notes)

            logger.debug(
                "Meeting notes saved to file",
                extra={
                    "req_id": event_id,
                    "plugin_name": self.name,
                    "output_path": str(output_path),
                    "notes_length": len(notes)
                }
            )

            return output_path

        except Exception as e:
            logger.error(
                "Error generating meeting notes",
                extra={
                    "req_id": event_id,
                    "plugin_name": self.name,
                    "error": str(e)
                }
            )
            return None

    async def _shutdown(self) -> None:
        """Shutdown plugin"""
        if self._executor:
            self._executor.shutdown(wait=True)
        logger.info(
            "Meeting notes plugin shutdown",
            extra={
                "req_id": self._req_id,
                "plugin_name": self.name
            }
        )
