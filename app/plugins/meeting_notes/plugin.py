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
            logger.warning("No event bus available for plugin")
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
                    "plugin": self.name,
                    "config": {
                        "max_workers": max_workers,
                        "model": self.model,
                        "output_dir": str(self.output_dir)
                    }
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
                    "plugin": self.name,
                    "subscribed_events": ["transcription.completed"],
                    "handler": "handle_transcription_completed",
                    "config": {
                        "max_workers": max_workers,
                        "model": self.model
                    }
                }
            )

        except Exception as e:
            logger.error(
                "Failed to initialize plugin",
                extra={
                    "plugin": self.name,
                    "error": str(e)
                },
                exc_info=True
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
            logger.error(f"Failed to extract transcript lines: {e}", exc_info=True)
            return []

    def _generate_meeting_notes(self, transcript_text: str) -> str:
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
            logger.error(f"Failed to generate meeting notes: {e}", exc_info=True)
            return f"Error generating meeting notes: {e}"

    async def _process_transcript(
        self, recording_id: str, transcript_text: str, original_event: Event
    ) -> None:
        """Process transcript and generate meeting notes"""
        try:
            # Generate meeting notes synchronously since the API call is blocking
            meeting_notes = self._generate_meeting_notes(transcript_text)

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
                        "plugin": self.name,
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
                    "plugin": self.name,
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
            logger.error(f"Failed to handle transcript event: {e}", exc_info=True)

    async def handle_transcription_completed(self, event: Event) -> None:
        """Handle transcription completed event"""
        try:
            recording_id = event.data.get("recording_id")
            transcription_details = event.data.get("current_event", {}).get("transcription", {})
            output_paths = transcription_details.get("output_paths", {})
            transcript_path = output_paths.get("transcript")

            # Process transcript and generate notes
            if transcript_path:
                with open(transcript_path) as f:
                    transcript_text = f.read()
                await self._process_transcript(recording_id, transcript_text, event)
            else:
                logger.error("No transcript path found in event data")
        except Exception as e:
            logger.error(
                "Failed to handle transcription completed event",
                extra={
                    "recording_id": recording_id
                    if "recording_id" in locals()
                    else None,
                    "error": str(e),
                    "event_payload": event.data if event else None,
                },
                exc_info=True,
            )
            raise

    async def _shutdown(self) -> None:
        """Shutdown plugin"""
        if self._executor:
            self._executor.shutdown(wait=True)
        logger.info("Meeting notes plugin shutdown")
