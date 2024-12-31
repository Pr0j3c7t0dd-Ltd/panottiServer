import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, TypedDict, cast

import requests

from app.plugins.base import PluginBase
from app.plugins.events.bus import EventBus
from app.plugins.events.models import Event
from app.utils.logging_config import get_logger

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
        max_workers = 4  # Default value

        if hasattr(self, "config") and self.config and hasattr(self.config, "config"):
            config_dict = self.config.config
            if isinstance(config_dict, dict):
                max_workers = config_dict.get("max_concurrent_tasks", max_workers)

        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        logger.info(
            "Meeting notes plugin initialized", extra={"max_workers": max_workers}
        )

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
                event_data: dict[str, Any] = {
                    "type": "meeting_notes_complete",
                    "recording_id": recording_id,
                    "output_path": str(output_file),
                }
                await self.event_bus.emit(event_data)

        except Exception as e:
            if self.event_bus:
                error_data: dict[str, Any] = {
                    "type": "meeting_notes_error",
                    "recording_id": recording_id,
                    "error": str(e),
                }
                await self.event_bus.emit(error_data)
            logger.error(f"Failed to process transcript: {e}", exc_info=True)

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
            if not event.payload:
                logger.error("No payload in transcription completed event")
                return

            # Extract data from event payload
            recording_id = event.payload["recording_id"]
            merged_transcript_path = event.payload["merged_transcript_path"]
            transcription_status = event.payload["transcription_status"]

            if transcription_status == "error":
                error_message = event.payload.get(
                    "error_message", "Unknown error in transcription"
                )
                logger.error(
                    f"Transcription failed: {error_message}",
                    extra={"recording_id": recording_id},
                )
                return

            # Read transcript file
            try:
                transcript_text = Path(merged_transcript_path).read_text()
            except Exception as e:
                logger.error(
                    f"Failed to read transcript file: {e}",
                    extra={"recording_id": recording_id},
                    exc_info=True,
                )
                return

            logger.info(
                "Processing transcript",
                extra={
                    "recording_id": recording_id,
                    "correlation_id": (
                        event.context.correlation_id if event.context else None
                    ),
                    "input_path": merged_transcript_path,
                    "meeting_title": event.payload.get("meeting_title"),
                    "meeting_provider": event.payload.get("meeting_provider"),
                },
            )

            # Process transcript
            await self._process_transcript(recording_id, transcript_text, event)

        except Exception as e:
            logger.error(
                "Failed to handle transcription completed event",
                extra={
                    "recording_id": (
                        recording_id if "recording_id" in locals() else None
                    ),
                    "error": str(e),
                    "event_payload": event.payload if event else None,
                },
            )
            raise

    async def _shutdown(self) -> None:
        """Shutdown plugin"""
        if self._executor:
            self._executor.shutdown(wait=True)
        logger.info("Meeting notes plugin shutdown")
