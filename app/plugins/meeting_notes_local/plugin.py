import asyncio
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC
from datetime import datetime as dt
from pathlib import Path
from typing import Any, cast

import aiohttp

from app.core.events import ConcreteEventBus as EventBus
from app.core.events import Event, EventPriority
from app.core.plugins import PluginBase
from app.models.recording.events import RecordingEvent
from app.utils.logging_config import get_logger

EventData = dict[str, Any] | RecordingEvent

logger = get_logger(__name__)


class MeetingNotesLocalPlugin(PluginBase):
    """Plugin for generating meeting notes from transcripts using Ollama LLM"""

    def __init__(self, config: Any, event_bus: EventBus | None = None) -> None:
        """Initialize the plugin"""
        super().__init__(config=config, event_bus=event_bus)
        self._req_id = str(uuid.uuid4())

        # Default values
        self.ollama_url = "http://localhost:11434/api/generate"
        self.model = "llama3.1:8b"
        self.output_dir = Path("data/meeting_notes_local")
        self.num_ctx = 128000
        self.max_concurrent_tasks = 4
        self.timeout = 300  # Default timeout of 5 minutes

        # Override with config values if available
        if config and hasattr(config, "config"):
            config_dict = config.config
            if isinstance(config_dict, dict):
                self.ollama_url = config_dict.get("ollama_url", self.ollama_url)
                self.model = config_dict.get("model_name", self.model)
                self.output_dir = Path(
                    config_dict.get("output_directory", str(self.output_dir))
                )
                self.num_ctx = config_dict.get("num_ctx", self.num_ctx)
                self.max_concurrent_tasks = config_dict.get(
                    "max_concurrent_tasks", self.max_concurrent_tasks
                )
                self.timeout = config_dict.get("timeout", self.timeout)

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize thread pool with configured max tasks
        self._executor = ThreadPoolExecutor(max_workers=self.max_concurrent_tasks)
        self._processing_lock = threading.Lock()

        logger.info(
            "Initializing meeting notes plugin",
            extra={
                "plugin_name": self.name,
                "output_directory": str(self.output_dir),
                "ollama_url": self.ollama_url,
                "num_ctx": self.num_ctx,
            },
        )

    async def _initialize(self) -> None:
        """Initialize plugin"""
        if not self.event_bus:
            logger.error(
                "Event bus is required for meeting notes plugin",
                extra={"req_id": self._req_id, "plugin_name": self.name},
            )
            raise RuntimeError("Event bus is required for meeting notes plugin")

        try:
            logger.debug(
                "Initializing meeting notes plugin",
                extra={
                    "req_id": self._req_id,
                    "plugin_name": self.name,
                    "max_workers": self.max_concurrent_tasks,
                    "model": self.model,
                    "output_dir": str(self.output_dir),
                    "ollama_url": self.ollama_url,
                    "num_ctx": self.num_ctx,
                },
            )

            # Subscribe to transcription completed event
            await self.event_bus.subscribe(
                "transcription_local.completed", self.handle_transcription_completed
            )

            logger.info(
                "Meeting notes plugin initialized",
                extra={
                    "req_id": self._req_id,
                    "plugin_name": self.name,
                    "max_workers": self.max_concurrent_tasks,
                    "model": self.model,
                    "event": "transcription_local.completed",
                    "output_dir": str(self.output_dir),
                },
            )

        except Exception as e:
            logger.error(
                "Failed to initialize plugin",
                extra={
                    "req_id": self._req_id,
                    "plugin_name": self.name,
                    "error": str(e),
                },
            )
            raise

    async def _generate_meeting_notes_from_text(self, transcript_text: str) -> str:
        """Generate meeting notes using Ollama LLM"""
        if not transcript_text:
            return "No transcript text found to generate notes from."

        # Log LLM request details
        logger.info(
            "Preparing LLM request",
            extra={
                "plugin_name": self.name,
                "model": self.model,
                "ollama_url": self.ollama_url,
                "num_ctx": self.num_ctx,
                "transcript_length": len(transcript_text),
            },
        )

        # Extract metadata section
        metadata_section = ""
        transcript_content = transcript_text
        if "## Metadata" in transcript_text:
            parts = transcript_text.split("## Metadata", 1)
            if len(parts) > 1:
                metadata_parts = parts[1].split(
                    "```", 3
                )  # Split into 3 parts to handle both json and transcript sections
                if len(metadata_parts) > 2:
                    metadata_section = (
                        metadata_parts[1].replace("json", "").strip()
                    )  # Get the JSON content
                    # Get the transcript section after metadata
                    transcript_content = (
                        "## Transcript" + metadata_parts[2].split("## Transcript", 1)[1]
                        if "## Transcript" in metadata_parts[2]
                        else ""
                    )

        logger.debug(
            "Extracted metadata and transcript",
            extra={
                "plugin_name": self.name,
                "has_metadata": bool(metadata_section),
                "metadata_length": len(metadata_section),
                "transcript_length": len(transcript_content),
            },
        )

        # Get today's date in the specified format
        today_date = dt.now(UTC).strftime("%B %-d, %Y at %-I:%M %p")

        # Prepare prompt with explicit metadata handling
        prompt = f"""Please analyze the following transcript and create comprehensive meeting notes in markdown format. The transcript includes METADATA in JSON format that you should use for the meeting title and information section.

Please ensure the notes are clear, concise, and well-organized using markdown formatting.

IMPORTANT:
1. Do not use placeholders - extract and use the actual values from the METADATA JSON and the transcript.
2. For attendees, use ONLY the email addresses or names from eventAttendees in the METADATA JSON, not the speakers list.
3. Don't include any other information in the notes, just the meeting notes.

START Transcript:
{transcript_content}
END Transcript

START METADATA JSON:
{metadata_section}
END METADATA JSON

VALIDATION REQUIREMENTS:
1. Every section marked with ## is REQUIRED
2. All formatting must match the examples exactly
3. Do not add any sections not specified in this prompt
4. Do not add any explanatory text or notes
5. Use ONLY information from the provided transcript and metadata
6. Ensure every action item has an owner in parentheses

Create meeting notes with the following sections:

# [Meeting Title (Use the exact meeting title from the METADATA JSON eventTitle field, if not available, create a short, descriptive title based on the transcription content)]

## Meeting Information
- Date: [Use the exact meeting title from the METADATA JSON recordingStarted field, Format EXACTLY as: "January 1, 2025 at 10:00 AM". If today's date is unavailable, use "{today_date}"]
- Duration: [Use the last timesamp in the transcript to determine the duration, Format EXACTLY as: "X hours Y minutes"]
- Attendees: [List ONLY the email addresses from eventAttendees in the METADATA JSON, one per line with a hyphen. If no eventAttendees are available, use "Unknown"]

## Executive Summary
[Provide a brief, high-level overview of the meeting's purpose and key outcomes in 2-3 sentences]

## Key Discussion Points
[For each main topic discussed:
1. Create a clear heading for the topic
2. Under each topic, provide 3-5 bullet points that:
   - Summarize key points made
   - Highlight important details
   - Note any concerns raised
   - Mention specific examples or cases discussed
Keep each bullet point concise but informative]

## Action Items
[Bulleted list of action items with owner and deadline formated EXACTLY as: `(OWNER) ACTION ITEM DESCRIPTION [DEADLINE IF MENTIONED]`. Identify the owner from the context of the meeting transcript]

## Decisions Made
[List specific decisions or conclusions reached during the meeting]

## Next Steps
[Outline any planned next steps or future actions discussed]
"""

        logger.debug(
            "Generated prompt for meeting notes",
            extra={"plugin_name": self.name, "prompt": prompt},
        )

        try:
            # Log request parameters
            logger.debug(
                "Sending request to Ollama",
                extra={
                    "plugin_name": self.name,
                    "model": self.model,
                    "ollama_url": self.ollama_url,
                    "prompt_length": len(prompt),
                    "num_ctx": self.num_ctx,
                    "timestamp": dt.now(UTC),
                },
            )

            async with aiohttp.ClientSession() as session:
                start_time = dt.now(UTC)
                async with session.post(
                    self.ollama_url,
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {"num_ctx": self.num_ctx, "temperature": 0},
                    },
                    timeout=aiohttp.ClientTimeout(total=self.timeout),
                ) as response:
                    response.raise_for_status()
                    result = await response.json()
                    end_time = dt.now(UTC)
                    duration = (end_time - start_time).total_seconds()

                    # Log response details
                    logger.info(
                        "Received response from Ollama",
                        extra={
                            "plugin_name": self.name,
                            "status_code": response.status,
                            "duration_seconds": duration,
                            "response_length": len(result.get("response", "")),
                            "model": self.model,
                            "timestamp": end_time,
                        },
                    )

                    # Clean and return the response
                    return cast(
                        str, self._clean_llm_response(result.get("response", ""))
                    )

        except aiohttp.ClientError as e:
            logger.error(
                "Network error while calling Ollama",
                extra={
                    "plugin_name": self.name,
                    "error_type": type(e).__name__,
                    "error": str(e),
                    "model": self.model,
                    "ollama_url": self.ollama_url,
                },
                exc_info=True,
            )
            return f"Error calling Ollama API: {e}"
        except Exception as e:
            logger.error(
                "Failed to generate meeting notes",
                extra={
                    "plugin_name": self.name,
                    "error_type": type(e).__name__,
                    "error": str(e),
                    "transcript_length": len(transcript_text),
                    "model": self.model,
                },
                exc_info=True,
            )
            return f"Error generating meeting notes: {e}"

    def _clean_llm_response(self, response: str) -> str:
        """Clean the LLM response by removing thinking tags and markdown code blocks."""
        # Remove thinking tags and their content
        while "<think>" in response and "</think>" in response:
            start = response.find("<think>")
            end = response.find("</think>") + len("</think>")
            response = response[:start] + response[end:]

        # Remove markdown code blocks
        response = response.replace("```markdown", "").replace("```", "")

        # Trim whitespace
        response = response.strip()

        return response

    async def _process_transcript(
        self, recording_id: str, transcript_text: str, original_event: Event
    ) -> None:
        """Process transcript and generate meeting notes"""
        try:
            # Generate meeting notes synchronously since the API call is blocking
            meeting_notes = await self._generate_meeting_notes_from_text(
                transcript_text
            )

            # Save to file
            output_file = self.output_dir / f"{recording_id}_notes.txt"
            output_file.write_text(meeting_notes)

            # Emit completion event
            if self.event_bus:
                from datetime import datetime as dt

                event_data = {
                    "recording": original_event.data.get("recording", {}),
                    "noise_reduction": original_event.data.get("noise_reduction", {}),
                    "transcription": original_event.data.get("transcription", {}),
                    "meeting_notes_local": {
                        "status": "completed",
                        "timestamp": dt.now(UTC).isoformat(),
                        "recording_id": recording_id,
                        "output_path": str(output_file),
                        "notes_path": str(output_file),
                        "input_paths": {
                            "transcript": str(
                                original_event.data.get("transcription", {}).get(
                                    "output_file", ""
                                )
                            )
                        },
                    },
                    "meeting_notes": {
                        "status": "completed",
                        "timestamp": dt.now(UTC).isoformat(),
                        "recording_id": recording_id,
                        "output_path": str(output_file),
                        "notes_path": str(output_file),
                        "input_paths": {
                            "transcript": str(
                                original_event.data.get("transcription", {}).get(
                                    "output_file", ""
                                )
                            )
                        },
                    },
                }
                # Include metadata in event data
                if isinstance(original_event, dict):
                    event_data["metadata"] = original_event.get("metadata", {})
                else:
                    event_data["metadata"] = (
                        getattr(original_event.context, "metadata", {})
                        if hasattr(original_event, "context")
                        else {}
                    )

                event = Event.create(
                    name="meeting_notes_local.completed",
                    data=event_data,
                    correlation_id=getattr(
                        original_event.context, "correlation_id", str(uuid.uuid4())
                    ),
                    source_plugin=self.name,
                    priority=EventPriority.NORMAL,
                )

                await self.event_bus.publish(event)

                logger.info(
                    "Meeting notes completion event published",
                    extra={
                        "plugin": self.name,
                        "event_name": "meeting_notes_local.completed",
                        "recording_id": recording_id,
                        "output_path": str(output_file),
                    },
                )

        except Exception as e:
            error_msg = f"Failed to process transcript: {e!s}"
            logger.error(
                error_msg,
                extra={
                    "req_id": self._req_id,
                    "plugin_name": self.name,
                    "recording_id": recording_id,
                    "error": str(e),
                },
                exc_info=True,
            )

            if self.event_bus:
                # Emit error event with preserved chain
                error_data = {
                    # Preserve original event data
                    "recording": original_event.data.get("recording", {})
                    if hasattr(original_event, "data")
                    else {},
                    "noise_reduction": original_event.data.get("noise_reduction", {})
                    if hasattr(original_event, "data")
                    else {},
                    "transcription": original_event.data.get("transcription", {})
                    if hasattr(original_event, "data")
                    else {},
                    # Add current event data
                    "meeting_notes": {
                        "status": "error",
                        "timestamp": dt.now(UTC).isoformat(),
                        "recording_id": recording_id,
                        "error": str(e),
                        "config": {
                            "model": self.model,
                            "num_ctx": self.num_ctx,
                            "timeout": self.timeout,
                            "max_concurrent_tasks": self.max_concurrent_tasks,
                        },
                    },
                    "metadata": original_event.data.get("metadata", {})
                    if hasattr(original_event, "data")
                    else {},
                    "context": {
                        "correlation_id": getattr(
                            original_event, "correlation_id", str(uuid.uuid4())
                        ),
                        "source_plugin": self.name,
                        "metadata": original_event.data.get("metadata", {})
                        if hasattr(original_event, "data")
                        else {},
                    },
                }

                error_event = Event.create(
                    name="meeting_notes_local.error",
                    data=error_data,
                    correlation_id=getattr(
                        original_event, "correlation_id", str(uuid.uuid4())
                    ),
                    source_plugin=self.name,
                    priority=EventPriority.NORMAL,
                )
                await self.event_bus.publish(error_event)

    async def handle_event(self, event_data: Event) -> None:
        """Handle an event"""
        if not isinstance(event_data, Event) or event_data.name != "transcript_ready":
            return

        transcript_text = (
            event_data.payload.get("transcript_text") if event_data.payload else None
        )
        if not transcript_text:
            logger.warning("No transcript text in event")
            return

        recording_id = (
            event_data.payload.get("recording_id", "unknown")
            if event_data.payload
            else "unknown"
        )

        try:
            if self._executor:
                await self._process_transcript(
                    recording_id, transcript_text, event_data
                )
        except Exception as e:
            logger.error("Failed to handle transcript event: %s", str(e), exc_info=True)

    async def handle_transcription_completed(self, event_data: Event) -> None:
        """Handle transcription completed event."""
        event_id = getattr(event_data, "event_id", None) or str(uuid.uuid4())

        # Get data from transcription section
        transcription_data = event_data.data.get("transcription", {})
        recording_id = transcription_data.get("recording_id")

        # Get metadata from event context if available, otherwise from data
        metadata = {}
        if hasattr(event_data, "data"):
            # First try context metadata
            context = event_data.data.get("context", {})
            if isinstance(context, dict):
                metadata.update(context.get("metadata", {}))

            # Then try direct metadata
            direct_metadata = event_data.data.get("metadata", {})
            if isinstance(direct_metadata, dict):
                metadata.update(direct_metadata)

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
                "Found metadata in event data",
                extra={
                    "req_id": event_id,
                    "plugin_name": self.name,
                    "recording_id": recording_id,
                    "event_type": type(event_data).__name__,
                    "metadata": metadata,
                },
            )

        try:
            logger.info(
                "Processing transcription completed event",
                extra={
                    "plugin_name": self.name,
                    "event_id": getattr(event_data, "event_id", None),
                },
            )

            # Debug logging for event data
            logger.debug(
                "Transcription event data",
                extra={
                    "plugin_name": self.name,
                    "event_data": str(event_data),
                    "event_data_type": type(event_data).__name__,
                    "has_transcript_path": hasattr(event_data, "transcript_path"),
                    "has_transcript_paths": hasattr(event_data, "transcript_paths"),
                },
            )

            # Get transcript path from various possible locations in event data
            transcript_path = await self._get_transcript_path(event_data)
            if not transcript_path:
                logger.warning(
                    "No transcript path found in event",
                    extra={
                        "plugin_name": self.name,
                        "event_id": getattr(event_data, "event_id", None),
                        "event_data": str(event_data),
                    },
                )
                return

            # Generate meeting notes
            output_path = await self._generate_meeting_notes(
                transcript_path, event_id, recording_id
            )

            if output_path:
                logger.info(
                    "Generated meeting notes",
                    extra={
                        "req_id": event_id,
                        "plugin_name": self.name,
                        "output_path": str(output_path),
                        "recording_id": recording_id,
                    },
                )

                # Emit completion event
                if self.event_bus:
                    try:
                        # Extract metadata
                        metadata = (
                            event_data.data.get("metadata", {})
                            if hasattr(event_data, "data")
                            else {}
                        )

                        completion_data = {
                            "recording": event_data.data.get("recording", {})
                            if hasattr(event_data, "data")
                            else {},
                            "noise_reduction": event_data.data.get(
                                "noise_reduction", {}
                            )
                            if hasattr(event_data, "data")
                            else {},
                            "transcription": event_data.data.get("transcription", {})
                            if hasattr(event_data, "data")
                            else {},
                            "meeting_notes": {
                                "status": "completed",
                                "timestamp": dt.now(UTC).isoformat(),
                                "recording_id": recording_id,
                                "notes_path": str(output_path),
                                "input_paths": {
                                    "transcript": str(transcript_path),
                                },
                                "config": {
                                    "model": self.model,
                                    "num_ctx": self.num_ctx,
                                    "timeout": self.timeout,
                                    "max_concurrent_tasks": self.max_concurrent_tasks,
                                },
                            },
                            "metadata": metadata,
                            "context": {
                                "correlation_id": getattr(
                                    event_data, "correlation_id", str(uuid.uuid4())
                                ),
                                "source_plugin": self.name,
                                "metadata": metadata,
                            },
                        }

                        completion_event = Event.create(
                            name="meeting_notes_local.completed",
                            data=completion_data,
                            correlation_id=getattr(
                                event_data, "correlation_id", str(uuid.uuid4())
                            ),
                            source_plugin=self.name,
                            priority=EventPriority.NORMAL,
                        )
                        await self.event_bus.publish(completion_event)
                        logger.info(
                            "Published meeting notes completion event",
                            extra={
                                "plugin": self.name,
                                "event_name": "meeting_notes_local.completed",
                                "recording_id": recording_id,
                            },
                        )
                        return
                    except Exception:
                        logger.error(
                            "Failed to publish completion event",
                            extra={
                                "plugin": self.name,
                                "event_name": "meeting_notes_local.completed",
                                "recording_id": recording_id,
                                "output_path": str(output_path),
                            },
                        )
            else:
                logger.warning(
                    "No event bus available to publish completion event",
                    extra={
                        "plugin_name": self.name,
                        "event_name": "meeting_notes_local.completed",
                        "recording_id": recording_id,
                        "output_path": str(output_path),
                    },
                )

        except Exception as e:
            logger.error(
                "Error processing transcription event",
                extra={
                    "plugin_name": self.name,
                    "error": str(e),
                },
            )

    async def _get_transcript_path(self, event: Event | RecordingEvent) -> Path | None:
        """Extract transcript path from event."""
        # Handle RecordingEvent type
        if isinstance(event, RecordingEvent):
            if event.output_file:
                return Path(event.output_file)
            return None

        # Handle generic Event type
        if hasattr(event, "data") and isinstance(event.data, dict):
            transcription_data = event.data.get("transcription", {})

            logger.debug(
                "Examining transcription data",
                extra={
                    "plugin_name": self.name,
                    "transcription_data": transcription_data,
                    "has_output_file": "output_file" in transcription_data,
                    "has_transcript_paths": "transcript_paths" in transcription_data,
                },
            )

            # First try to get single output file
            if transcription_data.get("output_file"):
                logger.debug(
                    "Found output_file",
                    extra={
                        "plugin_name": self.name,
                        "output_file": transcription_data["output_file"],
                    },
                )
                return Path(transcription_data["output_file"])

            # Then try multiple transcript paths - prefer merged, then system, then mic
            transcript_paths = transcription_data.get("transcript_paths", {})
            if transcript_paths:
                logger.debug(
                    "Found transcript_paths",
                    extra={
                        "plugin_name": self.name,
                        "transcript_paths": transcript_paths,
                    },
                )
                # Prefer merged transcript if available
                if transcript_paths.get("merged"):
                    return Path(transcript_paths["merged"])
                # Then try system audio transcript
                elif transcript_paths.get("system"):
                    return Path(transcript_paths["system"])
                # Finally try microphone transcript
                elif transcript_paths.get("mic"):
                    return Path(transcript_paths["mic"])

        logger.warning(
            "No transcript path found in event",
            extra={
                "plugin_name": self.name,
                "event_id": getattr(event, "event_id", None),
                "event_data": str(event),
            },
        )
        return None

    async def _read_transcript(self, transcript_path: str | Path) -> str:
        """Read transcript file contents"""
        try:
            # Convert string path to Path object if needed
            path = (
                Path(transcript_path)
                if isinstance(transcript_path, str)
                else transcript_path
            )

            # Run file read in thread pool
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(self._executor, path.read_text)

        except Exception:
            logger.error(
                "Failed to read transcript",
                extra={
                    "plugin_name": self.name,
                    "transcript_path": str(transcript_path),
                },
            )
            raise

    async def _generate_notes_with_llm(
        self, transcript: str, event_id: str
    ) -> str | None:
        """Generate notes using LLM from transcript text."""
        try:
            return await self._generate_meeting_notes_from_text(transcript)
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
                "output_dir": str(self.output_dir),
            },
        )
        stem = transcript_path.stem.replace("_merged", "")
        output_path = self.output_dir / f"{stem}_meeting_notes.md"
        logger.debug(
            "Generated output path",
            extra={"plugin_name": self.name, "output_path": str(output_path)},
        )
        return output_path

    async def _generate_meeting_notes(
        self, transcript_path: Path, event_id: str, recording_id: str | None = None
    ) -> Path | None:
        """Generate meeting notes from transcript."""
        try:
            logger.debug(
                "Starting meeting notes generation",
                extra={
                    "req_id": event_id,
                    "plugin_name": self.name,
                    "transcript_path": str(transcript_path),
                    "recording_id": recording_id,
                },
            )

            # Read transcript
            transcript = await self._read_transcript(transcript_path)
            if not transcript:
                logger.error(
                    "Failed to read transcript",
                    extra={
                        "req_id": event_id,
                        "plugin_name": self.name,
                        "transcript_path": str(transcript_path),
                    },
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
                        "transcript_length": len(transcript),
                    },
                )
                return None

            # Save notes to file in thread pool
            output_path = self._get_output_path(transcript_path)
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(self._executor, output_path.write_text, notes)

            logger.debug(
                "Meeting notes saved to file",
                extra={
                    "req_id": event_id,
                    "plugin_name": self.name,
                    "output_path": str(output_path),
                    "notes_length": len(notes),
                },
            )

            return output_path

        except Exception as e:
            logger.error(
                "Error generating meeting notes",
                extra={"req_id": event_id, "plugin_name": self.name, "error": str(e)},
            )
            return None

    async def _shutdown(self) -> None:
        """Shutdown plugin"""
        if self._executor:
            self._executor.shutdown(wait=True)
        logger.info(
            "Meeting notes plugin shutdown",
            extra={"req_id": self._req_id, "plugin_name": self.name},
        )
