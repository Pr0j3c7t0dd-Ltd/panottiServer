import asyncio
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime as dt, UTC, datetime
from pathlib import Path
from typing import Any, Literal

import google.generativeai as genai
import httpx
from anthropic import AsyncAnthropic  # Update import statement
from openai import AsyncOpenAI

from app.core.events import ConcreteEventBus as EventBus, Event, EventPriority, EventContext
from app.core.plugins import PluginBase  # Updated import path
from app.models.recording.events import RecordingEvent
from app.utils.logging_config import get_logger

EventData = dict[str, Any] | RecordingEvent
ProviderType = Literal["openai", "anthropic", "google"]

logger = get_logger(__name__)


class MeetingNotesRemotePlugin(PluginBase):
    """Plugin for generating meeting notes from transcripts using various LLM providers."""

    SYSTEM_PROMPT = "You are an expert meeting note taker who is attentive and detail oriented.  If you do a good job you will get a large bonus, so try very hard to get all the details correct.  Do not add any details that are not in the transcription, only create accurate meeting notes"

    def __init__(self, config: Any, event_bus: EventBus | None = None) -> None:
        """Initialize the plugin"""
        super().__init__(config, event_bus)
        self._req_id = str(uuid.uuid4())

        # Default values
        self.output_dir = Path("data/meeting_notes_remote")
        self.max_concurrent_tasks = 4
        self.timeout = 600
        self.provider: ProviderType = "openai"

        # Override with config values if available
        if config and hasattr(config, "config"):
            config_dict = config.config
            if isinstance(config_dict, dict):
                self.output_dir = Path(
                    config_dict.get("output_directory", str(self.output_dir))
                )
                self.max_concurrent_tasks = config_dict.get(
                    "max_concurrent_tasks", self.max_concurrent_tasks
                )
                self.timeout = config_dict.get("timeout", self.timeout)
                self.provider = config_dict.get("provider", self.provider)

                # Initialize provider-specific clients
                if self.provider == "openai":
                    self.client = AsyncOpenAI(
                        api_key=config_dict["openai"]["api_key"],
                        http_client=httpx.AsyncClient(verify=False),  # nosec B501 - Disabled SSL verification needed for Docker container environment
                    )
                    self.model = config_dict["openai"]["model"]
                elif self.provider == "anthropic":
                    # Anthropic SDK doesn't support custom http client configuration
                    self.client = AsyncAnthropic(
                        api_key=config_dict["anthropic"]["api_key"]
                    )
                    self.model = config_dict["anthropic"]["model"]
                elif self.provider == "google":
                    # Standard configuration for Google GenerativeAI
                    genai.configure(api_key=config_dict["google"]["api_key"])
                    self.model = config_dict["google"]["model"]
                    self.client = genai.GenerativeModel(self.model)
                else:
                    raise ValueError(f"Unsupported provider: {self.provider}")

        # Initialize thread pool with configured max tasks
        self._executor = ThreadPoolExecutor(max_workers=self.max_concurrent_tasks)
        self._processing_lock = threading.Lock()

        logger.info(
            "Initializing meeting notes plugin",
            extra={
                "plugin_name": self.name,
                "output_directory": str(self.output_dir),
                "provider": self.provider,
                "model": self.model,
            },
        )

    async def _initialize(self) -> None:
        """Initialize plugin"""
        if not self.event_bus:
            logger.warning(
                "No event bus available for plugin",
                extra={"req_id": self._req_id, "plugin_name": self.name},
            )
            return

        try:
            logger.debug(
                "Initializing meeting notes plugin",
                extra={
                    "req_id": self._req_id,
                    "plugin_name": self.name,
                    "max_workers": self.max_concurrent_tasks,
                    "model": self.model,
                    "output_dir": str(self.output_dir),
                    "provider": self.provider,
                    "model": self.model,
                },
            )

            # Create output directory
            self.output_dir.mkdir(parents=True, exist_ok=True)

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
                    "max_workers": self.max_concurrent_tasks,
                    "model": self.model,
                    "output_dir": str(self.output_dir),
                },
                exc_info=True,
            )
            raise

    async def handle_transcription_completed(self, event_data: Event | RecordingEvent) -> None:
        """Handle transcription completed event"""
        event_id = event_data.event_id
        recording_id = event_data.data.get("recording_id")
        metadata = event_data.data.get("metadata", {}) if hasattr(event_data, "data") else {}

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
            # Get transcript path
            transcript_path = await self._get_transcript_path(event_data)
            if not transcript_path:
                logger.error(
                    "No transcript path found in event",
                    extra={"plugin": self.name, "event_data": str(event_data)},
                )
                return

            # Get recording ID from event
            recording_id = None
            if isinstance(event_data, RecordingEvent):
                recording_id = event_data.recording_id
            elif hasattr(event_data, "data"):
                recording_data = event_data.data.get("recording", {})
                recording_id = recording_data.get("recording_id")

            if not recording_id:
                logger.error(
                    "No recording ID found in event",
                    extra={"plugin": self.name, "event_data": str(event_data)},
                )
                return

            # Get metadata from event
            metadata = {}
            if isinstance(event_data, dict):
                metadata = event_data.get("metadata", {})
            else:
                metadata = getattr(event_data.context, "metadata", {}) if hasattr(event_data, "context") else {}

            # Generate notes
            output_path = await self._generate_meeting_notes(
                transcript_path, str(uuid.uuid4()), recording_id
            )

            if self.event_bus:
                try:
                    completion_event = Event.create(
                        name="meeting_notes_remote.completed",
                        data={
                            "recording": event_data.data.get("recording", {}) if hasattr(event_data, "data") else {},
                            "noise_reduction": event_data.data.get("noise_reduction", {}) if hasattr(event_data, "data") else {},
                            "transcription": event_data.data.get("transcription", {}) if hasattr(event_data, "data") else {},
                            "meeting_notes_remote": {
                                "status": "completed",
                                "timestamp": dt.now(UTC).isoformat(),
                                "recording_id": recording_id,
                                "output_path": str(output_path),
                                "notes_path": str(output_path),
                                "input_paths": {
                                    "transcript": str(transcript_path),
                                },
                            },
                            "metadata": metadata,  # Pass through the metadata
                        },
                        source_plugin=self.name,
                        correlation_id=getattr(event_data.context, "correlation_id", str(uuid.uuid4())) if hasattr(event_data, "context") else str(uuid.uuid4()),
                        priority=EventPriority.NORMAL
                    )

                    await self.event_bus.publish(completion_event)
                    logger.info(
                        "Published meeting notes completion event",
                        extra={
                            "plugin": self.name,
                            "event_name": "meeting_notes_remote.completed",
                            "recording_id": recording_id,
                        },
                    )
                except Exception as e:
                    logger.error(
                        "Failed to publish completion event",
                        extra={
                            "plugin": self.name,
                            "error": str(e),
                        },
                    )
            else:
                logger.warning(
                    "No event bus available to publish completion event",
                    extra={
                        "plugin_name": self.name,
                        "event_name": "meeting_notes_remote.completed",
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
            if self.event_bus:
                try:
                    error_data = {
                        # Preserve original event data
                        "recording": event_data.data.get("recording", {}) if hasattr(event_data, "data") else {},
                        "noise_reduction": event_data.data.get("noise_reduction", {}) if hasattr(event_data, "data") else {},
                        "transcription": event_data.data.get("transcription", {}) if hasattr(event_data, "data") else {},
                        "meeting_notes_remote": {
                            "status": "error",
                            "timestamp": dt.now(UTC).isoformat(),
                            "recording_id": recording_id,
                            "error": str(e)
                        },
                        "metadata": metadata,  # Pass through the metadata
                    }

                    error_event = Event.create(
                        name="meeting_notes_remote.error",
                        data=error_data,
                        correlation_id=getattr(event_data.context, "correlation_id", None) or str(uuid.uuid4()),
                        source_plugin=self.name,
                        priority=EventPriority.NORMAL,
                    )
                    await self.event_bus.publish(error_event)
                except Exception as e:
                    logger.error(
                        "Failed to publish error event",
                        extra={
                            "plugin": self.name,
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
                }
            )
            
            # First try to get single output file
            if transcription_data.get("output_file"):
                logger.debug(
                    "Found output_file",
                    extra={
                        "plugin_name": self.name,
                        "output_file": transcription_data["output_file"]
                    }
                )
                return Path(transcription_data["output_file"])
                
            # Then try multiple transcript paths - prefer merged, then system, then mic
            transcript_paths = transcription_data.get("transcript_paths", {})
            if transcript_paths:
                logger.debug(
                    "Found transcript_paths",
                    extra={
                        "plugin_name": self.name,
                        "transcript_paths": transcript_paths
                    }
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
                "event_data": str(event)
            }
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

        except Exception as e:
            logger.error(
                "Failed to read transcript",
                extra={
                    "plugin_name": self.name,
                    "transcript_path": str(transcript_path),
                    "error": str(e),
                    "req_id": self._req_id,
                },
                exc_info=True,
            )
            raise

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

    async def _generate_notes_with_llm(
        self, transcript: str, event_id: str
    ) -> str | None:
        """Generate meeting notes using the configured LLM provider"""
        try:
            # Extract metadata section
            metadata_section = ""
            transcript_content = transcript
            if "## Metadata" in transcript:
                parts = transcript.split("## Metadata", 1)
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
                            "## Transcript"
                            + metadata_parts[2].split("## Transcript", 1)[1]
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

            system_prompt = """You are a professional meeting notes taker. Your task is to analyze the meeting transcript and create clear, concise, and well-structured meeting notes."""

            user_prompt = f"""Please analyze the following transcript and create comprehensive meeting notes in markdown format. The transcript includes METADATA in JSON format that you should use for the meeting title and information section.

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
- Date: [Use the exact meeting title from the METADATA JSON recordingStarted field, Format EXACTLY as: "January 1, 2025 at 10:00 AM"]
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
[Outline any planned next steps or future actions discussed]"""

            logger.info(
                "Sending request to LLM provider",
                extra={
                    "req_id": event_id,
                    "provider": self.provider,
                    "model": self.model,
                },
            )

            try:
                if self.provider == "openai":
                    response = await self.client.chat.completions.create(  # type: ignore
                        model=self.model,
                        messages=[
                            {"role": "system", "content": self.SYSTEM_PROMPT},
                            {"role": "user", "content": user_prompt},
                        ],
                        temperature=0,
                    )
                    content = response.choices[0].message.content
                    if not content:
                        raise ValueError("Received empty response from OpenAI")
                    return self._clean_llm_response(content)

                elif self.provider == "anthropic":
                    response = await self.client.messages.create(  # type: ignore
                        max_tokens=4000,
                        messages=[{"role": "user", "content": user_prompt}],
                        model=self.model,
                        system=self.SYSTEM_PROMPT,
                        temperature=0,
                    )
                    return self._clean_llm_response(response.content[0].text)  # type: ignore

                elif self.provider == "google":
                    response = await self.client.generate_content_async(  # type: ignore
                        f"{self.SYSTEM_PROMPT}\n\n{user_prompt}",
                        generation_config=genai.types.GenerationConfig(temperature=0),
                    )
                    return self._clean_llm_response(response.text)

            except Exception as e:
                logger.error(
                    f"Error calling {self.provider} API",
                    extra={"req_id": event_id, "error": str(e)},
                    exc_info=True,
                )
                return None

        except Exception as e:
            logger.error(
                "Error generating meeting notes",
                extra={"req_id": event_id, "error": str(e)},
                exc_info=True,
            )
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
                extra={
                    "req_id": event_id,
                    "plugin_name": self.name,
                    "error": str(e),
                    "transcript_path": str(transcript_path),
                    "recording_id": recording_id,
                },
                exc_info=True,
            )
            return None

    async def _shutdown(self) -> None:
        """Shutdown plugin"""
        try:
            if self._executor:
                self._executor.shutdown(wait=True)
            logger.info(
                "Meeting notes plugin shutdown",
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
