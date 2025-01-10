import os
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, cast
import asyncio
import aiohttp

import requests

from app.plugins.base import PluginBase
from app.plugins.events.bus import EventBus
from app.plugins.events.models import Event, EventContext, EventPriority
from app.utils.logging_config import get_logger
from app.models.recording.events import RecordingEvent, EventContext

EventData = dict[str, Any] | RecordingEvent

logger = get_logger(__name__)


class MeetingNotesRemotePlugin(PluginBase):
    """Plugin for generating meeting notes from transcripts using Ollama LLM.
    Listens for transcript_ready events and generates structured meeting notes."""

    def __init__(self, config: Any, event_bus: EventBus | None = None) -> None:
        """Initialize the plugin"""
        super().__init__(config, event_bus)
        self._req_id = str(uuid.uuid4())

        # Check if we're running in Docker first
        is_docker = os.path.exists('/.dockerenv')
        
        # Default values
        self.ollama_url = "http://ollama:11434/api/generate" if is_docker else "http://localhost:11434/api/generate"
        self.model = "llama3.1:latest"
        self.output_dir = Path("data/meeting_notes_remote")
        self.num_ctx = 128000
        self.max_concurrent_tasks = 4
        self.timeout = 300  # Default timeout of 5 minutes

        # Override with config values if available
        if config and hasattr(config, "config"):
            config_dict = config.config
            if isinstance(config_dict, dict):
                # Don't override ollama_url if we're in Docker
                if not is_docker:
                    self.ollama_url = config_dict.get("ollama_url", self.ollama_url)
                self.model = config_dict.get("model_name", self.model)
                self.output_dir = Path(config_dict.get("output_directory", str(self.output_dir)))
                self.num_ctx = config_dict.get("num_ctx", self.num_ctx)
                self.max_concurrent_tasks = config_dict.get("max_concurrent_tasks", self.max_concurrent_tasks)
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
                "is_docker": is_docker
            }
        )

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
            logger.debug(
                "Initializing meeting notes plugin",
                extra={
                    "req_id": self._req_id,
                    "plugin_name": self.name,
                    "max_workers": self.max_concurrent_tasks,
                    "model": self.model,
                    "output_dir": str(self.output_dir),
                    "ollama_url": self.ollama_url,
                    "num_ctx": self.num_ctx
                }
            )
            
            # Subscribe to transcription completed event
            await self.event_bus.subscribe(
                "transcription_local.completed",
                self.handle_transcription_completed
            )
            
            logger.info(
                "Meeting notes plugin initialized",
                extra={
                    "req_id": self._req_id,
                    "plugin_name": self.name,
                    "max_workers": self.max_concurrent_tasks,
                    "model": self.model,
                    "event": "transcription_local.completed",
                    "output_dir": str(self.output_dir)
                }
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
                    "output_dir": str(self.output_dir)
                },
                exc_info=True
            )
            raise

    async def handle_transcription_completed(self, event_data: EventData) -> None:
        """Handle transcription completed event"""
        try:
            event_id = str(uuid.uuid4())
            
            logger.info(
                "Processing transcription completed event",
                extra={
                    "plugin_name": self.name,
                    "event_id": getattr(event_data, "event_id", None)
                }
            )

            # Get transcript path and recording ID
            transcript_path = event_data.get("transcript_path") if isinstance(event_data, dict) else getattr(event_data, "transcript_path", None)
            recording_id = event_data.get("recording_id") if isinstance(event_data, dict) else getattr(event_data, "recording_id", None)

            if not transcript_path:
                logger.warning("No transcript path in event", extra={"plugin_name": self.name})
                return

            # Generate meeting notes
            output_path = await self._generate_meeting_notes(Path(transcript_path), event_id, recording_id)
            if not output_path:
                return

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
            completion_event = {
                "event": "meeting_notes_remote.completed",
                "recording_id": recording_id,
                "output_path": str(output_path),
                "notes_path": str(output_path),
                "status": "completed",
                "timestamp": datetime.utcnow().isoformat(),
                "plugin_id": self.name,
                "data": {
                    "current_event": {
                        "meeting_notes": {
                            "status": "completed",
                            "timestamp": datetime.utcnow().isoformat(),
                            "output_path": str(output_path)
                        }
                    }
                }
            }

            logger.debug(
                "Publishing completion event",
                extra={
                    "plugin": self.name,
                    "event_name": completion_event["event"],
                    "recording_id": recording_id,
                    "output_path": str(output_path)
                }
            )
            await self.event_bus.publish(completion_event)

            # Add verification log after publishing
            logger.info(
                "Meeting notes completion event published",
                extra={
                    "plugin_name": self.name,
                    "event_name": completion_event["event"],
                    "recording_id": recording_id,
                    "output_path": str(output_path)
                }
            )

        except Exception as e:
            logger.error(
                "Error processing transcription event",
                extra={
                    "plugin_name": self.name,
                    "error": str(e)
                },
                exc_info=True
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
                "transcript_length": len(transcript_text)
            }
        )

        # Extract metadata section
        metadata_section = ""
        transcript_content = transcript_text
        if "## Metadata" in transcript_text:
            parts = transcript_text.split("## Metadata", 1)
            if len(parts) > 1:
                metadata_parts = parts[1].split("```", 3)  # Split into 3 parts to handle both json and transcript sections
                if len(metadata_parts) > 2:
                    metadata_section = metadata_parts[1].replace('json', '').strip()  # Get the JSON content
                    # Get the transcript section after metadata
                    transcript_content = "## Transcript" + metadata_parts[2].split("## Transcript", 1)[1] if "## Transcript" in metadata_parts[2] else ""

        logger.debug(
            "Extracted metadata and transcript",
            extra={
                "plugin_name": self.name,
                "has_metadata": bool(metadata_section),
                "metadata_length": len(metadata_section),
                "transcript_length": len(transcript_content)
            }
        )

        # Prepare prompt with explicit metadata handling
        prompt = f"""Please analyze the following transcript and create comprehensive meeting notes in markdown format.
The transcript includes METADATA in JSON format that you should use for the meeting title and information section.

START Transcript:

{transcript_content}

END Transcript

START METADATA JSON:

{metadata_section}

END METADATA JSON

Create meeting notes with the following sections:

# Meeting Title
[The meeting title - extracted from the METADATA JSON event.title field]

## Meeting Information
- Date: [Format the METADATA JSON event.started as a readable date/time. Example format: "January 1, 2025 at 10:00 AM"]
- Duration: [The final timestamp entry in the transcript in an hours and minutes format]
- Attendees: [a bulleted list of ALL email addresses from the METADATA JSON event.attendees array (NOT from the speakers list)]

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
[Bulleted list of action items in the format of '(OWNER) ACTION ITEM DESCRIPTION [DEADLINE IF MENTIONED'. Identify the owner from the context of the meeting transcript]

## Decisions Made
[List specific decisions or conclusions reached during the meeting]

## Next Steps
[Outline any planned next steps or future actions discussed]

Please ensure the notes are clear, concise, and well-organized using markdown formatting.
IMPORTANT: 
1. Do not use placeholders - extract and use the actual values from the METADATA JSON and the transcript.
2. For attendees, use ONLY the email addresses or names from event.attendees in the METADATA JSON, not the speakers list
3. Don't include any other information in the notes, just the meeting notes
"""

        logger.debug(
            "Generated prompt for meeting notes",
            extra={
                "plugin_name": self.name,
                "prompt": prompt
            }
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
                    "timestamp": datetime.utcnow().isoformat()
                }
            )

            async with aiohttp.ClientSession() as session:
                start_time = datetime.utcnow()
                async with session.post(
                    self.ollama_url,
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "num_ctx": self.num_ctx
                        }
                    },
                    timeout=aiohttp.ClientTimeout(total=self.timeout)
                ) as response:
                    response.raise_for_status()
                    result = await response.json()
                    end_time = datetime.utcnow()
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
                            "timestamp": end_time.isoformat()
                        }
                    )

                    return cast(str, result.get("response", ""))

        except aiohttp.ClientError as e:
            logger.error(
                "Network error while calling Ollama",
                extra={
                    "plugin_name": self.name,
                    "error_type": type(e).__name__,
                    "error": str(e),
                    "model": self.model,
                    "ollama_url": self.ollama_url,
                    "num_ctx": self.num_ctx
                },
                exc_info=True
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
                    "model": self.model
                },
                exc_info=True
            )
            return f"Error generating meeting notes: {e}"

    async def _process_transcript(
        self, recording_id: str, transcript_text: str, original_event: Event
    ) -> None:
        """Process transcript and generate meeting notes"""
        try:
            # Generate meeting notes synchronously since the API call is blocking
            meeting_notes = await self._generate_meeting_notes_from_text(transcript_text)

            # Save to file
            output_file = self.output_dir / f"{recording_id}_notes.md"
            output_file.write_text(meeting_notes)

            # Emit completion event
            if self.event_bus:
                from datetime import datetime

                event_data = {
                    "recording_id": recording_id,
                    "output_path": str(output_file),
                    "notes_path": str(output_file),
                    "status": "completed",
                    "timestamp": datetime.utcnow().isoformat(),
                    "current_event": {
                        "meeting_notes": {
                            "status": "completed",
                            "timestamp": datetime.utcnow().isoformat(),
                            "output_path": str(output_file)
                        }
                    },
                    "event_history": {},
                    "transcription": original_event.payload.get("transcription", {}),
                    "noise_reduction": original_event.payload.get("noise_reduction", {}),
                    "recording": original_event.payload.get("recording", {})
                }

                event = Event(
                    event_id=str(uuid.uuid4()),
                    event="meeting_notes_remote.completed",
                    data=event_data,
                    context=EventContext(
                        correlation_id=str(uuid.uuid4()),
                        timestamp=datetime.utcnow().isoformat(),
                        source_plugin=self.name
                    )
                )
                
                await self.event_bus.publish(event)
                
                logger.info(
                    "Meeting notes completion event published",
                    extra={
                        "plugin_name": self.name,
                        "event_name": "meeting_notes_remote.completed",
                        "recording_id": recording_id,
                        "output_path": str(output_file)
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
                    "error": str(e),
                    "event_id": original_event.event_id
                },
                exc_info=True
            )
            
            if self.event_bus:
                # Emit error event with preserved chain
                event = Event(
                    event_id=str(uuid.uuid4()),
                    name="meeting_notes.error",
                    event_data={
                        "recording_id": recording_id,
                        "meeting_notes": {
                            "status": "error",
                            "timestamp": datetime.utcnow().isoformat(),
                            "error": str(e)
                        },
                        # Preserve previous event data
                        "transcription": original_event.payload.get("transcription", {}),
                        "noise_reduction": original_event.payload.get("noise_reduction", {}),
                        "recording": original_event.payload.get("recording", {})
                    },
                    context=EventContext(
                        recording_id=recording_id,
                        plugin_name=self.name,
                        req_id=self._req_id
                    )
                )
                await self.event_bus.publish(event)

    async def _get_transcript_path(self, event: Event | RecordingEvent) -> Path | None:
        """Extract transcript path from event."""
        # Handle RecordingEvent type
        if isinstance(event, RecordingEvent):
            if event.output_file:
                return Path(event.output_file)
            return None
            
        # Handle generic Event type
        if hasattr(event, 'payload') and isinstance(event.payload, dict):
            transcript_path = event.payload.get('transcript_path') or event.payload.get('output_file')
            if transcript_path:
                return Path(transcript_path)
        
        return None

    async def _read_transcript(self, transcript_path: str | Path) -> str:
        """Read transcript file contents"""
        try:
            # Convert string path to Path object if needed
            path = Path(transcript_path) if isinstance(transcript_path, str) else transcript_path
            
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
                    "req_id": self._req_id
                },
                exc_info=True
            )
            raise

    async def _generate_notes_with_llm(self, transcript: str, event_id: str) -> str | None:
        """Generate notes using LLM from transcript text."""
        try:
            return await self._generate_meeting_notes_from_text(transcript)
        except Exception as e:
            logger.error(
                "Failed to generate notes with LLM: %s",
                str(e),
                extra={
                    "plugin_name": self.name,
                    "error": str(e),
                    "event_id": event_id,
                    "transcript_length": len(transcript)
                },
                exc_info=True
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
            transcript = await self._read_transcript(transcript_path)
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
                    "error": str(e),
                    "transcript_path": str(transcript_path),
                    "recording_id": recording_id
                },
                exc_info=True
            )
            return None

    async def _shutdown(self) -> None:
        """Shutdown plugin"""
        try:
            if self._executor:
                self._executor.shutdown(wait=True)
            logger.info(
                "Meeting notes plugin shutdown",
                extra={
                    "req_id": self._req_id,
                    "plugin_name": self.name
                }
            )
        except Exception as e:
            logger.error(
                "Error during plugin shutdown",
                extra={
                    "req_id": self._req_id,
                    "plugin_name": self.name,
                    "error": str(e)
                },
                exc_info=True
            ) 