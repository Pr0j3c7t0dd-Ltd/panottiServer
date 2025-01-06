"""Audio transcription plugin for converting audio to text."""

import json
import logging
import os
import threading
import wave
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from faster_whisper import WhisperModel
from pathlib import Path
from typing import Any
import uuid
import asyncio

from app.models.database import DatabaseManager
from app.models.recording.events import RecordingEvent
from app.plugins.base import PluginBase, PluginConfig
from app.plugins.events.models import EventContext
from app.utils.logging_config import get_logger

logger = get_logger("app.plugins.audio_transcription.plugin")

EventData = dict[str, Any] | RecordingEvent


class AudioTranscriptionPlugin(PluginBase):
    """Plugin for transcribing audio files using Whisper"""

    def __init__(self, config: Any, event_bus: Any | None = None) -> None:
        super().__init__(config, event_bus)
        self._executor = ThreadPoolExecutor(max_workers=4)
        self._processing_lock = threading.Lock()
        self._db_initialized = False
        self._model = None
        self._output_dir = Path(os.getenv("TRANSCRIPTS_DIR", "data/transcripts"))
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._shutdown_event = asyncio.Event()

    async def _initialize(self) -> None:
        """Initialize plugin."""
        if not self.event_bus:
            logger.warning("No event bus available for plugin")
            return

        try:
            logger.debug(
                "Initializing audio transcription plugin",
                extra={
                    "plugin": self.name,
                    "output_dir": str(self._output_dir)
                }
            )

            # Initialize Whisper model
            logger.info("Initializing Whisper model")
            self._init_model()

            # Initialize database
            self.db = await DatabaseManager.get_instance()

            # Subscribe to noise reduction completed event
            await self.event_bus.subscribe(
                "noise_reduction.completed",
                self.handle_noise_reduction_completed
            )

            logger.info(
                "Audio transcription plugin initialized",
                extra={
                    "plugin": self.name,
                    "subscribed_events": ["noise_reduction.completed"],
                    "handler": "handle_noise_reduction_completed"
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

    async def _shutdown(self) -> None:
        """Shutdown plugin"""
        try:
            logger.info(
                "Shutting down audio transcription plugin",
                extra={
                    "plugin": self.name
                }
            )

            # Signal shutdown
            self._shutdown_event.set()

            # Unsubscribe from events first
            if self.event_bus:
                logger.debug("Unsubscribing from events", extra={"plugin": self.name})
                await self.event_bus.unsubscribe(
                    "noise_reduction.completed", self.handle_noise_reduction_completed
                )

            # Wait for any ongoing processing to complete
            async with asyncio.timeout(5):  # 5 second timeout
                while self._processing_lock.locked():
                    await asyncio.sleep(0.1)

            # Shutdown thread pool
            if self._executor:
                logger.debug("Shutting down thread pool", extra={"plugin": self.name})
                try:
                    self._executor.shutdown(wait=True, cancel_futures=True)
                except Exception as e:
                    logger.error(
                        "Error shutting down thread pool",
                        extra={
                            "plugin": self.name,
                            "error": str(e)
                        }
                    )
                finally:
                    self._executor = None

            # Release model resources
            if self._model:
                logger.debug("Releasing model resources", extra={"plugin": self.name})
                try:
                    # Force garbage collection of the model
                    model = self._model
                    self._model = None
                    del model
                except Exception as e:
                    logger.error(
                        "Error releasing model resources",
                        extra={
                            "plugin": self.name,
                            "error": str(e)
                        }
                    )

            logger.info(
                "Audio transcription plugin shutdown complete",
                extra={
                    "plugin": self.name
                }
            )

        except asyncio.TimeoutError:
            logger.error(
                "Timeout waiting for processing to complete during shutdown",
                extra={"plugin": self.name}
            )
        except Exception as e:
            logger.error(
                "Error during plugin shutdown",
                extra={
                    "plugin": self.name,
                    "error": str(e)
                },
                exc_info=True
            )

    async def handle_noise_reduction_completed(self, event_data: EventData) -> None:
        """Handle noise reduction completed event"""
        try:
            # Extract event type
            if isinstance(event_data, dict):
                event_type = event_data.get("event")
            else:
                event_type = getattr(event_data, "event", None)

            logger.debug(
                "Received noise reduction completed event",
                extra={
                    "plugin": self.name,
                    "event_type": event_type,
                    "event_data": str(event_data)
                }
            )

            if event_type != "noise_reduction.completed":
                logger.debug(
                    "Ignoring non-noise_reduction.completed event",
                    extra={
                        "plugin": self.name,
                        "event_type": event_type
                    }
                )
                return

            # Extract paths and metadata from event data
            if isinstance(event_data, dict):
                recording_id = event_data.get("recording_id")
                processed_mic_audio = event_data.get("output_path")
                
                # Get original event data
                original_event = None
                if "original_event" in event_data:
                    original_event = event_data["original_event"]
                elif "data" in event_data and "current_event" in event_data["data"]:
                    original_event = event_data["data"]["current_event"]
                
                logger.debug(
                    "Original event data",
                    extra={
                        "plugin": self.name,
                        "original_event": str(original_event)
                    }
                )
                
                # Extract system audio path
                original_sys_audio = None
                if original_event and "recording" in original_event:
                    original_sys_audio = original_event["recording"]["audio_paths"]["system"]
                elif "system_audio_path" in event_data:
                    original_sys_audio = event_data["system_audio_path"]
                
                # Extract metadata
                metadata = {}
                if original_event and "recording" in original_event:
                    metadata = original_event["recording"]["metadata"]
                elif "metadata" in event_data:
                    metadata = event_data["metadata"]
            else:
                recording_id = event_data.recording_id
                processed_mic_audio = getattr(event_data, "output_path", None)
                
                # Get original event data
                original_event = None
                if hasattr(event_data, "data") and hasattr(event_data.data, "current_event"):
                    original_event = event_data.data.current_event
                
                logger.debug(
                    "Original event data (object)",
                    extra={
                        "plugin": self.name,
                        "original_event": str(original_event)
                    }
                )
                
                # Extract system audio path
                original_sys_audio = getattr(event_data, "system_audio_path", None)
                if original_event and hasattr(original_event, "recording"):
                    original_sys_audio = original_event.recording.audio_paths.get("system", original_sys_audio)
                
                # Extract metadata
                metadata = getattr(event_data, "metadata", {})
                if original_event and hasattr(original_event, "recording"):
                    metadata = original_event.recording.metadata

            if not recording_id or not processed_mic_audio:
                logger.error(
                    "Missing required data in noise reduction event",
                    extra={
                        "plugin": self.name,
                        "event_data": str(event_data)
                    }
                )
                return

            logger.info(
                "Processing audio files",
                extra={
                    "plugin": self.name,
                    "recording_id": recording_id,
                    "processed_mic_audio": processed_mic_audio,
                    "original_sys_audio": original_sys_audio,
                    "metadata": metadata
                }
            )

            # Process both audio files
            mic_transcript = await self._process_audio(processed_mic_audio)
            sys_transcript = None
            if original_sys_audio and os.path.exists(original_sys_audio):
                logger.info(
                    "Processing system audio",
                    extra={
                        "plugin": self.name,
                        "system_audio_path": original_sys_audio
                    }
                )
                sys_transcript = await self._process_audio(original_sys_audio)
            else:
                logger.warning(
                    "System audio not found or not accessible",
                    extra={
                        "plugin": self.name,
                        "system_audio_path": original_sys_audio,
                        "exists": original_sys_audio and os.path.exists(original_sys_audio)
                    }
                )

            # Create merged transcript
            output_base = self._output_dir / f"{recording_id}_merged"
            merged_transcript = output_base.with_suffix(".txt")

            # Get labels from metadata
            mic_label = metadata.get("microphoneLabel", "Microphone")
            sys_label = metadata.get("systemLabel", "System")

            logger.debug(
                "Merging transcripts",
                extra={
                    "plugin": self.name,
                    "mic_transcript": str(mic_transcript),
                    "sys_transcript": str(sys_transcript),
                    "mic_label": mic_label,
                    "sys_label": sys_label,
                    "metadata": metadata
                }
            )

            # Merge transcripts
            transcript_files = [str(mic_transcript)]
            labels = [mic_label]
            if sys_transcript:
                transcript_files.append(str(sys_transcript))
                labels.append(sys_label)

            self.merge_transcripts(
                transcript_files=transcript_files,
                output_path=str(merged_transcript),
                labels=labels,
                original_event=metadata
            )

            # Create and emit event
            if self.event_bus:
                await self.event_bus.publish({
                    "event": "transcription.completed",
                    "recording_id": recording_id,
                    "output_path": str(merged_transcript),
                    "input_paths": {
                        "microphone": processed_mic_audio,
                        "system": original_sys_audio
                    },
                    "transcript_paths": {
                        "microphone": str(mic_transcript),
                        "system": str(sys_transcript) if sys_transcript else None,
                        "merged": str(merged_transcript)
                    },
                    "event_id": f"{recording_id}_transcription_completed",
                    "timestamp": datetime.utcnow().isoformat(),
                    "plugin_id": self.name,
                    "metadata": metadata
                })

        except Exception as e:
            logger.error(
                "Error handling noise reduction completed event",
                extra={
                    "plugin": self.name,
                    "error": str(e)
                },
                exc_info=True
            )

    async def _emit_transcription_event(
        self,
        recording_id: str,
        status: str,
        output_file: str | None = None,
        error: str | None = None,
        original_event: dict | None = None,
    ) -> None:
        """Emit transcription event with enriched data chain"""
        if not self.event_bus:
            return

        # Structure new event data with nested history
        event_data = {
            "recording_id": recording_id,
            "transcription": {
                "status": status,
                "timestamp": datetime.utcnow().isoformat(),
                "output_paths": {
                    "transcript": output_file
                } if output_file else None,
                "error": error
            }
        }

        # Preserve previous event data in nested structure
        if original_event:
            if "noise_reduction" in original_event:
                event_data["noise_reduction"] = original_event["noise_reduction"]
            if "recording" in original_event:
                event_data["recording"] = original_event["recording"]

        # Create and emit event
        event = RecordingEvent(
            recording_timestamp=datetime.utcnow().isoformat(),
            recording_id=recording_id,
            event="transcription.completed",
            data=event_data,
            context=original_event.get("context") if original_event else None
        )

        await self.event_bus.publish(event)
        
        # Structured logging
        log_data = {
            "plugin": self.name,
            "recording_id": recording_id,
            "event": "transcription.completed",
            "status": status
        }
        if output_file:
            log_data["output_file"] = output_file
        if error:
            log_data["error"] = error
            
        logger.info("Emitted transcription event", extra=log_data)

    async def _init_database(self) -> None:
        """Initialize database tables."""
        if not self.db:
            return

        # Create tables using the connection from our db instance
        with self.db.get_connection() as conn:  # Now this will work
            conn.execute("""
                CREATE TABLE IF NOT EXISTS transcriptions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    recording_id TEXT NOT NULL,
                    transcript TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (recording_id) REFERENCES recordings(recording_id)
                )
            """)
            conn.commit()

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
        """Merge multiple transcript files ordered by timestamp"""
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Read all transcripts and parse timestamps
        segments = []
        for transcript_file, label in zip(transcript_files, labels, strict=False):
            with open(transcript_file) as tf:
                for line in tf:
                    if line.strip() and "[" in line and "]" in line:
                        try:
                            # Parse timestamp and text
                            time_part = line[line.find("[")+1:line.find("]")]
                            start_time = float(time_part.split("->")[0].strip()[:-1])  # Remove 's' suffix
                            text = line[line.find("]")+1:].strip()
                            segments.append((start_time, label, text))
                        except (ValueError, IndexError) as e:
                            logger.warning(
                                f"Failed to parse line in transcript: {line}",
                                extra={
                                    "plugin": self.name,
                                    "error": str(e),
                                    "line": line
                                }
                            )

        # Sort segments by timestamp
        segments.sort(key=lambda x: x[0])

        # Write merged transcript to file
        with open(output_path, "w") as f:
            # Add metadata header
            f.write("# Merged Transcript\n\n")
            f.write("## Metadata\n")
            
            # Add event metadata
            if "eventTitle" in original_event:
                f.write(f"- Event: {original_event['eventTitle']}\n")
            if "eventProvider" in original_event:
                f.write(f"- Source: {original_event['eventProvider']}\n")
            if "eventProviderId" in original_event:
                f.write(f"- Event ID: {original_event['eventProviderId']}\n")
            if "recordingStarted" in original_event:
                f.write(f"- Started: {original_event['recordingStarted']}\n")
            if "recordingEnded" in original_event:
                f.write(f"- Ended: {original_event['recordingEnded']}\n")
            if "eventAttendees" in original_event:
                f.write("- Attendees:\n")
                for attendee in original_event["eventAttendees"]:
                    f.write(f"  - {attendee}\n")
            f.write("\n")

            # Write chronologically ordered segments
            f.write("## Transcript\n\n")
            for start_time, label, text in segments:
                minutes = int(start_time // 60)
                seconds = start_time % 60
                f.write(f"[{minutes:02d}:{seconds:05.2f}] {label}: {text}\n")

    async def _process_audio(self, audio_path: str) -> Path:
        """Process audio file and return transcript path"""
        # Create output paths
        output_base = self._output_dir / Path(audio_path).stem
        transcript_file = output_base.with_suffix(".txt")
        
        # Process audio file here...
        if self._model is None:
            self._init_model()
            
        # Transcribe the audio
        segments, _ = self._model.transcribe(audio_path)
        
        # Write transcript to file
        transcript_file.parent.mkdir(parents=True, exist_ok=True)
        with open(transcript_file, "w") as f:
            for segment in segments:
                f.write(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}\n")
        
        return transcript_file

    def _init_model(self) -> None:
        """Initialize the faster-whisper model."""
        try:
            # Get model configuration
            config_dict = self.config.config or {}
            model_name = config_dict.get("model_name", "base.en")
            
            # Get project root directory
            current_dir = Path(__file__).resolve().parent
            project_root = current_dir.parent.parent.parent
            model_dir = project_root / "models" / "whisper"
            model_dir.mkdir(parents=True, exist_ok=True)
            
            # Initialize the model
            self._model = WhisperModel(
                model_size_or_path=str(model_dir),
                device="cpu",
                device_index=0,
                compute_type="default"
            )

            logger.info(
                "Faster-Whisper model initialized",
                extra={
                    "plugin": self.name,
                    "model_name": model_name,
                    "model_dir": str(model_dir)
                }
            )

        except Exception as e:
            logger.error(
                "Failed to initialize Whisper model",
                extra={
                    "plugin": self.name,
                    "error": str(e)
                },
                exc_info=True
            )
            raise
