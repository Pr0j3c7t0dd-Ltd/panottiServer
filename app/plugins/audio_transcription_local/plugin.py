"""Audio transcription plugin for converting audio to text locally."""

import asyncio
import json
import os
import threading
import wave
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timezone
from pathlib import Path
from typing import Any
import uuid

import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel

from app.core.events.types import EventContext
from app.core.plugins import PluginBase
from app.models.database import DatabaseManager
from app.models.recording.events import RecordingEvent
from app.plugins.audio_transcription_local.transcript_cleaner import TranscriptCleaner
from app.utils.logging_config import get_logger
from app.core.events import Event, EventPriority

logger = get_logger("app.plugins.audio_transcription_local.plugin")

EventData = dict[str, Any] | RecordingEvent


class AudioTranscriptionLocalPlugin(PluginBase):
    """Plugin for transcribing audio files using Whisper locally"""

    def __init__(self, config: Any, event_bus: Any | None = None) -> None:
        super().__init__(config, event_bus)
        self._executor = ThreadPoolExecutor(max_workers=4)
        self._processing_lock = threading.Lock()
        self._db_initialized = False
        self._model = None
        self._output_dir = Path(os.getenv("TRANSCRIPTS_DIR", "data/transcripts_local"))
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._shutdown_event = asyncio.Event()
        self._transcript_cleaner = TranscriptCleaner()

    async def _initialize(self) -> None:
        """Initialize plugin."""
        if not self.event_bus:
            logger.warning("No event bus available for plugin")
            return

        try:
            logger.debug(
                "Initializing audio transcription plugin",
                extra={"plugin": self.name, "output_dir": str(self._output_dir)},
            )

            # Initialize Whisper model
            logger.info("Initializing Whisper model")
            self._init_model()

            # Initialize database
            self.db = await DatabaseManager.get_instance()

            # Subscribe to noise reduction completed event
            await self.event_bus.subscribe(
                "noise_reduction.completed", self.handle_noise_reduction_completed
            )

            logger.info(
                "Audio transcription plugin initialized",
                extra={
                    "plugin": self.name,
                    "subscribed_events": ["noise_reduction.completed"],
                    "handler": "handle_noise_reduction_completed",
                },
            )

        except Exception as e:
            logger.error(
                "Failed to initialize plugin",
                extra={"plugin": self.name, "error": str(e)},
                exc_info=True,
            )
            raise

    async def _shutdown(self) -> None:
        """Shutdown plugin"""
        try:
            logger.info(
                "Shutting down audio transcription plugin", extra={"plugin": self.name}
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
                        extra={"plugin": self.name, "error": str(e)},
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
                        extra={"plugin": self.name, "error": str(e)},
                    )

            logger.info(
                "Audio transcription plugin shutdown complete",
                extra={"plugin": self.name},
            )

        except asyncio.TimeoutError:
            logger.error(
                "Timeout waiting for processing to complete during shutdown",
                extra={"plugin": self.name},
            )
        except Exception as e:
            logger.error(
                "Error during plugin shutdown",
                extra={"plugin": self.name, "error": str(e)},
                exc_info=True,
            )

    async def handle_noise_reduction_completed(self, event_data: EventData) -> None:
        """Handle noise reduction completed event"""
        try:
            # Extract event type
            if isinstance(event_data, dict):
                event_type = event_data.get("name")
            else:
                event_type = getattr(event_data, "name", None)

            logger.debug(
                "Received noise reduction completed event",
                extra={
                    "plugin": self.name,
                    "event_type": event_type,
                    "event_data": str(event_data),
                },
            )

            if event_type != "noise_reduction.completed":
                logger.debug(
                    "Ignoring non-noise_reduction.completed event",
                    extra={"plugin": self.name, "event_type": event_type},
                )
                return

            # Handle both dictionary and object access patterns
            if isinstance(event_data, dict):
                data = event_data.get("data", {})
            elif hasattr(event_data, "data"):
                data = event_data.data
            else:
                logger.error(
                    "Invalid event data format",
                    extra={
                        "plugin": self.name,
                        "event_data": str(event_data),
                    },
                )
                return

            noise_reduction_data = data.get("noise_reduction", {})
            recording_data = data.get("recording", {})
            metadata = data.get("metadata", {})
            context = data.get("context", {})

            # Combine metadata from both sources
            combined_metadata = {}
            combined_metadata.update(metadata)
            combined_metadata.update(context.get("metadata", {}))

            # Get recording ID and audio paths
            recording_id = noise_reduction_data.get("recording_id")
            processed_mic_audio = noise_reduction_data.get("output_path")
            original_sys_audio = noise_reduction_data.get("system_audio_path")

            # Get speaker labels from metadata
            speaker_labels = combined_metadata.get("speaker_labels", {})
            if not speaker_labels:
                speaker_labels = {
                    "microphone": combined_metadata.get("microphoneLabel"),
                    "system": combined_metadata.get("systemLabel")
                }

            mic_label = speaker_labels.get("microphone", "Microphone")
            sys_label = speaker_labels.get("system", "System")

            logger.debug(
                "Processing event data",
                extra={
                    "plugin": self.name,
                    "recording_id": recording_id,
                    "processed_mic_audio": processed_mic_audio,
                    "original_sys_audio": original_sys_audio,
                    "speaker_labels": speaker_labels,
                    "metadata": json.dumps(combined_metadata),
                },
            )

            # Validate required data
            if not recording_id or not processed_mic_audio:
                logger.error(
                    "Missing required data in noise reduction event",
                    extra={
                        "plugin": self.name,
                        "recording_id": recording_id,
                        "processed_mic_audio": processed_mic_audio,
                        "original_sys_audio": original_sys_audio,
                        "noise_reduction_data": str(noise_reduction_data),
                    },
                )
                return

            logger.info(
                "Processing audio files",
                extra={
                    "plugin": self.name,
                    "recording_id": recording_id,
                    "processed_mic_audio": processed_mic_audio,
                    "original_sys_audio": original_sys_audio,
                },
            )

            # Process both audio files with speaker labels
            mic_transcript = await self._process_audio(processed_mic_audio, mic_label, combined_metadata) if processed_mic_audio else None
            sys_transcript = await self._process_audio(original_sys_audio, sys_label, combined_metadata) if original_sys_audio else None

            if original_sys_audio:
                if os.path.exists(original_sys_audio):
                    logger.info(
                        "Processing system audio",
                        extra={
                            "plugin": self.name,
                            "system_audio_path": original_sys_audio,
                        },
                    )
                    try:
                        sys_transcript = await self._process_audio(
                            original_sys_audio, sys_label, combined_metadata
                        )
                        if not sys_transcript:
                            logger.error(
                                "Failed to process system audio",
                                extra={
                                    "plugin": self.name,
                                    "system_audio_path": original_sys_audio,
                                }
                            )
                    except Exception as e:
                        logger.error(
                            "Error processing system audio",
                            extra={
                                "plugin": self.name,
                                "system_audio_path": original_sys_audio,
                                "error": str(e)
                            },
                            exc_info=True
                        )
                else:
                    logger.error(
                        "System audio file not found",
                        extra={
                            "plugin": self.name,
                            "system_audio_path": original_sys_audio,
                        }
                    )
            else:
                logger.warning(
                    "No system audio path found in event",
                    extra={
                        "plugin": self.name,
                        "event_data": str(event_data)
                    }
                )

            # Create merged transcript
            output_base = self._output_dir / f"{recording_id}_merged"
            merged_transcript = output_base.with_suffix(".md")

            logger.debug(
                "Merging transcripts",
                extra={
                    "plugin": self.name,
                    "mic_transcript": str(mic_transcript),
                    "sys_transcript": str(sys_transcript),
                },
            )

            # Get metadata and speaker labels
            labels = []

            if mic_transcript:
                labels.append(mic_label)
            if sys_transcript:
                labels.append(sys_label)

            if len(labels) > 0:
                # Filter out None values from transcript list
                transcript_files = [f for f in [str(mic_transcript) if mic_transcript else None, str(sys_transcript) if sys_transcript else None] if f is not None]
                await self.merge_transcripts(
                    transcript_files,
                    str(merged_transcript), 
                    labels, 
                    event_data if isinstance(event_data, dict) else event_data.dict()
                )

                # Emit completion event with all transcript paths
                await self._emit_transcription_event(
                    recording_id=recording_id,
                    status="completed",
                    output_file=str(merged_transcript),
                    original_event=event_data if isinstance(event_data, dict) else event_data.dict(),
                    transcript_paths={
                        "mic": str(mic_transcript) if mic_transcript else None,
                        "system": str(sys_transcript) if sys_transcript else None,
                        "merged": str(merged_transcript)
                    }
                )
        except Exception as e:
            logger.error(
                "Error handling noise reduction completed event",
                extra={"plugin": self.name, "error": str(e)},
                exc_info=True,
            )

    async def _emit_transcription_event(
        self,
        recording_id: str,
        status: str,
        output_file: str | None = None,
        error: str | None = None,
        original_event: dict | None = None,
        transcript_paths: dict[str, str | None] | None = None,
    ) -> None:
        """Emit transcription event."""
        if not self.event_bus:
            logger.warning("No event bus available to publish transcription event")
            return

        # Extract metadata and labels from original event
        combined_metadata = {}
        
        # Extract metadata from original event
        if original_event and isinstance(original_event, dict):
            # Direct metadata from event
            event_metadata = original_event.get("metadata", {})
            if isinstance(event_metadata, dict):
                combined_metadata.update(event_metadata)
            
            # Context metadata
            context = original_event.get("context", {})
            if isinstance(context, dict):
                context_metadata = context.get("metadata", {})
                if isinstance(context_metadata, dict):
                    combined_metadata.update(context_metadata)

        # Log metadata for debugging
        logger.debug(
            "Processing metadata from event",
            extra={
                "plugin": self.name,
                "recording_id": recording_id,
                "event_metadata": str(event_metadata),
                "original_event_metadata": str(original_event),
                "combined_metadata": str(combined_metadata)
            }
        )

        # Get speaker labels and correlation ID
        speaker_label = combined_metadata.get("microphoneLabel", "Microphone")
        system_label = combined_metadata.get("systemLabel", "System")
        correlation_id = str(original_event.get("correlation_id", "")) if original_event else str(uuid.uuid4())

        try:
            if error:
                event = Event.create(
                    name="transcription_local.error",
                    data={
                        "recording": original_event.get("recording", {}) if original_event else {},
                        "noise_reduction": original_event.get("noise_reduction", {}) if original_event else {},
                        "transcription": {
                            "status": "error",
                            "timestamp": datetime.now(UTC).isoformat(),
                            "recording_id": recording_id,
                            "error": error,
                            "config": {
                                "model": self.get_config("model", "base"),
                                "language": self.get_config("language", "en"),
                                "speaker_labels": {
                                    "microphone": speaker_label,
                                    "system": system_label
                                }
                            }
                        },
                        "metadata": combined_metadata,
                        "context": {
                            "correlation_id": correlation_id,
                            "source_plugin": self.name,
                            "metadata": combined_metadata
                        }
                    },
                    correlation_id=correlation_id,
                    source_plugin=self.name,
                    priority=EventPriority.NORMAL
                )
            else:
                event = Event.create(
                    name="transcription_local.completed",
                    data={
                        "recording": original_event.get("recording", {}) if original_event else {},
                        "noise_reduction": original_event.get("noise_reduction", {}) if original_event else {},
                        "transcription": {
                            "status": status,
                            "timestamp": datetime.now(UTC).isoformat(),
                            "recording_id": recording_id,
                            "output_file": output_file,
                            "transcript_paths": transcript_paths,
                            "config": {
                                "model": self.get_config("model", "base"),
                                "language": self.get_config("language", "en"),
                                "speaker_labels": {
                                    "microphone": speaker_label,
                                    "system": system_label
                                }
                            }
                        },
                        "metadata": combined_metadata,
                        "context": {
                            "correlation_id": correlation_id,
                            "source_plugin": self.name,
                            "metadata": combined_metadata
                        }
                    },
                    correlation_id=correlation_id,
                    source_plugin=self.name,
                    priority=EventPriority.NORMAL
                )

            await self.event_bus.publish(event)

            # Structured logging
            log_data = {
                "plugin": self.name,
                "recording_id": recording_id,
                "event": event.name,
                "status": status,
            }
            if output_file:
                log_data["output_file"] = output_file
            if error:
                log_data["error"] = error

            logger.info("Emitted transcription event", extra=log_data)

        except Exception as e:
            logger.error(
                "Failed to emit transcription event",
                extra={
                    "plugin": self.name,
                    "error": str(e),
                    "recording_id": recording_id,
                },
                exc_info=True,
            )

    async def _init_database(self) -> None:
        """Initialize database tables."""
        if not self.db:
            return

        # Create tables using the connection from our db instance
        with self.db.get_connection() as conn:  # Now this will work
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS transcriptions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    recording_id TEXT NOT NULL,
                    transcript TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (recording_id) REFERENCES recordings(recording_id)
                )
            """
            )
            conn.commit()

    async def _update_task_status(
        self,
        recording_id: str,
        status: str,
        output_paths: list[str] | None = None,
        merged_output_path: str | None = None,
        error_message: str | None = None,
    ) -> None:
        """Update the status of a processing task in the database"""
        db = await DatabaseManager.get_instance()
        with db.get_connection() as conn:
            cursor = conn.cursor()
            update_values = [
                status,
                json.dumps(output_paths) if output_paths else None,
                merged_output_path,
                error_message,
                datetime.now(tz=timezone.utc).isoformat(),
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

    async def _create_task(self, recording_id: str, input_paths: list[str]) -> None:
        """Create a new processing task in the database"""
        db = await DatabaseManager.get_instance()
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
            if not os.path.exists(wav_path):
                self.logger.error(
                    "WAV file does not exist",
                    extra={"file": wav_path},
                )
                return False

            with wave.open(wav_path, "rb") as wav_file:
                channels = wav_file.getnchannels()
                sample_width = wav_file.getsampwidth()
                framerate = wav_file.getframerate()
                n_frames = wav_file.getnframes()
                duration = n_frames / float(framerate)

                # Log detailed validation info
                self.logger.info(
                    "WAV file validated",
                    extra={
                        "file": wav_path,
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
                extra={"file": wav_path, "error": str(e)},
                exc_info=True
            )
            return False

    async def transcribe_audio(
        self, audio_path: str, output_path: str, label: str, metadata: dict | None = None
    ) -> Path:
        """Transcribe an audio file using Whisper"""
        try:
            # Ensure model is initialized
            if self._model is None:
                self._init_model()

            # Run transcription in thread pool
            loop = asyncio.get_running_loop()
            segments, info = await loop.run_in_executor(
                self._executor,
                lambda: self._model.transcribe(  # type: ignore[union-attr]
                    audio_path,
                    condition_on_previous_text=False,
                    word_timestamps=True,
                    vad_filter=True,
                    vad_parameters=dict(
                        min_silence_duration_ms=500,
                        speech_pad_ms=100,
                    ),
                    beam_size=5,
                )
            )

            # Create output directory if it doesn't exist
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)

            def write_transcript():
                with open(output_file, "w") as f:
                    # Write header
                    f.write("# Audio Transcript\n\n")
                    f.write(f"Speaker: {label}\n\n")
                    
                    # Write metadata if provided
                    if metadata:
                        f.write("## Metadata\n\n")
                        f.write("```json\n")
                        f.write(json.dumps(metadata, indent=2))
                        f.write("\n```\n\n")
                    
                    f.write("## Segments\n\n")

                    # Process segments
                    for segment in segments:
                        if not hasattr(segment, "words") or not segment.words:
                            continue

                        # Group words into ~5 second chunks
                        words = segment.words
                        chunk_words = []
                        chunk_start = words[0].start

                        for word in words:
                            if word.start - chunk_start > 5.0 and chunk_words:
                                # Write chunk
                                chunk_text = " ".join(w.word for w in chunk_words).strip()
                                if chunk_text:
                                    start_time = self._format_timestamp(chunk_start)
                                    end_time = self._format_timestamp(word.start)
                                    f.write(f"[{start_time} -> {end_time}] {label}: {chunk_text}\n")

                                # Start new chunk
                                chunk_start = word.start
                                chunk_words = [word]
                            else:
                                chunk_words.append(word)

                        # Write final chunk
                        if chunk_words:
                            chunk_text = " ".join(w.word for w in chunk_words).strip()
                            if chunk_text:
                                start_time = self._format_timestamp(chunk_start)
                                end_time = self._format_timestamp(chunk_words[-1].end)
                                f.write(f"[{start_time} -> {end_time}] {label}: {chunk_text}\n")

            await loop.run_in_executor(self._executor, write_transcript)
            return output_file

        except Exception as e:
            logger.error(
                "Error transcribing audio",
                extra={
                    "plugin": self.name,
                    "audio_path": audio_path,
                    "output_path": output_path,
                    "error": str(e)
                },
                exc_info=True
            )
            raise

    def _format_timestamp(self, seconds: float) -> str:
        """Format seconds into MM:SS.ss timestamp"""
        minutes = int(seconds // 60)
        seconds = seconds % 60
        return f"{minutes:02d}:{seconds:05.2f}"

    async def merge_transcripts(
        self,
        transcript_files: list[str],
        output_path: str,
        labels: list[str],
        original_event: dict,
    ) -> None:
        """Merge multiple transcript files ordered by timestamp"""
        # Update output path to .md
        output_path = str(Path(output_path).with_suffix(".md"))

        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Read all transcripts and parse timestamps
        segments = []
        for transcript_file, label in zip(transcript_files, labels, strict=False):
            with open(transcript_file) as tf:
                in_metadata = False
                for line in tf:
                    # Skip empty lines
                    if not line.strip():
                        continue

                    # Track metadata section
                    if line.strip() == "```json":
                        in_metadata = True
                        continue
                    elif line.strip() == "```":
                        in_metadata = False
                        continue

                    # Skip metadata section and headers
                    if in_metadata or line.startswith("#"):
                        continue

                    # Only parse lines that look like transcript entries
                    if line.strip() and "[" in line and "]" in line and "->" in line:
                        try:
                            # Parse timestamp and text
                            time_part = line[line.find("[") + 1 : line.find("]")]
                            start_time = float(
                                time_part.split("->")[0].strip().split(":")[0]
                            ) * 60 + float(
                                time_part.split("->")[0].strip().split(":")[1]
                            )
                            text = line[line.find("]") + 1 :].strip()
                            segments.append((start_time, label, text))
                        except (ValueError, IndexError) as e:
                            logger.warning(
                                "Failed to parse timestamp in transcript line",
                                extra={
                                    "plugin": self.name,
                                    "error": str(e),
                                    "line": line,
                                    "file": transcript_file,
                                },
                            )

        # Sort segments by timestamp
        segments.sort(key=lambda x: x[0])

        # Write merged transcript to file
        with open(output_path, "w") as f:
            # Add metadata header
            f.write("# Merged Transcript\n\n")
            f.write("## Metadata\n\n")

            # Use original metadata
            metadata = original_event.get("metadata", {})

            # Write metadata as JSON in markdown code block
            f.write("```json\n")
            f.write(json.dumps(metadata, indent=2))
            f.write("\n```\n\n")

            # Write chronologically ordered segments
            f.write("## Transcript\n\n")
            for start_time, label, text in segments:
                minutes = int(start_time // 60)
                seconds = start_time % 60
                # Remove doubled label from text if it exists
                text = text.replace(f"{label}: ", "").strip()
                text = text.replace("Microphone: ", "").replace("System: ", "").strip()

                # Clean up transcript if enabled in config
                if (
                    hasattr(self.config, "config")
                    and isinstance(self.config.config, dict)
                    and self.config.config.get("clean_up_transcript", False)
                ):
                    text = self._transcript_cleaner.clean_transcript(text)

                f.write(f"[{minutes:02d}:{seconds:05.2f}] {label}: {text}\n")

    async def _process_audio(
        self, audio_path: str, speaker_label: str | None = None, metadata: dict | None = None
    ) -> Path | None:
        """Process audio file and return transcript path"""
        try:
            # Validate audio path
            if not audio_path or not os.path.exists(audio_path):
                logger.error(
                    "Invalid audio path",
                    extra={"plugin": self.name, "audio_path": audio_path}
                )
                return None

            # Get base filename without extension
            audio_file = Path(audio_path)
            base_name = audio_file.stem
            
            # Create output path
            output_path = self._output_dir / f"{base_name}.md"
            
            try:
                # Validate WAV file
                if not self.validate_wav_file(audio_path):
                    logger.error(
                        "Invalid WAV file",
                        extra={"plugin": self.name, "audio_path": audio_path}
                    )
                    return None

                # Transcribe audio
                transcript_path = await self.transcribe_audio(
                    audio_path,
                    str(output_path),
                    speaker_label or "Speaker",
                    metadata
                )

                if not transcript_path or not os.path.exists(transcript_path):
                    logger.error(
                        "Transcription failed - no output file generated",
                        extra={
                            "plugin": self.name,
                            "audio_path": audio_path,
                            "expected_output": str(output_path)
                        }
                    )
                    return None

                return transcript_path

            except Exception as e:
                logger.error(
                    "Error during transcription",
                    extra={
                        "plugin": self.name,
                        "audio_path": audio_path,
                        "error": str(e)
                    },
                    exc_info=True
                )
                return None

        except Exception as e:
            logger.error(
                "Error processing audio file",
                extra={
                    "plugin": self.name,
                    "audio_path": audio_path,
                    "error": str(e)
                },
                exc_info=True
            )
            return None

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

            # Initialize the model with offline settings
            self._model = WhisperModel(
                model_size_or_path=model_name,
                device="cpu",
                device_index=0,
                compute_type="default",
                download_root=str(model_dir),
                local_files_only=False,  # Allow downloading if not present
            )

            logger.info(
                "Faster-Whisper model initialized",
                extra={
                    "plugin": self.name,
                    "model_name": model_name,
                    "model_dir": str(model_dir),
                },
            )

        except Exception as e:
            logger.error(
                "Failed to initialize Whisper model",
                extra={"plugin": self.name, "error": str(e)},
                exc_info=True,
            )
            raise
