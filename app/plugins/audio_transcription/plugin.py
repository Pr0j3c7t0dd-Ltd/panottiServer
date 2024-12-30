import asyncio
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import logging
import wave

from faster_whisper import WhisperModel

from app.plugins.base import PluginBase
from app.plugins.events.models import Event, EventContext, EventPriority
from app.models.database import DatabaseManager

logger = logging.getLogger(__name__)

class AudioTranscriptionPlugin(PluginBase):
    """Plugin for transcribing audio files using Whisper"""
    
    def __init__(self, config, event_bus=None):
        super().__init__(config, event_bus)
        self._executor = None
        self._processing_lock = threading.Lock()
        self._db_initialized = False
        self._model = None
        
    async def _initialize(self) -> None:
        """Initialize plugin"""
        # Initialize database table
        await self._init_database()
        
        # Initialize thread pool executor
        max_workers = self.get_config("max_concurrent_tasks", 4)
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        
        # Initialize Whisper model
        model_name = self.get_config("model_name", "base.en")
        self._model = WhisperModel(model_name, device="cpu", compute_type="int8")
        
        # Subscribe to noise reduction completed event
        self.event_bus.subscribe("noise_reduction.completed", self.handle_noise_reduction_completed)
        
        self.logger.info(
            "AudioTranscriptionPlugin initialized successfully",
            extra={
                "plugin": "audio_transcription",
                "max_workers": max_workers,
                "model": model_name,
                "output_directory": self.get_config("output_directory", "data/transcripts"),
                "db_initialized": self._db_initialized
            }
        )
        
    async def _shutdown(self) -> None:
        """Shutdown plugin"""
        # Unsubscribe from events
        self.event_bus.unsubscribe("noise_reduction.completed", self.handle_noise_reduction_completed)
        
        # Shutdown thread pool
        if self._executor:
            self._executor.shutdown(wait=True)
            
        self.logger.info("Audio transcription plugin shutdown")
        
    async def _init_database(self) -> None:
        """Initialize database table for tracking processing state"""
        db = DatabaseManager.get_instance()
        with db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS transcription_tasks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    recording_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    input_paths TEXT,  -- JSON array of input file paths
                    output_paths TEXT, -- JSON array of output transcript paths
                    merged_output_path TEXT,
                    error_message TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            conn.commit()
        self._db_initialized = True
        
    def _update_task_status(self, recording_id: str, status: str, 
                           output_paths: Optional[List[str]] = None,
                           merged_output_path: Optional[str] = None,
                           error_message: Optional[str] = None) -> None:
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
                recording_id
            ]
            cursor.execute('''
                UPDATE transcription_tasks
                SET status = ?, output_paths = ?, merged_output_path = ?, 
                    error_message = ?, updated_at = ?
                WHERE recording_id = ?
            ''', update_values)
            conn.commit()
            
    def _create_task(self, recording_id: str, input_paths: List[str]) -> None:
        """Create a new processing task in the database"""
        db = DatabaseManager.get_instance()
        with db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO transcription_tasks 
                (recording_id, status, input_paths)
                VALUES (?, ?, ?)
            ''', (recording_id, 'pending', json.dumps(input_paths)))
            conn.commit()

    def validate_wav_file(self, wav_path: str) -> bool:
        """Validate that the file is a proper WAV file and can be opened"""
        try:
            with wave.open(wav_path, 'rb') as wav_file:
                channels = wav_file.getnchannels()
                sample_width = wav_file.getsampwidth()
                framerate = wav_file.getframerate()
                n_frames = wav_file.getnframes()
                duration = n_frames / float(framerate)
                
                self.logger.info(
                    f"WAV file validated",
                    extra={
                        "file": os.path.basename(wav_path),
                        "channels": channels,
                        "sample_width": sample_width,
                        "framerate": framerate,
                        "duration": duration
                    }
                )
                return True
        except Exception as e:
            self.logger.error(
                f"WAV file validation failed",
                extra={
                    "file": os.path.basename(wav_path),
                    "error": str(e)
                }
            )
            return False

    def transcribe_audio(self, audio_path: str, output_path: str, label: str) -> Optional[str]:
        """Transcribe an audio file using Whisper"""
        try:
            if not self.validate_wav_file(audio_path):
                raise ValueError(f"Invalid WAV file: {audio_path}")
                
            self.logger.info(f"Transcribing {os.path.basename(audio_path)}...")
            
            # Transcribe the audio file
            segments, info = self._model.transcribe(
                audio_path,
                beam_size=5,
                word_timestamps=True
            )
            
            # Generate markdown with timestamps and labels
            with open(output_path, 'w') as f:
                f.write(f"# Transcript from {label}\n\n")
                for segment in segments:
                    timestamp = f"[{segment.start:.2f}s - {segment.end:.2f}s]"
                    f.write(f"{timestamp} ({label}) {segment.text}\n\n")
                    
            self.logger.info(
                f"Transcription completed",
                extra={
                    "file": os.path.basename(audio_path),
                    "output": os.path.basename(output_path)
                }
            )
            
            return output_path
            
        except Exception as e:
            self.logger.error(
                f"Transcription failed",
                extra={
                    "file": os.path.basename(audio_path),
                    "error": str(e)
                }
            )
            return None
            
    def merge_transcripts(self, transcript_files: List[str], output_path: str, labels: List[str], 
                         original_event: dict) -> str:
        """Merge multiple transcript files based on timestamps"""
        try:
            segments = []
            file_labels = {}  # Map file paths to their labels
            
            # Create mapping of files to labels
            for file_path, label in zip(transcript_files, labels):
                file_labels[file_path] = label
            
            # Write merged transcript
            with open(output_path, 'w') as f:
                # Write metadata header
                f.write("# Merged Transcript\n\n")
                f.write("## Recording Metadata\n")
                f.write("```json\n")
                metadata = {
                    "recording_id": original_event.get("recording_id"),
                    "recording_timestamp": original_event.get("recording_timestamp"),
                    "event_title": original_event.get("metadata", {}).get("eventTitle"),
                    "event_provider": original_event.get("metadata", {}).get("eventProvider"),
                    "event_provider_id": original_event.get("metadata", {}).get("eventProviderId"),
                    "event_attendees": original_event.get("metadata", {}).get("eventAttendees", []),
                    "system_label": original_event.get("metadata", {}).get("systemLabel"),
                    "microphone_label": original_event.get("metadata", {}).get("microphoneLabel"),
                    "recording_started": original_event.get("metadata", {}).get("recordingStarted"),
                    "recording_ended": original_event.get("metadata", {}).get("recordingEnded"),
                    "system_audio_path": original_event.get("systemAudioPath"),
                    "microphone_audio_path": original_event.get("microphoneAudioPath")
                }
                f.write(json.dumps(metadata, indent=2))
                f.write("\n```\n\n")
                f.write("## Transcript\n\n")
            
                # Read all transcripts and parse segments
                for file_path in transcript_files:
                    label = file_labels[file_path]
                    with open(file_path, 'r') as tf:
                        for line in tf:
                            line = line.strip()
                            if line and not line.startswith('#'):  # Skip header
                                # Parse timestamp and text
                                timestamp_end = line.find(']')
                                if timestamp_end != -1:
                                    timestamp = line[1:timestamp_end]
                                    
                                    # Extract text after timestamp, removing any existing label
                                    text = line[timestamp_end + 2:]
                                    label_start = text.find('(')
                                    label_end = text.find(')')
                                    if label_start != -1 and label_end != -1:
                                        # Remove the existing label
                                        text = text[label_end + 2:].strip()
                                    
                                    # Parse start and end times
                                    times = timestamp.split(' - ')
                                    start = float(times[0].replace('s', ''))
                                    end = float(times[1].replace('s', ''))
                                    
                                    segments.append((start, end, text, label))
                
                # Sort segments by start time
                segments.sort(key=lambda x: x[0])
                
                # Write transcript content
                for start, end, text, label in segments:
                    f.write(f"[{start:.2f}s - {end:.2f}s] ({label}) {text}\n\n")
                    
            return output_path
            
        except Exception as e:
            self.logger.error(
                f"Transcript merge failed",
                extra={
                    "output_file": os.path.basename(output_path),
                    "error": str(e)
                }
            )
            raise

    def handle_noise_reduction_completed(self, event: Event) -> None:
        """Handle noise_reduction.completed events"""
        try:
            recording_id = event.payload["recording_id"]
            original_event = event.payload["original_event"]
            
            # Get custom labels from original event or use defaults
            mic_label = original_event.get("microphone_label", "microphone")
            sys_label = original_event.get("system_label", "system")
            
            # Get input files
            input_files = []
            input_labels = []
            file_types = []  # "microphone" or "system" for filenames
            
            # Add system audio if present
            sys_audio = original_event.get("system_audio_path")
            if sys_audio and sys_audio.strip():
                input_files.append(sys_audio)
                input_labels.append(sys_label)
                file_types.append("system")
            
            # Add microphone audio (prefer cleaned version if available)
            mic_audio = event.payload.get("microphone_cleaned_file")
            if not mic_audio:
                mic_audio = original_event.get("microphone_audio_path")
            if mic_audio and mic_audio.strip():
                input_files.append(mic_audio)
                input_labels.append(mic_label)
                file_types.append("microphone")
                
            if not input_files:
                raise ValueError("No valid input files found in event")
                
            # Create processing task
            self._create_task(recording_id, input_files)
            
            # Generate output paths
            output_dir = Path(self.get_config("output_directory", "data/transcripts"))
            output_dir.mkdir(parents=True, exist_ok=True)
            
            output_files = []
            for input_file, file_type in zip(input_files, file_types):
                base_name = Path(input_file).stem
                output_path = output_dir / f"{base_name}_{file_type}_transcript.md"
                output_files.append(str(output_path))
                
            merged_output = output_dir / f"{recording_id}_transcript.md"
            
            # Process in thread pool
            self._executor.submit(
                self._process_audio,
                recording_id,
                input_files,
                output_files,
                str(merged_output),
                event,
                input_labels  # Pass labels to _process_audio
            )
            
        except Exception as e:
            self.logger.error(
                f"Failed to handle noise reduction completed event",
                extra={
                    "recording_id": event.payload["recording_id"],
                    "error": str(e)
                }
            )
            self._emit_completion_event(
                event.payload["recording_id"],
                event,
                None,
                "error",
                str(e)
            )

    def _process_audio(self, recording_id: str, input_files: List[str],
                      output_files: List[str], merged_output: str,
                      original_event: Event, input_labels: List[str]) -> None:
        """Process audio files in a separate thread"""
        try:
            # Update status to processing
            self._update_task_status(recording_id, "processing")
            
            # Transcribe each file
            for input_file, output_file, label in zip(input_files, output_files, input_labels):
                self.transcribe_audio(input_file, output_file, label)
                
            # Merge transcripts
            if len(output_files) > 1:
                self.merge_transcripts(output_files, merged_output, input_labels, original_event.payload)
            else:
                # If only one transcript, just copy it and add metadata
                with open(output_files[0], 'r') as src:
                    content = src.read()
                with open(merged_output, 'w') as dest:
                    # Write metadata
                    dest.write("# Merged Transcript\n\n")
                    dest.write("## Recording Metadata\n")
                    dest.write("```json\n")
                    metadata = {
                        "recording_id": original_event.payload.get("recording_id"),
                        "recording_timestamp": original_event.payload.get("recording_timestamp"),
                        "event_title": original_event.payload.get("metadata", {}).get("eventTitle"),
                        "event_provider": original_event.payload.get("metadata", {}).get("eventProvider"),
                        "event_provider_id": original_event.payload.get("metadata", {}).get("eventProviderId"),
                        "event_attendees": original_event.payload.get("metadata", {}).get("eventAttendees", []),
                        "system_label": original_event.payload.get("metadata", {}).get("systemLabel"),
                        "microphone_label": original_event.payload.get("metadata", {}).get("microphoneLabel"),
                        "recording_started": original_event.payload.get("metadata", {}).get("recordingStarted"),
                        "recording_ended": original_event.payload.get("metadata", {}).get("recordingEnded"),
                        "system_audio_path": original_event.payload.get("systemAudioPath"),
                        "microphone_audio_path": original_event.payload.get("microphoneAudioPath")
                    }
                    dest.write(json.dumps(metadata, indent=2))
                    dest.write("\n```\n\n")
                    dest.write("## Transcript\n\n")
                    dest.write(content)
                
            # Update status to completed
            self._update_task_status(
                recording_id,
                "completed",
                output_files,
                merged_output
            )
            
            # Emit completion event
            self._emit_completion_event(
                recording_id,
                original_event,
                merged_output,
                "completed",
                output_files=output_files
            )
            
        except Exception as e:
            self.logger.error(
                f"Audio processing failed",
                extra={
                    "recording_id": recording_id,
                    "error": str(e)
                }
            )
            self._update_task_status(recording_id, "error", error_message=str(e))
            self._emit_completion_event(
                recording_id,
                original_event,
                None,
                "error",
                str(e)
            )

    def _emit_completion_event(self, recording_id: str, original_event: Event,
                             output_file: Optional[str], status: str,
                             error_message: Optional[str] = None,
                             output_files: Optional[List[str]] = None) -> None:
        """Emit event when processing is complete"""
        event = Event(
            name="transcription.completed",
            payload={
                "recording_id": recording_id,
                "original_event": original_event.payload,
                "output_file": output_file,
                "output_files": output_files,
                "status": status,
                "error": error_message
            },
            context=EventContext(
                correlation_id=original_event.context.correlation_id,
                source_plugin=self.name
            ),
            priority=EventPriority.LOW
        )
        
        self.event_bus.publish(event)
        
        self.logger.info(
            "Emitted completion event",
            extra={
                "recording_id": recording_id,
                "status": status,
                "correlation_id": original_event.context.correlation_id
            }
        )
