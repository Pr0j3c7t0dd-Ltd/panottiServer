import asyncio
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import numpy as np
from scipy import signal
from scipy.io import wavfile

from app.plugins.base import PluginBase
from app.plugins.events.models import Event, EventContext, EventPriority
from app.models.database import DatabaseManager

class NoiseReductionPlugin(PluginBase):
    """Plugin for reducing background noise from microphone recordings"""
    
    def __init__(self, config, event_bus=None):
        super().__init__(config, event_bus)
        self._executor = None
        self._processing_lock = threading.Lock()
        self._db_initialized = False
        
    async def _initialize(self) -> None:
        """Initialize plugin"""
        # Initialize database table
        await self._init_database()
        
        # Initialize thread pool executor
        max_workers = self.get_config("max_concurrent_tasks", 4)
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        
        # Subscribe to recording_ended event
        self.event_bus.subscribe("recording_ended", self.handle_recording_ended)
        
        self.logger.info(
            "NoiseReductionPlugin initialized successfully",
            extra={
                "plugin": "noise_reduction",
                "max_workers": max_workers,
                "output_directory": self.get_config("output_directory", "data/cleaned_audio"),
                "noise_reduce_factor": self.get_config("noise_reduce_factor", 0.7),
                "db_initialized": self._db_initialized
            }
        )
        
    async def _shutdown(self) -> None:
        """Shutdown plugin"""
        # Unsubscribe from events
        self.event_bus.unsubscribe("recording_ended", self.handle_recording_ended)
        
        # Shutdown thread pool
        if self._executor:
            self._executor.shutdown(wait=True)
            
        self.logger.info("Noise reduction plugin shutdown")
        
    async def _init_database(self) -> None:
        """Initialize database table for tracking processing state"""
        db = DatabaseManager.get_instance()
        with db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS noise_reduction_tasks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    recording_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    input_mic_path TEXT,
                    input_sys_path TEXT,
                    output_path TEXT,
                    error_message TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            conn.commit()
        self._db_initialized = True
        
    def _update_task_status(self, recording_id: str, status: str, 
                           output_path: Optional[str] = None, 
                           error_message: Optional[str] = None) -> None:
        """Update the status of a processing task in the database"""
        db = DatabaseManager.get_instance()
        with db.get_connection() as conn:
            cursor = conn.cursor()
            update_values = [
                status,
                output_path,
                error_message,
                datetime.utcnow().isoformat(),
                recording_id
            ]
            cursor.execute('''
                UPDATE noise_reduction_tasks
                SET status = ?, output_path = ?, error_message = ?, updated_at = ?
                WHERE recording_id = ?
            ''', update_values)
            conn.commit()
            
    def _create_task(self, recording_id: str, mic_path: str, sys_path: str) -> None:
        """Create a new processing task in the database"""
        db = DatabaseManager.get_instance()
        with db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO noise_reduction_tasks 
                (recording_id, status, input_mic_path, input_sys_path)
                VALUES (?, ?, ?, ?)
            ''', (recording_id, 'pending', mic_path, sys_path))
            conn.commit()
            
    def reduce_noise(self, mic_file: str, noise_file: str, output_file: str, 
                    noise_reduce_factor: float = 0.7) -> None:
        """
        Reduce background noise from microphone audio using a noise profile.
        
        Args:
            mic_file: Path to microphone recording WAV file
            noise_file: Path to system recording WAV file (noise profile)
            output_file: Path to save cleaned audio
            noise_reduce_factor: Amount of noise reduction (0 to 1)
        """
        self.logger.info(
            "Starting noise reduction process",
            extra={
                "plugin": "noise_reduction",
                "mic_file": mic_file,
                "noise_file": noise_file,
                "output_file": output_file,
                "noise_reduce_factor": noise_reduce_factor
            }
        )
        
        # Read both audio files
        mic_rate, mic_data = wavfile.read(mic_file)
        noise_rate, noise_data = wavfile.read(noise_file)
        
        # Convert stereo to mono by averaging channels
        if len(mic_data.shape) > 1:
            mic_data = np.mean(mic_data, axis=1)
        if len(noise_data.shape) > 1:
            noise_data = np.mean(noise_data, axis=1)
            
        # Convert to float32 for processing
        mic_data = mic_data.astype(np.float32)
        noise_data = noise_data.astype(np.float32)
        
        # Normalize audio
        mic_data = mic_data / np.max(np.abs(mic_data))
        noise_data = noise_data / np.max(np.abs(noise_data))
        
        # Calculate STFT parameters
        nperseg = 2048  # Window size
        noverlap = nperseg // 2  # 50% overlap
        
        # Compute noise profile using STFT
        _, _, noise_spectrogram = signal.stft(
            noise_data,
            fs=noise_rate,
            nperseg=nperseg,
            noverlap=noverlap
        )
        
        # Calculate noise profile magnitude
        noise_profile = np.mean(np.abs(noise_spectrogram), axis=1)
        
        # Compute STFT of microphone audio
        f, t, mic_spectrogram = signal.stft(
            mic_data,
            fs=mic_rate,
            nperseg=nperseg,
            noverlap=noverlap
        )
        
        # Apply spectral subtraction
        mic_mag = np.abs(mic_spectrogram)
        mic_phase = np.angle(mic_spectrogram)
        
        # Expand noise profile to match spectrogram shape
        noise_profile = noise_profile.reshape(-1, 1)
        
        # Subtract noise profile from magnitude spectrum
        cleaned_mag = np.maximum(
            mic_mag - noise_profile * noise_reduce_factor,
            mic_mag * 0.1  # Spectral floor to prevent complete silence
        )
        
        # Reconstruct complex spectrogram
        cleaned_spectrogram = cleaned_mag * np.exp(1j * mic_phase)
        
        # Inverse STFT to get cleaned audio
        _, cleaned_audio = signal.istft(
            cleaned_spectrogram,
            fs=mic_rate,
            nperseg=nperseg,
            noverlap=noverlap
        )
        
        # Normalize output
        cleaned_audio = cleaned_audio / np.max(np.abs(cleaned_audio))
        
        # Convert back to int16 for WAV file
        cleaned_audio = (cleaned_audio * 32767).astype(np.int16)
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Save cleaned audio
        wavfile.write(output_file, mic_rate, cleaned_audio)
        
    async def handle_recording_ended(self, event: Event) -> None:
        """Handle recording_ended events"""
        try:
            # Extract recording information from event
            recording_id = event.payload.get("recording_id")
            mic_path = event.payload.get("microphone_audio_path")
            sys_path = event.payload.get("system_audio_path")
            
            self.logger.info(
                "Processing recording",
                extra={
                    "recording_id": recording_id,
                    "correlation_id": event.context.correlation_id
                }
            )
            
            # Skip processing if either audio path is missing
            if not mic_path or not sys_path:
                self.logger.warning(
                    "Skipping noise reduction - missing audio paths",
                    extra={
                        "recording_id": recording_id,
                        "mic_path": mic_path,
                        "sys_path": sys_path
                    }
                )
                await self._emit_completion_event(
                    recording_id, event, None, "skipped"
                )
                return
                
            # Create output filename
            output_dir = self.get_config("output_directory", "data/cleaned_audio")
            output_file = os.path.join(
                output_dir,
                f"{recording_id}_microphone_cleaned.wav"
            )
            
            # Create processing task in database
            self._create_task(recording_id, mic_path, sys_path)
            
            # Process audio in thread pool
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                self._executor,
                self._process_audio,
                recording_id,
                mic_path,
                sys_path,
                output_file,
                event
            )
            
        except Exception as e:
            self.logger.error(
                "Error processing recording",
                extra={
                    "recording_id": recording_id,
                    "error": str(e)
                }
            )
            self._update_task_status(
                recording_id, 
                "failed",
                error_message=str(e)
            )
            
    def _process_audio(self, recording_id: str, mic_path: str, 
                      sys_path: str, output_file: str, 
                      original_event: Event) -> None:
        """Process audio files in a separate thread"""
        try:
            # Update status to processing
            self._update_task_status(recording_id, "processing")
            
            # Get noise reduction factor from config
            noise_reduce_factor = self.get_config("noise_reduce_factor", 0.7)
            
            self.logger.info(
                "Starting audio processing in thread",
                extra={
                    "plugin": "noise_reduction",
                    "recording_id": recording_id,
                    "mic_path": mic_path,
                    "sys_path": sys_path,
                    "output_file": output_file,
                    "noise_reduce_factor": noise_reduce_factor,
                    "correlation_id": original_event.context.correlation_id,
                    "thread_id": threading.get_ident()
                }
            )
            
            # Process audio
            self.reduce_noise(
                mic_path,
                sys_path,
                output_file,
                noise_reduce_factor
            )
            
            self.logger.info(
                "Audio processing completed successfully",
                extra={
                    "plugin": "noise_reduction",
                    "recording_id": recording_id,
                    "output_file": output_file,
                    "correlation_id": original_event.context.correlation_id,
                    "thread_id": threading.get_ident()
                }
            )
            
            # Update status to completed
            self._update_task_status(
                recording_id,
                "completed",
                output_path=output_file
            )
            
            # Emit completion event
            future = asyncio.run_coroutine_threadsafe(
                self._emit_completion_event(
                    recording_id,
                    original_event,
                    output_file,
                    "success"
                ),
                asyncio.get_event_loop()
            )
            # Wait for the future to complete
            future.result()
            
        except Exception as e:
            self.logger.error(
                "Error in audio processing thread",
                extra={
                    "plugin": "noise_reduction",
                    "recording_id": recording_id,
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "correlation_id": original_event.context.correlation_id,
                    "thread_id": threading.get_ident()
                }
            )
            self._update_task_status(
                recording_id,
                "failed",
                error_message=str(e)
            )
            
    async def _emit_completion_event(self, recording_id: str, 
                                   original_event: Event,
                                   output_file: Optional[str],
                                   status: str) -> None:
        """Emit event when processing is complete"""
        event = Event(
            name="noise_reduction.completed",
            payload={
                "recording_id": recording_id,
                "original_event": original_event.payload,
                "output_file": output_file,
                "status": status
            },
            context=EventContext(
                correlation_id=original_event.context.correlation_id,
                source_plugin=self.name
            ),
            priority=EventPriority.LOW
        )
        
        await self.event_bus.publish(event)
        
        self.logger.info(
            "Emitted completion event",
            extra={
                "recording_id": recording_id,
                "status": status,
                "correlation_id": original_event.context.correlation_id
            }
        )