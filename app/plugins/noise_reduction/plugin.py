import asyncio
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional
import logging
import numpy as np
from scipy import signal
from scipy.io import wavfile
import warnings

from app.plugins.base import PluginBase
from app.plugins.events.models import Event, EventContext, EventPriority
from app.models.database import DatabaseManager

logger = logging.getLogger(__name__)

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
        
        # Validate input files exist
        if not os.path.exists(mic_file):
            raise FileNotFoundError(f"Microphone audio file not found: {mic_file}")
        if not os.path.exists(noise_file):
            raise FileNotFoundError(f"System audio file not found: {noise_file}")
            
        # Create output directory with proper path handling
        output_dir = os.path.dirname(os.path.abspath(output_file))
        try:
            os.makedirs(output_dir, exist_ok=True)
        except Exception as e:
            raise IOError(f"Failed to create output directory {output_dir}: {str(e)}")
        
        # Read both audio files, suppressing WAV file warnings
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", wavfile.WavFileWarning)
                mic_rate, mic_data = wavfile.read(mic_file)
                noise_rate, noise_data = wavfile.read(noise_file)
        except Exception as e:
            raise IOError(f"Failed to read WAV files: {str(e)}")
        
        # Store original shape for later reconstruction
        original_shape = mic_data.shape
        is_stereo = len(original_shape) > 1
        
        # Convert to mono temporarily for noise processing
        if is_stereo:
            mic_mono = np.mean(mic_data, axis=1)
            noise_mono = np.mean(noise_data, axis=1) if len(noise_data.shape) > 1 else noise_data
        else:
            mic_mono = mic_data
            noise_mono = noise_data
            
        # Convert to float32 for processing
        mic_mono = mic_mono.astype(np.float32)
        noise_mono = noise_mono.astype(np.float32)
        
        # Normalize audio with safety checks
        mic_max = np.max(np.abs(mic_mono))
        noise_max = np.max(np.abs(noise_mono))
        
        if mic_max == 0:
            raise ValueError("Microphone audio is silent")
        if noise_max == 0:
            raise ValueError("System audio is silent")
            
        mic_mono = mic_mono / mic_max
        noise_mono = noise_mono / noise_max
        
        # Calculate STFT parameters
        nperseg = 2048  # Window size
        noverlap = nperseg // 2  # 50% overlap
        
        # Compute noise profile using STFT
        _, _, noise_spectrogram = signal.stft(
            noise_mono,
            fs=noise_rate,
            nperseg=nperseg,
            noverlap=noverlap
        )
        
        # Calculate noise profile magnitude
        noise_profile = np.mean(np.abs(noise_spectrogram), axis=1)
        
        if is_stereo:
            # Process each channel separately
            cleaned_channels = []
            for channel in range(original_shape[1]):
                channel_data = mic_data[:, channel].astype(np.float32)
                channel_max = np.max(np.abs(channel_data))
                if channel_max > 0:
                    channel_data = channel_data / channel_max
                
                # Compute STFT of channel
                f, t, channel_spectrogram = signal.stft(
                    channel_data,
                    fs=mic_rate,
                    nperseg=nperseg,
                    noverlap=noverlap
                )
                
                # Apply spectral subtraction
                channel_mag = np.abs(channel_spectrogram)
                channel_phase = np.angle(channel_spectrogram)
                
                # Expand noise profile to match spectrogram shape
                noise_profile_reshaped = noise_profile.reshape(-1, 1)
                
                # Subtract noise profile from magnitude spectrum with safety floor
                cleaned_mag = np.maximum(
                    channel_mag - noise_profile_reshaped * noise_reduce_factor,
                    channel_mag * 0.1  # Spectral floor to prevent complete silence
                )
                
                # Reconstruct complex spectrogram
                cleaned_spectrogram = cleaned_mag * np.exp(1j * channel_phase)
                
                # Inverse STFT to get cleaned audio
                _, cleaned_channel = signal.istft(
                    cleaned_spectrogram,
                    fs=mic_rate,
                    nperseg=nperseg,
                    noverlap=noverlap
                )
                
                # Normalize channel
                cleaned_max = np.max(np.abs(cleaned_channel))
                if cleaned_max > 0:
                    cleaned_channel = cleaned_channel / cleaned_max
                
                cleaned_channels.append(cleaned_channel)
            
            # Stack channels back into stereo
            cleaned_audio = np.stack(cleaned_channels, axis=1)
        else:
            # Mono processing (original code)
            # Compute STFT of microphone audio
            f, t, mic_spectrogram = signal.stft(
                mic_mono,
                fs=mic_rate,
                nperseg=nperseg,
                noverlap=noverlap
            )
            
            # Apply spectral subtraction
            mic_mag = np.abs(mic_spectrogram)
            mic_phase = np.angle(mic_spectrogram)
            
            # Expand noise profile to match spectrogram shape
            noise_profile = noise_profile.reshape(-1, 1)
            
            # Subtract noise profile from magnitude spectrum with safety floor
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
            
            # Normalize output with safety checks
            cleaned_max = np.max(np.abs(cleaned_audio))
            if cleaned_max > 0:
                cleaned_audio = cleaned_audio / cleaned_max
        
        # Convert back to int16 with clipping protection
        cleaned_audio = np.clip(cleaned_audio * 32767, -32768, 32767).astype(np.int16)
        
        # Save cleaned audio with error handling
        try:
            wavfile.write(output_file, mic_rate, cleaned_audio)
            
            # Verify the file was written successfully
            if not os.path.exists(output_file):
                raise IOError(f"Failed to write output file: {output_file}")
                
            # Verify the file size is non-zero
            if os.path.getsize(output_file) == 0:
                raise IOError(f"Output file is empty: {output_file}")
                
        except Exception as e:
            if os.path.exists(output_file):
                try:
                    os.remove(output_file)
                except:
                    pass
            raise IOError(f"Failed to save cleaned audio: {str(e)}")
        
    async def handle_recording_ended(self, event: Event) -> None:
        """Handle recording_ended events"""
        try:
            # Extract recording information from event
            recording_id = event.payload.get("recordingId")
            mic_path = event.payload.get("microphoneAudioPath")
            sys_path = event.payload.get("systemAudioPath")
            
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
            # Set up event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            self.logger.info(
                "Starting audio processing in thread",
                extra={
                    "plugin": "noise_reduction",
                    "recording_id": recording_id,
                    "mic_path": mic_path,
                    "sys_path": sys_path,
                    "output_file": output_file,
                    "noise_reduce_factor": self.get_config("noise_reduce_factor", 0.7),
                    "correlation_id": original_event.context.correlation_id,
                    "thread_id": threading.get_ident()
                }
            )
            
            # Create processing task in database and update status
            self._create_task(recording_id, mic_path, sys_path)
            self._update_task_status(recording_id, "processing")
            
            # Process audio
            self.reduce_noise(
                mic_path,
                sys_path,
                output_file,
                self.get_config("noise_reduce_factor", 0.7)
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
            future = loop.run_until_complete(
                self._emit_completion_event(
                    recording_id,
                    original_event,
                    output_file,
                    "success"
                )
            )
            
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
            
            # Emit error completion event
            if loop and loop.is_running():
                future = loop.run_until_complete(
                    self._emit_completion_event(
                        recording_id,
                        original_event,
                        None,
                        "error"
                    )
                )
        finally:
            # Clean up the event loop
            if loop:
                loop.close()
            
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
