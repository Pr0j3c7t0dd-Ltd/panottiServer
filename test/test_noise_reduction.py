# Standard library imports
import asyncio
import os
from pathlib import Path

# Local imports
from app.models.recording.events import RecordingEvent
from app.plugins.base import PluginConfig
from app.plugins.events.bus import EventBus
from app.plugins.noise_reduction.plugin import NoiseReductionPlugin


async def test_noise_reduction():
    # Initialize plugin
    config = PluginConfig(name="noise_reduction", version="1.0.0")
    event_bus = EventBus()
    plugin = NoiseReductionPlugin(config, event_bus)

    # Initialize plugin
    await plugin._initialize()

    # Create test event
    recording_id = "test_recording"
    test_dir = Path(os.path.dirname(__file__))
    mic_audio = str(test_dir / "fixtures/test_audio.wav")
    system_audio = str(
        test_dir / "fixtures/test_audio.wav"
    )  # Using same file as noise profile for test

    event = RecordingEvent(
        recording_timestamp="20241231162736",
        recording_id=recording_id,
        event="recording.ended",
        name="recording.ended",
        data={},
        microphoneAudioPath=mic_audio,
        systemAudioPath=system_audio,
    )

    # Process the event
    await plugin._handle_recording_ended(event)

    # Check output directory for processed file
    output_dir = Path("data/audio_processing")
    output_files = list(output_dir.glob(f"{recording_id}*cleaned.wav"))
    print(f"Generated output files: {output_files}")

    # Clean up
    await plugin._shutdown()


if __name__ == "__main__":
    asyncio.run(test_noise_reduction())
