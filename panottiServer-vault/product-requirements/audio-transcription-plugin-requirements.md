I want to create a plugin using the plugin architecture in the app that will transcribe an audio file with timestamps. I eventually want to be able to combine multiple transcripts together and order them based on the relative timestamp of the recording to produce a single file with the timestamped transcriptions merged in order. The incoming incoming system audio audio file is in the event's "original_event":"system_audio_path" and the incoming microphone audio is either  in the "original_event":"system_audio_path", or in the event "microphone_cleaned_file" (See below for instructions on when to use either of the files).  The transcription file should be saved in markdown format.

I want the plugin to listen for the 'noise_reduction.completed' even type (see the /app/plugins/noise_reduction/plugin.py for event info) to trigger the processing. 

The plugin config should include a directory to store the transcription files in.  The format of the transcription filename should be in the same format as the input files, e.g. <recording_id>_transcript.md.  Example: 20241216140128_2096E65E_transcript.md

The plugin should create it's own table in the sqlite database to manage the processing and state if required.  The table should capture the recording_id from the events table, although the same recording_id may be used to process multiple files.

The plugin should emit it's own event when the process is complete.  The completed event should pass on the information from the original event, plus the file information for the newly created merged transcript file, including the path to the merged transcript file.  the event should also add the file information for both of the individual transcript files from the system audio and microphone audio.

Ideally the processing should use multi-threading so that processing can happen concurrently.

Include the "system_audio_path" from the original_event if it is not null or empty in the array of files to transcribe.

Also, include the "microphone_audio_path" from the original_event is null or empty in the array of files to transcribe, UNLESS, the 'microphone_cleaned_file' from the incoming event is populated, then use that instead of the "microphone_audio_path" in the array of files to transcribe.

The working code to be implemented is below, no need to change, apart to incorporate into the plugin structure and app logging.  Ensure you have the correct logging in place.

Ensure you add a detailed README.md to the plugin, with any information needed for additional python package requirements.txt . In the README.md, make sure we mention that we need to `brew install openai whisper`

Make the whisper model a variable in the plugin settings, but default it to 'base.en'

Look at the current example plugin in the /app/plugins/example and /app/plugins/noise_reduction directory to get context on how to create the plugin.  Especially pay attention to how worker threads are handled in the noise_reduction plugin as an example.

---
Working Function to implement:

```
import wave

import json

import datetime

import os

from typing import List, Dict

from pathlib import Path

from faster_whisper import WhisperModel

  

class WhisperTranscriber:

def __init__(self, model_name: str = "base", device: str = "cuda"):

"""

Initialize the transcriber with specified model.

Available models: tiny, base, small, medium, large

device: "cpu" or "cuda" for GPU support

"""

print(f"Loading Whisper {model_name} model...")

self.model = WhisperModel(model_name, device=device, compute_type="int8")

print("Model loaded successfully!")

  

def validate_wav_file(self, wav_path: str) -> bool:

"""

Validate that the file is a proper WAV file and can be opened

Returns True if valid, False otherwise

"""

try:

with wave.open(wav_path, 'rb') as wav_file:

# Get basic WAV file properties

channels = wav_file.getnchannels()

sample_width = wav_file.getsampwidth()

framerate = wav_file.getframerate()

n_frames = wav_file.getnframes()

duration = n_frames / float(framerate)

print(f"WAV file details for {os.path.basename(wav_path)}:")

print(f"- Channels: {channels}")

print(f"- Sample width: {sample_width} bytes")

print(f"- Sample rate: {framerate} Hz")

print(f"- Duration: {duration:.2f} seconds")

return True

except Exception as e:

print(f"Error validating WAV file {wav_path}: {str(e)}")

return False

  

def transcribe_audio(self, audio_path: str, start_time: float = 0.0) -> Dict:

"""

Transcribe a WAV file and return segments with timestamps.

start_time: The offset in seconds for this recording's timestamps

"""

# Validate the WAV file first

if not audio_path.lower().endswith('.wav'):

raise ValueError(f"File {audio_path} is not a WAV file")

if not self.validate_wav_file(audio_path):

raise ValueError(f"Invalid or corrupted WAV file: {audio_path}")

  

print(f"\nTranscribing {os.path.basename(audio_path)}...")

# Transcribe the audio file

segments, info = self.model.transcribe(

audio_path,

word_timestamps=True,

language="en",

task="transcribe",

vad_filter=True # Simple voice activity detection to filter non-speech

)

  

# Convert segments to dictionary format

result = {

"text": "",

"segments": [],

"language": info.language,

"source_file": os.path.basename(audio_path)

}

  

# Process segments and add start_time offset

for segment in segments:

seg_dict = {

"start": segment.start + start_time,

"end": segment.end + start_time,

"text": segment.text,

"words": [

{

"word": word.word,

"start": word.start + start_time,

"end": word.end + start_time,

"probability": word.probability

}

for word in (segment.words or [])

]

}

result["segments"].append(seg_dict)

result["text"] += segment.text + " "

  

result["text"] = result["text"].strip()

return result

  

@staticmethod

def format_timestamp(seconds: float) -> str:

"""Convert seconds to HH:MM:SS format"""

return str(datetime.timedelta(seconds=seconds)).split('.')[0]

  

def save_transcript(self, result: Dict, output_path: str):

"""Save the transcript with timestamps in markdown format"""

os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

with open(output_path, 'w', encoding='utf-8') as f:

f.write(f"# Transcript: {result['source_file']}\n\n")

for segment in result["segments"]:

start_time = self.format_timestamp(segment["start"])

end_time = self.format_timestamp(segment["end"])

f.write(f'**[{start_time} -> {end_time}]** {segment["text"].strip()}\n\n')

  

def save_json(self, result: Dict, output_path: str):

"""Save the full transcription result as JSON"""

os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

with open(output_path, 'w', encoding='utf-8') as f:

json.dump(result, f, indent=2, ensure_ascii=False)

  

def merge_transcripts(transcript_files: List[Dict], output_path: str):

"""

Merge multiple transcripts and sort by timestamp

transcript_files should be a list of dictionaries from whisper transcriptions

"""

# Collect all segments

all_segments = []

for transcript in transcript_files:

# Add source file info to each segment for better tracking

for segment in transcript["segments"]:

segment["source"] = transcript["source_file"]

all_segments.extend(transcript["segments"])

  

# Sort segments by start time

all_segments.sort(key=lambda x: x["start"])

  

# Create merged result and save in markdown format

os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

with open(output_path, 'w', encoding='utf-8') as f:

for segment in all_segments:

start_stamp = WhisperTranscriber.format_timestamp(segment["start"])

end_stamp = WhisperTranscriber.format_timestamp(segment["end"])

source = "microphone" if "microphone" in segment["source"].lower() else "system"

f.write(f'[{start_stamp} → {end_stamp}] ({source}): {segment["text"].strip()}\n')

  

def main():

# Your input files
# NOTE: these will be the audio input files from the server event, see above for rules on whent to use which file

mic_file = '/Users/todd/Desktop/untitled folder/20241216101704_C6A49F0E_microphone_cleaned.wav'

system_file = '/Users/todd/Desktop/untitled folder/20241216101704_C6A49F0E_system_audio.wav'

# Initialize transcriber

transcriber = WhisperTranscriber(model_name="base.en", device="cpu")

  

# Both files start at the same time (based on timestamp in filename)

audio_files = [

{"path": mic_file, "start_time": 0.0},

{"path": system_file, "start_time": 0.0}

]

  

results = []

for audio_info in audio_files:

try:

# Transcribe each file

result = transcriber.transcribe_audio(

audio_info["path"],

start_time=audio_info["start_time"]

)

results.append(result)

# Save individual transcripts in the same directory as the input files

output_dir = os.path.dirname(audio_info["path"])

base_name = os.path.splitext(os.path.basename(audio_info["path"]))[0]

transcriber.save_transcript(result, os.path.join(output_dir, f"{base_name}_transcript.md"))

print(f"Successfully transcribed {os.path.basename(audio_info['path'])}")

except Exception as e:

print(f"Error processing {os.path.basename(audio_info['path'])}: {str(e)}")

if results:

# Save merged transcript in the same directory as the input files

output_dir = os.path.dirname(mic_file)

merge_transcripts(results, os.path.join(output_dir, "merged_transcript.md"))

print("\nTranscription and merging complete!")

else:

print("\nNo transcripts were generated due to errors.")

  

if __name__ == "__main__":

main()
```