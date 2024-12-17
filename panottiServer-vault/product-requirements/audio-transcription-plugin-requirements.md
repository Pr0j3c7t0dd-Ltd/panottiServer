I want to create a plugin using the plugin architecture in the app that will transcribe an audio file with timestamps. I eventually want to be able to combine multiple transcripts together and order them based on the relative timestamp of the recording to produce a single file with the timestamped transcriptions merged in order. The incoming incoming system audio audio file is in the event's "original_event":"system_audio_path" and the incoming microphone audio is either  in the "original_event":"system_audio_path", or in the event "output_file" (See below for instructions on when to use either of the files).  The transcription file should be saved in markdown format.

I want the plugin to listen for the 'noise_reduction.completed' even type (see the /app/plugins/noise_reduction/plugin.py for event info) to trigger the processing. 

The plugin config should include a directory to store the transcription files in.  The format of the transcription filename should be in the same format as the input files, e.g. <recording_id>_transcript.md.  Example: 20241216140128_2096E65E_transcript.md

The plugin should create it's own table in the sqlite database to manage the processing and state if required.  The table should capture the recording_id from the events table, although the same recording_id may be used to process multiple files.

The plugin should emit it's own event when the process is complete.  The completed event should pass on the information from the original event, plus the file information for the newly created transcript file, including the path to the transcript file.

Ideally the processing should use multi-threading so that processing can happen concurrently.

Include the "system_audio_path" from the original_event if it is not null or empty in the array of files to transcribe.

Also, include the "microphone_audio_path" from the original_event is null or empty in the array of files to transcribe, UNLESS, the 'output_file' from the incoming event is populated, then use that instead of the "microphone_audio_path" in the array of files to transcribe.

The working code to be implemented is below, no need to change, apart to incorporate into the plugin structure and app logging.  Ensure you have the correct logging in place.

Ensure you add a detailed README.md to the plugin, with any information needed for additional python package requirements.txt

Look at the current example plugin in the /app/plugins/example and /app/plugins/noise_reduction directory to get context on how to create the plugin.

---
Working Function to implement:

```
sdsad
```