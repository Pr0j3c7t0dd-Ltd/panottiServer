I want to create a plugin using the plugin architecture in the app that will create meeting notes from a transcription file using a local Ollama LLM.

I want the plugin to listen for the 'transcription.completed' even type (see the /app/plugins/audio_transcription/plugin.py for event info) to trigger the processing. 

 The incoming incoming transcript file is in the event's "output_path". The meeting notes file should be saved in markdown format.  check the event emitted from the 'transcription.completed' to get the file for the final merged transcript file as an input into this plugin.

The plugin config should include a directory to store the meeting note files in.  The format of the meeting note filename should be in the same format as the input files, e.g. <recording_id>_meeting_notes.md.  Example: 20241216140128_2096E65E_meeting_notes.md

The plugin should create it's own table in the sqlite database to manage the meeting note processing and state if required.  The table should capture the recording_id from the events table, although the same recording_id may be used to process multiple files.

The plugin should emit it's own event when the process is complete.  The completed event should pass on the information from the original event, plus the file information for the newly created meeting notes file, including the path to the meeting notes file.

Ideally the processing should use multi-threading so that processing can happen concurrently.

The working code to be implemented is below, no need to change, apart to incorporate into the plugin structure and app logging.  Ensure you have the correct logging in place.

Ensure you add a detailed README.md to the plugin, with any information needed for additional python package requirements.txt

Make sure the plugin.yaml file for the new plugin includes the ollama_url (defaulted to http://localhost:11434/api/generate) and model_name (defaulted to llama3.1:latest).  It should also include num_ctx (defaulted to 128000).  Update the plugin to use these values.

Look at the current example plugin in the /app/plugins/example ,  /app/plugins/noise_reduction and /app/plugins/audio_transcription directories to get context on how to create the plugin.  Especially pay attention to how worker threads are handled in the noise_reduction plugin as an example.

---
Working Function to implement:

```
import json

import re

from datetime import datetime

import requests

  

class MeetingNotesGenerator:

def __init__(self, model_name="llama3.1:latest"):

self.ollama_url = "http://localhost:11434/api/generate"

self.model_name = model_name

def _extract_transcript_lines(self, transcript_text):

transcript_section = transcript_text.split("## Transcript")[1].strip()

lines = []

for line in transcript_section.split('\n'):

if line.strip():

match = re.match(r'\[([\d.]+)s - ([\d.]+)s\] \((.*?)\) (.*)', line.strip())

if match:

start_time, end_time, speaker, content = match.groups()

lines.append({

'start_time': float(start_time),

'end_time': float(end_time),

'speaker': speaker,

'content': content.strip()

})

return lines

  

def _extract_metadata(self, transcript_text):

try:

metadata_section = transcript_text.split("## Recording Metadata")[1].split("## Transcript")[0].strip()

return metadata_section

except IndexError:

return None

  

def generate(self, transcript_text):

transcript_lines = self._extract_transcript_lines(transcript_text)

metadata = self._extract_metadata(transcript_text)

if not transcript_lines:

raise ValueError("Could not extract transcript lines")

prompt = f"""Please analyze the meeting metadata and transcript below and create comprehensive meeting notes in markdown format. Please ensure the notes are clear and concise.

---

  

Meeting Metadata:

  

{metadata}

  

---

  

Meeting Transcript:

  

{transcript_text}

  

---

  

The meeting notes should include the following sections:

  

# Event Title: [event_title from the Meeting Metadata, or if not available, create a meeting title]

  

## Meeting Information

Date: [Convert Meeting Metadata meeting date timestamp into a human readable format, just the date/time, no additional comments]

Duration: [Convert the number of seconds from the transcript into a human-readable format]

Location: [Event provider from the Meeting Metadata, if available]

  

## Attendees

[List each attendee from the <Metadata> event_attendees as a bullet point for each attendee. If no event_attendees are available, use "Unknown".]

  

## Executive Summary

[Provide a brief, high-level overview of the meeting's purpose and key outcomes]

  

## Agenda Items Discussed

[List and elaborate on the main topics discussed]

  

## Notes and Additional Information

[Include any other relevant information, clarifications, or important context]

  

## Key Decisions

[List all decisions made during the meeting]

  

## Risks and Issues

[Document any risks, blockers, or issues raised during the meeting]

  

## Open Questions

[List any questions that remained unanswered or need further discussion]

  

## Action Items

[

List action items in format: "- [Owner/Responsible Person] Action description".

NOTE: If no clear owner is mentioned, use "UNASSIGNED". Example:

- [<Owner Name>] Create project timeline by Friday

- [UNASSIGNED] Review security documentation

]

  

## Next Steps

[Outline immediate next steps and upcoming milestones]

  

## Next Meeting

[If discussed, include details about the next meeting]

"""

try:

response = requests.post(self.ollama_url, json={

"model": self.model_name,

"prompt": prompt,

"stream": False,

"options": {

"num_ctx": 128000

}

})

response.raise_for_status()

return response.json()['response']

except requests.exceptions.RequestException as e:

raise Exception(f"Error generating notes: {str(e)}")

  

def main():

with open('test_transcript.md', 'r') as f:

transcript_text = f.read()

generator = MeetingNotesGenerator()

try:

notes = generator.generate(transcript_text)

print(notes)

with open('meeting_notes.md', 'w') as f:

f.write(notes)

except Exception as e:

print(f"Error: {str(e)}")

  

if __name__ == "__main__":

main()
```