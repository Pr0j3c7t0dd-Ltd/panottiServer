I want to create a plugin using the plugin architecture in the app that will give a desktop notification to the user when an action is complete.

I want the plugin to listen for the 'meeting_notes.completed' even type (see the /app/plugins/meeting_notes/plugin.py for event info) to trigger the processing. 

 The incoming incoming meeting notes file is in the event's "notes_path". The notifier plugin will pop up a notification (see example code below).

The plugin config should include a boolean for opening the meeting notes.  if this setting is true, then then meeting notes .md file will be opened automatically.  If false, only the notification will show.

The plugin will use terminal-notifier, so make sure the README.md file has instructions on how to install (i.e. brew install terminal-notifier)

The plugin should emit it's own event when the process is complete.  The completed event should pass on the information from the original event.

Ideally the processing should use multi-threading so that processing can happen concurrently.

The working code to be implemented is below, no need to change, apart to incorporate into the plugin structure and app logging.  Ensure you have the correct logging in place.

Ensure you add a detailed README.md to the plugin, with any information needed for additional python package requirements.txt

Look at the current example plugin in the /app/plugins/example ,  /app/plugins/noise_reduction and /app/plugins/audio_transcription directories to get context on how to create the plugin.  Especially pay attention to how worker threads are handled in the noise_reduction plugin as an example.

---
Working Function to implement:

```
import os

import subprocess

  

def notify_and_open(title, message, file_path):

abs_path = os.path.abspath(file_path)

if not os.path.exists(abs_path):

raise FileNotFoundError(f"File not found: {abs_path}")

# Send notification

subprocess.run(['terminal-notifier',

'-title', title,

'-message', message,

'-sound', 'default'])

# Open file

os.system(f'open "{abs_path}"')

  

if __name__ == "__main__":

# Meeting notes from the meeting_notes plguin event
file_path = "meeting_notes.md" 

notify_and_open("Meeting Notes Ready", "Your meeting notes are ready to view.", file_path)
  

# Install first: brew install terminal-notifier
```