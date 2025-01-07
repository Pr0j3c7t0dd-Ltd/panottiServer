"""Meeting notes plugin for generating meeting notes from transcripts using
Ollama LLM"""

from .plugin import MeetingNotesLocalPlugin

Plugin = MeetingNotesLocalPlugin

__all__ = ["Plugin"]
