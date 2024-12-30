"""Meeting notes plugin for generating meeting notes from transcripts using Ollama LLM"""

from .plugin import MeetingNotesPlugin

Plugin = MeetingNotesPlugin

__all__ = ['Plugin']
