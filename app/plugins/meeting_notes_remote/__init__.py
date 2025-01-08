"""Meeting notes plugin for generating meeting notes from transcripts using
Ollama LLM"""

from .plugin import MeetingNotesRemotePlugin

Plugin = MeetingNotesRemotePlugin

__all__ = ["Plugin"] 