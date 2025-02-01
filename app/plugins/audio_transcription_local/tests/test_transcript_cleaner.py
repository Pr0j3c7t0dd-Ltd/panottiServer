import pytest
from app.plugins.audio_transcription_local.transcript_cleaner import TranscriptCleaner

@pytest.fixture
def transcript_cleaner():
    return TranscriptCleaner()


def test_remove_filler_words(transcript_cleaner):
    text = "Um, this is a test, you know, with filler words."
    expected = "this is a test with filler words."
    assert transcript_cleaner.remove_filler_words(text) == expected


def test_handle_phrase_repetitions(transcript_cleaner):
    text = "This is a test. This is a test."
    expected = "This is a test. This is a test."
    assert transcript_cleaner.handle_phrase_repetitions(text) == expected


def test_clean_transcript(transcript_cleaner):
    text = "Um, this is a test, you know, with filler words. This is a test."
    expected = "This is a test with filler words. This is a test."
    assert transcript_cleaner.clean_transcript(text) == expected 