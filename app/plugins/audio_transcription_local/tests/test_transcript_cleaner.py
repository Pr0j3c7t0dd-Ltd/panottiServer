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


def test_remove_filler_words_edge_cases(transcript_cleaner):
    # Test with no filler words
    text = "This is a clean sentence."
    expected = "this is a clean sentence."
    assert transcript_cleaner.remove_filler_words(text) == expected

    # Test with only filler words
    text = "Um, you know, like,"
    expected = ""
    assert transcript_cleaner.remove_filler_words(text) == expected

    # Test with context-dependent words
    text = "Well, this is like a test, right?"
    expected = "well, this is a test, right?"
    assert transcript_cleaner.remove_filler_words(text) == expected


def test_handle_phrase_repetitions_edge_cases(transcript_cleaner):
    # Test with no repetitions
    text = "This is a unique sentence."
    expected = "This is a unique sentence."
    assert transcript_cleaner.handle_phrase_repetitions(text) == expected

    # Test with complex repetitions
    text = "This is a test. This is a test. This is a test."
    expected = "This is a test. This is a test. This is a test."
    assert transcript_cleaner.handle_phrase_repetitions(text) == expected

    # Test with empty input
    text = ""
    expected = ""
    assert transcript_cleaner.handle_phrase_repetitions(text) == expected

    # Test with single word
    text = "Test"
    expected = "Test"
    assert transcript_cleaner.handle_phrase_repetitions(text) == expected

    # Test with repeated phrases
    text = "Test test test. Test test test."
    expected = "Test test. Test test."
    assert transcript_cleaner.handle_phrase_repetitions(text) == expected

    # Test with non-repeated phrases
    text = "This is a test. That is a test."
    expected = "This is a test. That is a test."
    assert transcript_cleaner.handle_phrase_repetitions(text) == expected

    # Test with phrases that should not repeat
    text = "This is a test. This is not a test."
    expected = "This is a test. This is not a test."
    assert transcript_cleaner.handle_phrase_repetitions(text) == expected

    # Test with partial repetitions
    text = "This is a test. This is a test of the system."
    expected = "This is a test. This is a test of the system."
    assert transcript_cleaner.handle_phrase_repetitions(text) == expected

    # Test with punctuation
    text = "This is a test, test, test."
    expected = "This is a test, test."
    assert transcript_cleaner.handle_phrase_repetitions(text) == expected

    # Test with leading and trailing spaces
    text = "   This is a test.   "
    expected = "This is a test"
    assert transcript_cleaner.handle_phrase_repetitions(text) == expected


def test_clean_transcript_edge_cases(transcript_cleaner):
    # Test with mixed content
    text = "Um, this is a test, you know, like, this is a test."
    expected = "This is a test"
    assert transcript_cleaner.clean_transcript(text) == expected

    # Test with punctuation and capitalization
    text = "Hello! Um, this is a test, you know."
    expected = "Hello! this is a test."
    assert transcript_cleaner.clean_transcript(text) == expected

    # Test with capitalization
    text = "this is a test."
    expected = "This is a test."
    assert transcript_cleaner.clean_transcript(text) == expected 