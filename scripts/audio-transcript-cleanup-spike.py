import re
from typing import List, Set, Tuple

class TranscriptCleaner:
    def __init__(self):
        # Common filler words to remove
        self.filler_words = {
            'um', 'uh', 'ah', 'er', 'like', 'you know', 
            'i mean', 'sort of', 'kind of', 'basically'
        }
        
        # Words that might be meaningful and should be kept in some contexts
        self.context_dependent_words = {
            'like', 'so', 'well', 'right'
        }

    def remove_filler_words(self, text: str) -> str:
        """Remove common filler words and clean up remaining punctuation."""
        # Convert to lowercase for processing
        text_lower = text.lower()
        
        # Create pattern for matching filler words with surrounding punctuation
        pattern = r'[,\s]*\b(' + '|'.join(re.escape(word) for word in self.filler_words) + r')\b[,\s]*'
        
        # Remove filler words and their surrounding punctuation
        cleaned = re.sub(pattern, ' ', text_lower, flags=re.IGNORECASE)
        
        # Clean up any remaining multiple commas
        cleaned = re.sub(r',\s*,', ',', cleaned)
        
        # Clean up spaces before punctuation
        cleaned = re.sub(r'\s+([,\.\?!])', r'\1', cleaned)
        
        # Clean up extra spaces
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        # Remove leading/trailing commas from sentences
        cleaned = re.sub(r'(^\s*,\s*)|(\s*,\s*$)', '', cleaned)
        cleaned = re.sub(r'\.\s*,\s*', '. ', cleaned)
        
        return cleaned

    def handle_phrase_repetitions(self, text: str) -> str:
        """Remove repeated phrases and clean up remaining punctuation."""
        # Split text into sentences to preserve sentence boundaries
        sentences = text.split('. ')
        cleaned_sentences = []
        
        for sentence in sentences:
            # Split into words but preserve punctuation
            words = sentence.split()
            if not words:
                continue
                
            result = []
            i = 0
            
            while i < len(words):
                # Look for phrases of up to 4 words that might be repeated
                repeat_found = False
                for phrase_length in range(4, 0, -1):
                    if i + phrase_length * 2 <= len(words):
                        phrase1 = ' '.join(words[i:i+phrase_length])
                        phrase2 = ' '.join(words[i+phrase_length:i+phrase_length*2])
                        
                        # Remove punctuation for comparison
                        clean_phrase1 = re.sub(r'[,\.]', '', phrase1.lower())
                        clean_phrase2 = re.sub(r'[,\.]', '', phrase2.lower())
                        
                        if clean_phrase1 == clean_phrase2:
                            result.extend(words[i:i+phrase_length])
                            i += phrase_length * 2
                            repeat_found = True
                            break
                
                if not repeat_found:
                    if i < len(words):
                        result.append(words[i])
                    i += 1
            
            # Clean up any stranded commas after removing repetitions
            cleaned_sentence = ' '.join(result)
            cleaned_sentence = re.sub(r',\s*,', ',', cleaned_sentence)
            cleaned_sentence = re.sub(r'(^\s*,\s*)|(\s*,\s*$)', '', cleaned_sentence)
            cleaned_sentences.append(cleaned_sentence)
        
        return '. '.join(cleaned_sentences)

    def clean_transcript(self, text: str) -> str:
        """Main function to clean the transcript."""
        # First remove filler words
        cleaned_text = self.remove_filler_words(text)
        
        # Then handle phrase repetitions
        cleaned_text = self.handle_phrase_repetitions(cleaned_text)
        
        # Final cleanup of spaces and punctuation
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
        cleaned_text = re.sub(r'\s+([,\.\?!])', r'\1', cleaned_text)
        
        # Restore proper capitalization at sentence beginnings
        cleaned_text = '. '.join(s.capitalize() for s in cleaned_text.split('. '))
        
        return cleaned_text

def main():
    # Example usage
    cleaner = TranscriptCleaner()
    
    sample_text = """
    Um, so like I was thinking, you know, we could uh maybe try to, 
    try to implement this new feature. It's like, basically going to 
    help us um improve the user experience experience experience.
    """
    
    cleaned = cleaner.clean_transcript(sample_text)
    print("\nOriginal text:")
    print(sample_text)
    print("\nCleaned text:")
    print(cleaned)

if __name__ == "__main__":
    main()