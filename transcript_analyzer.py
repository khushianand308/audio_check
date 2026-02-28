import re
from collections import Counter

class TranscriptAnalyzer:
    """
    Analyzes transcript text for potential quality issues like hallucinations,
    repetitions, and low lexical diversity.
    """
    def __init__(self):
        # Common filler words in English (can be extended for local dialects)
        self.filler_words = {"uh", "um", "ah", "er", "ha", "like", "you know"}

    def get_lexical_diversity(self, text):
        """
        Computes the Type-Token Ratio (TTR).
        Low diversity often indicates repetitive or broken transcription.
        """
        words = re.findall(r'\b\w+\b', text.lower())
        if not words:
            return 0.0
        return len(set(words)) / len(words)

    def get_repetition_score(self, text):
        """
        Detects excessive word repetitions.
        Returns the ratio of the most frequent word to the total number of words.
        """
        words = re.findall(r'\b\w+\b', text.lower())
        if not words:
            return 0.0
        
        counts = Counter(words)
        most_common_word, count = counts.most_common(1)[0]
        
        # If a single word makes up more than 30% of a long transcript, it's suspicious
        return count / len(words)

    def get_filler_density(self, text):
        """Computes the density of filler words."""
        words = re.findall(r'\b\w+\b', text.lower())
        if not words:
            return 0.0
        
        filler_count = sum(1 for w in words if w in self.filler_words)
        return filler_count / len(words)

    def check_for_garbled_text(self, text):
        """Checks for random characters or very short, non-sensical strings."""
        # Simple heuristic: ratio of non-alphanumeric characters
        if not text:
            return True
        
        non_alphanumeric = len(re.findall(r'[^a-zA-Z0-9\s.,?!]', text))
        return (non_alphanumeric / len(text)) > 0.1

    def analyze(self, text):
        """Performs a full analysis of the transcript text."""
        if not text or len(text.strip()) == 0:
            return {"error": "Empty transcript", "is_reliable": False}

        diversity = self.get_lexical_diversity(text)
        repetition = self.get_repetition_score(text)
        filler_density = self.get_filler_density(text)
        is_garbled = self.check_for_garbled_text(text)

        reasons = []
        is_reliable = True

        if diversity < 0.2 and len(text.split()) > 10:
            is_reliable = False
            reasons.append(f"Very low lexical diversity ({diversity:.2f})")
        
        if repetition > 0.4 and len(text.split()) > 5:
            is_reliable = False
            reasons.append(f"High repetition score ({repetition:.2f})")
            
        if filler_density > 0.2:
            is_reliable = False
            reasons.append(f"High filler word density ({filler_density:.2f})")
            
        if is_garbled:
            is_reliable = False
            reasons.append("Transcript contains garbled text or symbols")

        return {
            "lexical_diversity": float(diversity),
            "repetition_score": float(repetition),
            "filler_density": float(filler_density),
            "is_garbled": is_garbled,
            "is_reliable": is_reliable,
            "reasons": reasons,
            "word_count": len(text.split())
        }
