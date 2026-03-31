"""Simple rule-based punctuation restoration."""

import re


class PunctuationRestorer:
    """Adds basic punctuation to raw STT output using simple heuristics.

    For production use, consider a transformer-based punctuation model.
    """

    # Common sentence-ending patterns
    SENTENCE_END_WORDS = {
        "right", "ok", "okay", "yes", "no", "yeah", "sure", "thanks",
        "please", "goodbye", "bye",
    }

    # Question starter words
    QUESTION_STARTERS = {
        "who", "what", "when", "where", "why", "how", "is", "are",
        "was", "were", "do", "does", "did", "can", "could", "would",
        "should", "will", "shall", "have", "has", "had",
    }

    def restore(self, text: str) -> str:
        """Add basic punctuation to unpunctuated text.

        Args:
            text: Raw lowercase text without punctuation

        Returns:
            Text with basic sentence-ending punctuation.
        """
        if not text.strip():
            return text

        text = text.strip()

        # Simple heuristic: if it starts with a question word, add ?
        first_word = text.split()[0].lower() if text.split() else ""
        if first_word in self.QUESTION_STARTERS:
            if not text.endswith(("?", ".", "!")):
                text = text + "?"
        else:
            if not text.endswith(("?", ".", "!")):
                text = text + "."

        return text
