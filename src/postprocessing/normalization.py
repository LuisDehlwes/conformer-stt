"""Text normalization for STT output."""

import re


class TextNormalizer:
    """Cleans up and normalizes raw STT text output."""

    def normalize(self, text: str) -> str:
        """Apply all normalization steps.

        Args:
            text: Raw transcription text

        Returns:
            Cleaned and normalized text.
        """
        text = self._collapse_whitespace(text)
        text = self._fix_apostrophes(text)
        text = text.strip()
        return text

    def _collapse_whitespace(self, text: str) -> str:
        """Replace multiple spaces with single space."""
        return re.sub(r"\s+", " ", text)

    def _fix_apostrophes(self, text: str) -> str:
        """Fix spacing around apostrophes: i ' m → i'm"""
        text = re.sub(r"\s+'\s+", "'", text)
        text = re.sub(r"\s+'", "'", text)
        text = re.sub(r"'\s+", "'", text)
        return text
