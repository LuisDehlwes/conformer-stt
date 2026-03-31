"""True-casing: restore capitalization to raw STT output."""


class TrueCase:
    """Restores basic capitalization to text."""

    def apply(self, text: str) -> str:
        """Capitalize first letter of sentences.

        Args:
            text: Input text (possibly all lowercase)

        Returns:
            Text with capitalized sentence beginnings.
        """
        if not text:
            return text

        result = []
        capitalize_next = True

        for char in text:
            if capitalize_next and char.isalpha():
                result.append(char.upper())
                capitalize_next = False
            else:
                result.append(char)

            if char in ".!?":
                capitalize_next = True

        return "".join(result)
