"""Character-level tokenizer for CTC-based STT."""


class CharTokenizer:
    """Maps between characters and integer token IDs.

    Vocabulary:
      0 = <blank> (CTC blank)
      1-26 = a-z
      27 = '  (apostrophe)
      28 = <space>
    """

    BLANK = "<blank>"
    SPACE = "<space>"

    def __init__(self, lowercase: bool = True):
        self.lowercase = lowercase
        self._build_vocab()

    def _build_vocab(self):
        self.char_to_id: dict[str, int] = {}
        self.id_to_char: dict[int, str] = {}

        # Index 0 = CTC blank
        self.char_to_id[self.BLANK] = 0
        self.id_to_char[0] = self.BLANK

        # a-z = 1-26
        for i, c in enumerate("abcdefghijklmnopqrstuvwxyz", start=1):
            self.char_to_id[c] = i
            self.id_to_char[i] = c

        # apostrophe = 27
        self.char_to_id["'"] = 27
        self.id_to_char[27] = "'"

        # space = 28
        self.char_to_id[self.SPACE] = 28
        self.id_to_char[28] = " "

        self.blank_id = 0
        self.vocab_size = len(self.char_to_id)

    def encode(self, text: str) -> list[int]:
        """Convert text string to list of token IDs."""
        if self.lowercase:
            text = text.lower()

        tokens = []
        for c in text:
            if c == " ":
                tokens.append(self.char_to_id[self.SPACE])
            elif c in self.char_to_id:
                tokens.append(self.char_to_id[c])
            # Skip unknown characters
        return tokens

    def decode(self, token_ids: list[int], remove_blanks: bool = True,
               collapse_repeats: bool = True) -> str:
        """Convert token IDs back to text string.

        Args:
            token_ids: List of integer token IDs
            remove_blanks: Remove CTC blank tokens
            collapse_repeats: Collapse repeated tokens (CTC decoding)
        """
        chars = []
        prev_id = None

        for tid in token_ids:
            if collapse_repeats and tid == prev_id:
                prev_id = tid
                continue
            if remove_blanks and tid == self.blank_id:
                prev_id = tid
                continue
            if tid in self.id_to_char:
                chars.append(self.id_to_char[tid])
            prev_id = tid

        return "".join(chars)

    def get_vocab_list(self) -> list[str]:
        """Return vocabulary as ordered list (for pyctcdecode)."""
        return [self.id_to_char[i] for i in range(self.vocab_size)]

    def __len__(self) -> int:
        return self.vocab_size
