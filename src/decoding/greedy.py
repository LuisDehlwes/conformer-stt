"""Greedy CTC decoding."""

import torch

from ..preprocessing.tokenizer import CharTokenizer


class GreedyDecoder:
    """Greedy CTC decoder: takes argmax at each time step."""

    def __init__(self, tokenizer: CharTokenizer):
        self.tokenizer = tokenizer

    def decode(self, log_probs: torch.Tensor, lengths: torch.Tensor | None = None
               ) -> list[str]:
        """Decode a batch of log probabilities to text.

        Args:
            log_probs: (B, T, vocab_size) log probabilities
            lengths: (B,) actual output lengths (optional)

        Returns:
            List of decoded text strings.
        """
        predicted_ids = log_probs.argmax(dim=-1)  # (B, T)
        batch_size = predicted_ids.size(0)

        results = []
        for i in range(batch_size):
            if lengths is not None:
                length = lengths[i].item()
                tokens = predicted_ids[i, :length].cpu().tolist()
            else:
                tokens = predicted_ids[i].cpu().tolist()

            text = self.tokenizer.decode(
                tokens, remove_blanks=True, collapse_repeats=True
            )
            results.append(text)

        return results

    def decode_single(self, log_probs: torch.Tensor) -> str:
        """Decode a single utterance.

        Args:
            log_probs: (T, vocab_size) or (1, T, vocab_size)
        """
        if log_probs.dim() == 3:
            log_probs = log_probs.squeeze(0)

        predicted_ids = log_probs.argmax(dim=-1).cpu().tolist()
        return self.tokenizer.decode(
            predicted_ids, remove_blanks=True, collapse_repeats=True
        )
