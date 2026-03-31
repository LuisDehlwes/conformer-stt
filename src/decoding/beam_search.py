"""Beam search CTC decoding with optional language model."""

import numpy as np
import torch

from ..preprocessing.tokenizer import CharTokenizer


class BeamSearchDecoder:
    """CTC Beam Search decoder with optional n-gram language model.

    Uses pyctcdecode for efficient beam search with KenLM integration.
    Falls back to a pure-Python beam search if pyctcdecode is not available.
    """

    def __init__(
        self,
        tokenizer: CharTokenizer,
        beam_width: int = 20,
        lm_path: str | None = None,
        lm_alpha: float = 0.5,
        lm_beta: float = 1.0,
    ):
        self.tokenizer = tokenizer
        self.beam_width = beam_width
        self._decoder = None

        # Try to use pyctcdecode
        try:
            from pyctcdecode import build_ctcdecoder

            vocab_list = tokenizer.get_vocab_list()
            # pyctcdecode expects "" as blank token at index 0
            vocab_list[0] = ""

            kenlm_model = None
            if lm_path is not None:
                kenlm_model = lm_path

            self._decoder = build_ctcdecoder(
                labels=vocab_list,
                kenlm_model_path=kenlm_model,
                alpha=lm_alpha,
                beta=lm_beta,
            )
            self._use_pyctcdecode = True
        except ImportError:
            self._use_pyctcdecode = False

    def decode(self, log_probs: torch.Tensor, lengths: torch.Tensor | None = None
               ) -> list[str]:
        """Decode a batch of log probabilities.

        Args:
            log_probs: (B, T, vocab_size)
            lengths: (B,) actual lengths

        Returns:
            List of decoded strings.
        """
        batch_size = log_probs.size(0)
        results = []

        for i in range(batch_size):
            if lengths is not None:
                length = lengths[i].item()
                probs = log_probs[i, :length]
            else:
                probs = log_probs[i]

            text = self.decode_single(probs)
            results.append(text)

        return results

    def decode_single(self, log_probs: torch.Tensor) -> str:
        """Decode a single utterance.

        Args:
            log_probs: (T, vocab_size)
        """
        if self._use_pyctcdecode and self._decoder is not None:
            # pyctcdecode expects numpy log probs
            logits_np = log_probs.cpu().numpy()
            text = self._decoder.decode(logits_np, beam_width=self.beam_width)
            return text
        else:
            return self._simple_beam_search(log_probs)

    def _simple_beam_search(self, log_probs: torch.Tensor) -> str:
        """Simple beam search without LM (fallback)."""
        T, V = log_probs.shape
        blank_id = self.tokenizer.blank_id

        # Each beam: (log_prob, prefix_tokens)
        beams = [(0.0, [])]

        for t in range(T):
            new_beams: dict[tuple, float] = {}

            for beam_prob, prefix in beams:
                for v in range(V):
                    token_prob = log_probs[t, v].item()
                    new_prob = beam_prob + token_prob

                    if v == blank_id:
                        key = tuple(prefix)
                    elif len(prefix) > 0 and prefix[-1] == v:
                        # Repeat token → collapse
                        key = tuple(prefix)
                    else:
                        key = tuple(prefix + [v])

                    if key not in new_beams or new_beams[key] < new_prob:
                        new_beams[key] = new_prob

            # Keep top beams
            sorted_beams = sorted(
                new_beams.items(), key=lambda x: x[1], reverse=True
            )
            beams = [
                (prob, list(tokens))
                for tokens, prob in sorted_beams[:self.beam_width]
            ]

        # Return best beam
        if beams:
            best_tokens = beams[0][1]
            return self.tokenizer.decode(
                best_tokens, remove_blanks=False, collapse_repeats=False
            )
        return ""
