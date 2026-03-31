"""CTC Decoder layer: linear projection from encoder output to vocabulary."""

import torch
import torch.nn as nn


class CTCDecoder(nn.Module):
    """Simple CTC decoder: linear projection to vocabulary size.

    Takes encoder output and projects to character probabilities.
    """

    def __init__(self, d_model: int, vocab_size: int, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Linear(d_model, vocab_size)

    def forward(self, encoder_out: torch.Tensor) -> torch.Tensor:
        """
        Args:
            encoder_out: (B, T, d_model)

        Returns:
            logits: (B, T, vocab_size) - raw logits (pre-softmax)
        """
        x = self.dropout(encoder_out)
        logits = self.projection(x)
        return logits

    def get_log_probs(self, encoder_out: torch.Tensor) -> torch.Tensor:
        """Get log probabilities for CTC loss.

        Args:
            encoder_out: (B, T, d_model)

        Returns:
            log_probs: (B, T, vocab_size)
        """
        logits = self.forward(encoder_out)
        return torch.nn.functional.log_softmax(logits, dim=-1)
