"""CTC Loss wrapper with convenient defaults."""

import torch
import torch.nn as nn


class CTCLoss(nn.Module):
    """CTC Loss wrapper for speech recognition training."""

    def __init__(self, blank_id: int = 0, reduction: str = "mean",
                 zero_infinity: bool = True):
        super().__init__()
        self.ctc_loss = nn.CTCLoss(
            blank=blank_id,
            reduction=reduction,
            zero_infinity=zero_infinity,
        )

    def forward(
        self,
        log_probs: torch.Tensor,
        targets: torch.Tensor,
        input_lengths: torch.Tensor,
        target_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Compute CTC loss.

        Args:
            log_probs: (B, T, vocab_size) - log probabilities from model
            targets: (B, S) - padded target token sequences
            input_lengths: (B,) - actual encoder output lengths
            target_lengths: (B,) - actual target lengths

        Returns:
            Scalar loss value.
        """
        # CTC loss expects (T, B, vocab_size)
        log_probs = log_probs.transpose(0, 1)

        return self.ctc_loss(log_probs, targets, input_lengths, target_lengths)
