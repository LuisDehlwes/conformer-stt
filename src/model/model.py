"""Complete STT Model: Encoder + CTC Decoder."""

import torch
import torch.nn as nn

from .encoder import ConformerEncoder
from .decoder import CTCDecoder


class STTModel(nn.Module):
    """End-to-end Speech-to-Text model.

    Conformer Encoder → CTC Decoder (linear projection)
    """

    def __init__(self, encoder: ConformerEncoder, decoder: CTCDecoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, features: torch.Tensor, feature_lengths: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            features: (B, n_mels, T) mel spectrogram
            feature_lengths: (B,) time lengths

        Returns:
            log_probs: (B, T', vocab_size) log probabilities
            output_lengths: (B,) output sequence lengths
        """
        encoder_out, output_lengths = self.encoder(features, feature_lengths)
        log_probs = self.decoder.get_log_probs(encoder_out)
        return log_probs, output_lengths

    def get_logits(self, features: torch.Tensor, feature_lengths: torch.Tensor
                   ) -> tuple[torch.Tensor, torch.Tensor]:
        """Get raw logits (for beam search decoding).

        Returns:
            logits: (B, T', vocab_size)
            output_lengths: (B,)
        """
        encoder_out, output_lengths = self.encoder(features, feature_lengths)
        logits = self.decoder(encoder_out)
        return logits, output_lengths

    @classmethod
    def from_config(cls, config: dict) -> "STTModel":
        """Build model from config dict."""
        model_cfg = config.get("model", config)
        enc_cfg = model_cfg.get("encoder", {})
        dec_cfg = model_cfg.get("decoder", {})

        encoder = ConformerEncoder.from_config(enc_cfg)
        decoder = CTCDecoder(
            d_model=enc_cfg.get("d_model", 256),
            vocab_size=dec_cfg.get("vocab_size", 29),
            dropout=enc_cfg.get("dropout", 0.1),
        )

        return cls(encoder=encoder, decoder=decoder)

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def save(self, path: str):
        """Save model state dict."""
        torch.save(self.state_dict(), path)

    def load(self, path: str, device: str = "cpu"):
        """Load model state dict."""
        state_dict = torch.load(path, map_location=device, weights_only=True)
        self.load_state_dict(state_dict)
