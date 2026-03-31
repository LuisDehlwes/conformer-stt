"""Conformer Encoder: stack of Conformer blocks with subsampling."""

import torch
import torch.nn as nn

from .subsampling import ConvSubsampling
from .conformer import ConformerBlock, PositionalEncoding


class ConformerEncoder(nn.Module):
    """Full Conformer encoder.

    Pipeline: Conv Subsampling → Positional Encoding → N × Conformer Blocks
    """

    def __init__(
        self,
        input_dim: int = 80,
        d_model: int = 256,
        num_layers: int = 12,
        num_heads: int = 4,
        feed_forward_dim: int = 1024,
        conv_kernel_size: int = 31,
        dropout: float = 0.1,
        subsampling_factor: int = 4,
        subsampling_conv_channels: int = 256,
    ):
        super().__init__()
        self.d_model = d_model

        # Convolutional subsampling (4x reduction)
        self.subsampling = ConvSubsampling(
            input_dim=input_dim,
            d_model=d_model,
            conv_channels=subsampling_conv_channels,
        )

        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, dropout=dropout)

        # Stack of Conformer blocks
        self.layers = nn.ModuleList([
            ConformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                feed_forward_dim=feed_forward_dim,
                conv_kernel_size=conv_kernel_size,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

    def forward(self, features: torch.Tensor, feature_lengths: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            features: (B, n_mels, T) - mel spectrogram features
            feature_lengths: (B,) - actual time lengths

        Returns:
            encoder_out: (B, T', d_model) - encoded representations
            output_lengths: (B,) - output time lengths
        """
        # Convolutional subsampling
        x, lengths = self.subsampling(features, feature_lengths)  # (B, T', d_model)

        # Positional encoding
        x = self.pos_encoding(x)

        # Create padding mask: True for positions that should be ignored
        max_len = x.size(1)
        key_padding_mask = torch.arange(max_len, device=x.device).unsqueeze(0) >= lengths.unsqueeze(1)

        # Conformer blocks
        for layer in self.layers:
            x = layer(x, key_padding_mask=key_padding_mask)

        return x, lengths

    @classmethod
    def from_config(cls, config: dict) -> "ConformerEncoder":
        enc_cfg = config.get("encoder", config)
        return cls(
            input_dim=enc_cfg.get("input_dim", 80),
            d_model=enc_cfg.get("d_model", 256),
            num_layers=enc_cfg.get("num_layers", 12),
            num_heads=enc_cfg.get("num_heads", 4),
            feed_forward_dim=enc_cfg.get("feed_forward_dim", 1024),
            conv_kernel_size=enc_cfg.get("conv_kernel_size", 31),
            dropout=enc_cfg.get("dropout", 0.1),
            subsampling_factor=enc_cfg.get("subsampling_factor", 4),
            subsampling_conv_channels=enc_cfg.get("subsampling_conv_channels", 256),
        )
