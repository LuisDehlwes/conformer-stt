"""Convolutional subsampling layer for reducing input sequence length."""

import torch
import torch.nn as nn


class ConvSubsampling(nn.Module):
    """Convolutional subsampling that reduces time dimension by factor of 4.

    Two convolution layers with stride 2 each → 4x downsampling.
    Input:  (B, n_mels, T)
    Output: (B, T//4, d_model)
    """

    def __init__(self, input_dim: int, d_model: int, conv_channels: int = 256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, conv_channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(conv_channels, conv_channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        # After 2x stride-2 convs on freq axis: input_dim -> ceil(input_dim/4)
        # We need to compute the output dim to create the linear projection
        conv_out_freq = self._calc_conv_out_dim(input_dim, 2)
        self.linear = nn.Linear(conv_channels * conv_out_freq, d_model)

    def _calc_conv_out_dim(self, dim: int, num_layers: int) -> int:
        """Calculate output dimension after stride-2 convolutions."""
        for _ in range(num_layers):
            dim = (dim + 2 * 1 - 3) // 2 + 1  # padding=1, kernel=3, stride=2
        return dim

    def forward(self, x: torch.Tensor, lengths: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input features (B, n_mels, T)
            lengths: Original time lengths (B,)

        Returns:
            Subsampled features (B, T', d_model) and new lengths (B,)
        """
        # Reshape for Conv2d: (B, 1, n_mels, T) - treat as single-channel image
        x = x.unsqueeze(1)  # (B, 1, n_mels, T)

        x = self.conv(x)  # (B, conv_channels, n_mels//4, T//4)

        B, C, F, T = x.shape
        x = x.permute(0, 3, 1, 2).reshape(B, T, C * F)  # (B, T//4, conv_channels * F)

        x = self.linear(x)  # (B, T//4, d_model)

        # Update lengths (two stride-2 convolutions)
        new_lengths = lengths
        for _ in range(2):
            new_lengths = (new_lengths + 2 * 1 - 3) // 2 + 1

        return x, new_lengths

    def get_output_length(self, input_length: int) -> int:
        """Calculate output time length for a given input time length."""
        length = input_length
        for _ in range(2):
            length = (length + 2 * 1 - 3) // 2 + 1
        return length
