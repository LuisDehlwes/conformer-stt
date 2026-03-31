"""Conformer block: the core building block of the Conformer encoder.

Architecture per block:
    Input → FFN(½) → MHSA → ConvModule → FFN(½) → LayerNorm → Output

Reference: Gulati et al., "Conformer: Convolution-augmented Transformer
for Speech Recognition", 2020
"""

import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 10000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding. x: (B, T, d_model)"""
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class FeedForwardModule(nn.Module):
    """Feed-forward module with expansion factor."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_ff),
            nn.SiLU(),  # Swish activation
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultiHeadSelfAttentionModule(nn.Module):
    """Multi-head self-attention with relative positional encoding."""

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor,
                key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            x: (B, T, d_model)
            key_padding_mask: (B, T), True for padded positions
        """
        residual = x
        x = self.layer_norm(x)
        x, _ = self.attention(x, x, x, key_padding_mask=key_padding_mask)
        x = self.dropout(x)
        return residual + x


class ConvolutionModule(nn.Module):
    """Convolution module in Conformer.

    Pointwise Conv → GLU → Depthwise Conv → BatchNorm → Swish → Pointwise Conv
    """

    def __init__(self, d_model: int, kernel_size: int = 31, dropout: float = 0.1):
        super().__init__()
        assert kernel_size % 2 == 1, "Kernel size must be odd"

        self.layer_norm = nn.LayerNorm(d_model)
        self.pointwise_conv1 = nn.Conv1d(d_model, 2 * d_model, kernel_size=1)
        self.glu = nn.GLU(dim=1)
        self.depthwise_conv = nn.Conv1d(
            d_model, d_model, kernel_size=kernel_size,
            padding=kernel_size // 2, groups=d_model,
        )
        self.batch_norm = nn.BatchNorm1d(d_model)
        self.activation = nn.SiLU()
        self.pointwise_conv2 = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, d_model)"""
        residual = x
        x = self.layer_norm(x)

        # (B, T, d_model) → (B, d_model, T) for Conv1d
        x = x.transpose(1, 2)
        x = self.pointwise_conv1(x)  # (B, 2*d_model, T)
        x = self.glu(x)              # (B, d_model, T)
        x = self.depthwise_conv(x)   # (B, d_model, T)
        x = self.batch_norm(x)
        x = self.activation(x)
        x = self.pointwise_conv2(x)  # (B, d_model, T)
        x = self.dropout(x)

        # Back to (B, T, d_model)
        x = x.transpose(1, 2)

        return residual + x


class ConformerBlock(nn.Module):
    """Single Conformer block.

    FFN(½) → MHSA → Conv → FFN(½) → LayerNorm
    """

    def __init__(
        self,
        d_model: int = 256,
        num_heads: int = 4,
        feed_forward_dim: int = 1024,
        conv_kernel_size: int = 31,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.ffn1 = FeedForwardModule(d_model, feed_forward_dim, dropout)
        self.mhsa = MultiHeadSelfAttentionModule(d_model, num_heads, dropout)
        self.conv = ConvolutionModule(d_model, conv_kernel_size, dropout)
        self.ffn2 = FeedForwardModule(d_model, feed_forward_dim, dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor,
                key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            x: (B, T, d_model)
            key_padding_mask: (B, T), True for padded positions
        """
        # Half-step FFN
        x = x + 0.5 * self.ffn1(x)

        # Multi-head self-attention
        x = self.mhsa(x, key_padding_mask=key_padding_mask)

        # Convolution module
        x = self.conv(x)

        # Half-step FFN
        x = x + 0.5 * self.ffn2(x)

        # Final layer norm
        x = self.layer_norm(x)

        return x
