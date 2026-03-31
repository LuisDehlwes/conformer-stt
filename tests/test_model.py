"""Tests for model components."""

import pytest
import torch

from src.model.subsampling import ConvSubsampling
from src.model.conformer import ConformerBlock, PositionalEncoding
from src.model.encoder import ConformerEncoder
from src.model.decoder import CTCDecoder
from src.model.model import STTModel


class TestConvSubsampling:
    def test_output_shape(self):
        sub = ConvSubsampling(input_dim=80, d_model=256)
        x = torch.randn(2, 80, 200)  # (B, n_mels, T)
        lengths = torch.tensor([200, 150])
        out, new_lengths = sub(x, lengths)
        assert out.dim() == 3
        assert out.shape[0] == 2
        assert out.shape[2] == 256  # d_model

    def test_length_reduction(self):
        sub = ConvSubsampling(input_dim=80, d_model=256)
        lengths = torch.tensor([200])
        _, new_lengths = sub(torch.randn(1, 80, 200), lengths)
        assert new_lengths[0].item() == sub.get_output_length(200)
        # Should be roughly T/4
        assert new_lengths[0].item() <= 200 // 4 + 2


class TestConformerBlock:
    def test_output_shape(self):
        block = ConformerBlock(d_model=64, num_heads=4, feed_forward_dim=128,
                                conv_kernel_size=15, dropout=0.0)
        x = torch.randn(2, 50, 64)
        out = block(x)
        assert out.shape == x.shape

    def test_with_padding_mask(self):
        block = ConformerBlock(d_model=64, num_heads=4, feed_forward_dim=128,
                                conv_kernel_size=15, dropout=0.0)
        x = torch.randn(2, 50, 64)
        mask = torch.zeros(2, 50, dtype=torch.bool)
        mask[1, 30:] = True  # Second sample is shorter
        out = block(x, key_padding_mask=mask)
        assert out.shape == x.shape


class TestConformerEncoder:
    def test_output_shape(self):
        encoder = ConformerEncoder(
            input_dim=80, d_model=64, num_layers=2, num_heads=4,
            feed_forward_dim=128, conv_kernel_size=15, dropout=0.0,
        )
        features = torch.randn(2, 80, 200)
        lengths = torch.tensor([200, 150])
        out, new_lengths = encoder(features, lengths)
        assert out.dim() == 3
        assert out.shape[0] == 2
        assert out.shape[2] == 64

    def test_from_config(self):
        config = {
            "input_dim": 80, "d_model": 64, "num_layers": 2,
            "num_heads": 4, "feed_forward_dim": 128,
            "conv_kernel_size": 15, "dropout": 0.0,
        }
        encoder = ConformerEncoder.from_config(config)
        assert encoder.d_model == 64


class TestCTCDecoder:
    def test_output_shape(self):
        decoder = CTCDecoder(d_model=64, vocab_size=29)
        x = torch.randn(2, 50, 64)
        logits = decoder(x)
        assert logits.shape == (2, 50, 29)

    def test_log_probs(self):
        decoder = CTCDecoder(d_model=64, vocab_size=29)
        x = torch.randn(2, 50, 64)
        log_probs = decoder.get_log_probs(x)
        assert log_probs.shape == (2, 50, 29)
        # Log probs should be <= 0
        assert (log_probs <= 0).all()
        # Should sum to ~1 in prob space
        probs = log_probs.exp().sum(dim=-1)
        assert torch.allclose(probs, torch.ones_like(probs), atol=1e-5)


class TestSTTModel:
    def setup_method(self):
        self.config = {
            "model": {
                "encoder": {
                    "input_dim": 80, "d_model": 64, "num_layers": 2,
                    "num_heads": 4, "feed_forward_dim": 128,
                    "conv_kernel_size": 15, "dropout": 0.0,
                },
                "decoder": {"vocab_size": 29},
            }
        }

    def test_forward(self):
        model = STTModel.from_config(self.config)
        features = torch.randn(2, 80, 200)
        lengths = torch.tensor([200, 150])
        log_probs, out_lengths = model(features, lengths)
        assert log_probs.dim() == 3
        assert log_probs.shape[0] == 2
        assert log_probs.shape[2] == 29

    def test_get_logits(self):
        model = STTModel.from_config(self.config)
        features = torch.randn(1, 80, 100)
        lengths = torch.tensor([100])
        logits, out_lengths = model.get_logits(features, lengths)
        assert logits.shape[2] == 29

    def test_count_parameters(self):
        model = STTModel.from_config(self.config)
        params = model.count_parameters()
        assert params > 0

    def test_save_and_load(self, tmp_path):
        model = STTModel.from_config(self.config)
        save_path = str(tmp_path / "test_model.pt")
        model.save(save_path)

        model2 = STTModel.from_config(self.config)
        model2.load(save_path)

        # Check weights are the same
        for p1, p2 in zip(model.parameters(), model2.parameters()):
            assert torch.equal(p1, p2)
