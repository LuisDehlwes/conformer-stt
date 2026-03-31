"""Tests for preprocessing components."""

import pytest
import torch

from src.preprocessing.tokenizer import CharTokenizer
from src.preprocessing.feature_extractor import FeatureExtractor
from src.preprocessing.augmentation import SpecAugment


class TestCharTokenizer:
    def setup_method(self):
        self.tokenizer = CharTokenizer(lowercase=True)

    def test_vocab_size(self):
        assert len(self.tokenizer) == 29  # blank + 26 letters + apostrophe + space

    def test_encode_simple(self):
        tokens = self.tokenizer.encode("hello")
        assert tokens == [8, 5, 12, 12, 15]  # h=8, e=5, l=12, l=12, o=15

    def test_encode_with_space(self):
        tokens = self.tokenizer.encode("hi there")
        assert 28 in tokens  # space token

    def test_encode_uppercase(self):
        tokens = self.tokenizer.encode("HELLO")
        expected = self.tokenizer.encode("hello")
        assert tokens == expected

    def test_decode_basic(self):
        tokens = [8, 5, 12, 12, 15]
        text = self.tokenizer.decode(tokens, remove_blanks=True, collapse_repeats=False)
        assert text == "hello"

    def test_decode_with_blanks(self):
        tokens = [0, 8, 0, 5, 0, 12, 12, 15, 0]
        text = self.tokenizer.decode(tokens, remove_blanks=True, collapse_repeats=True)
        assert text == "helo"  # collapse_repeats removes the double l

    def test_decode_ctc(self):
        # Typical CTC output: repeated tokens with blanks
        tokens = [8, 8, 0, 5, 5, 0, 12, 12, 0, 12, 0, 15]
        text = self.tokenizer.decode(tokens, remove_blanks=True, collapse_repeats=True)
        assert text == "hello"

    def test_encode_apostrophe(self):
        tokens = self.tokenizer.encode("it's")
        assert 27 in tokens  # apostrophe token

    def test_roundtrip(self):
        original = "hello world"
        tokens = self.tokenizer.encode(original)
        decoded = self.tokenizer.decode(tokens, remove_blanks=True, collapse_repeats=False)
        assert decoded == original

    def test_get_vocab_list(self):
        vocab = self.tokenizer.get_vocab_list()
        assert len(vocab) == 29
        assert vocab[0] == "<blank>"


class TestFeatureExtractor:
    def setup_method(self):
        self.extractor = FeatureExtractor(
            sample_rate=16000, n_mels=80, n_fft=512,
            win_length=400, hop_length=160,
        )

    def test_extract_shape(self):
        waveform = torch.randn(1, 16000)  # 1 second of audio
        features = self.extractor.extract(waveform)
        assert features.dim() == 2
        assert features.shape[0] == 80  # n_mels

    def test_extract_1d_input(self):
        waveform = torch.randn(16000)  # 1D input
        features = self.extractor.extract(waveform)
        assert features.dim() == 2
        assert features.shape[0] == 80

    def test_output_length(self):
        length = self.extractor.get_output_length(16000)
        # (16000 - 400) / 160 + 1 = 98
        assert length == 98

    def test_from_config(self):
        config = {
            "features": {
                "sample_rate": 16000,
                "n_mels": 40,
                "n_fft": 512,
                "win_length": 400,
                "hop_length": 160,
            }
        }
        extractor = FeatureExtractor.from_config(config)
        assert extractor.n_mels == 40


class TestSpecAugment:
    def test_spec_augment_shape(self):
        spec = torch.randn(80, 100)  # (n_mels, time)
        augmenter = SpecAugment(freq_masks=2, freq_mask_width=10,
                                 time_masks=2, time_mask_ratio=0.1)
        augmented = augmenter(spec)
        assert augmented.shape == spec.shape

    def test_spec_augment_has_zeros(self):
        spec = torch.ones(80, 100)
        augmenter = SpecAugment(freq_masks=2, freq_mask_width=10,
                                 time_masks=2, time_mask_ratio=0.1)
        augmented = augmenter(spec)
        # Should have some masked (zero) regions
        assert (augmented == 0).any()

    def test_spec_augment_preserves_original(self):
        spec = torch.randn(80, 100)
        original = spec.clone()
        augmenter = SpecAugment()
        _ = augmenter(spec)
        # Original should be unchanged
        assert torch.equal(spec, original)
