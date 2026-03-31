"""Tests for decoding components."""

import pytest
import torch

from src.decoding.greedy import GreedyDecoder
from src.preprocessing.tokenizer import CharTokenizer


class TestGreedyDecoder:
    def setup_method(self):
        self.tokenizer = CharTokenizer()
        self.decoder = GreedyDecoder(tokenizer=self.tokenizer)

    def test_decode_shape(self):
        # log_probs: (B, T, vocab_size)
        log_probs = torch.randn(2, 50, 29).log_softmax(dim=-1)
        results = self.decoder.decode(log_probs)
        assert len(results) == 2
        assert all(isinstance(r, str) for r in results)

    def test_decode_all_blank(self):
        # Create log_probs where blank has highest prob
        log_probs = torch.full((1, 20, 29), -10.0)
        log_probs[:, :, 0] = 0.0  # blank has highest logit
        log_probs = log_probs.log_softmax(dim=-1)
        results = self.decoder.decode(log_probs)
        assert results[0] == ""

    def test_decode_known_sequence(self):
        # Manually create log_probs for "hi"
        # h=8, i=9, blank=0
        T = 10
        log_probs = torch.full((1, T, 29), -100.0)
        # Frames 0-2: h (index 8)
        log_probs[0, 0:3, 8] = 0.0
        # Frames 3-4: blank
        log_probs[0, 3:5, 0] = 0.0
        # Frames 5-8: i (index 9)
        log_probs[0, 5:9, 9] = 0.0
        # Frame 9: blank
        log_probs[0, 9, 0] = 0.0
        log_probs = log_probs.log_softmax(dim=-1)
        results = self.decoder.decode(log_probs)
        assert results[0] == "hi"

    def test_decode_repeated_chars(self):
        # "aa" requires blank between them in CTC
        T = 6
        log_probs = torch.full((1, T, 29), -100.0)
        # Frames 0-1: a (index 1)
        log_probs[0, 0:2, 1] = 0.0
        # Frame 2: blank
        log_probs[0, 2, 0] = 0.0
        # Frames 3-4: a (index 1)
        log_probs[0, 3:5, 1] = 0.0
        # Frame 5: blank
        log_probs[0, 5, 0] = 0.0
        log_probs = log_probs.log_softmax(dim=-1)
        results = self.decoder.decode(log_probs)
        assert results[0] == "aa"

    def test_batch_decode(self):
        log_probs = torch.randn(4, 30, 29).log_softmax(dim=-1)
        results = self.decoder.decode(log_probs)
        assert len(results) == 4
