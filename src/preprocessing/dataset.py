"""PyTorch Dataset and DataLoader utilities for STT training."""

import json
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from .audio_loader import AudioLoader
from .feature_extractor import FeatureExtractor
from .tokenizer import CharTokenizer
from .augmentation import SpecAugment


class STTDataset(Dataset):
    """Speech-to-Text dataset that loads audio and returns features + tokens."""

    def __init__(
        self,
        manifest_path: str,
        audio_loader: AudioLoader,
        feature_extractor: FeatureExtractor,
        tokenizer: CharTokenizer,
        augment: SpecAugment | None = None,
        min_duration: float = 0.5,
        max_duration: float = 20.0,
    ):
        self.audio_loader = audio_loader
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer
        self.augment = augment

        # Load manifest
        self.samples = []
        with open(manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line.strip())
                duration = entry.get("duration", 999)
                if min_duration <= duration <= max_duration:
                    self.samples.append(entry)

        # Sort by duration for efficient batching
        self.samples.sort(key=lambda x: x.get("duration", 0))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]

        # Load audio
        waveform, _ = self.audio_loader.load(sample["audio_path"])

        # Extract features
        features = self.feature_extractor.extract(waveform)  # (n_mels, time)

        # Apply augmentation during training
        if self.augment is not None:
            features = self.augment(features)

        # Tokenize text
        tokens = self.tokenizer.encode(sample["text"])
        tokens = torch.tensor(tokens, dtype=torch.long)

        return {
            "features": features,         # (n_mels, time)
            "tokens": tokens,             # (token_len,)
            "feature_length": features.shape[1],
            "token_length": len(tokens),
            "text": sample["text"],
        }


def collate_fn(batch: list[dict]) -> dict:
    """Custom collate function that pads features and tokens to batch."""
    # Transpose features to (time, n_mels) for padding, then back
    features_list = [item["features"].T for item in batch]  # list of (time, n_mels)
    tokens_list = [item["tokens"] for item in batch]

    # Pad features
    features_padded = pad_sequence(features_list, batch_first=True)  # (B, max_time, n_mels)
    features_padded = features_padded.transpose(1, 2)  # (B, n_mels, max_time)

    # Pad tokens
    tokens_padded = pad_sequence(tokens_list, batch_first=True, padding_value=0)

    feature_lengths = torch.tensor(
        [item["feature_length"] for item in batch], dtype=torch.long
    )
    token_lengths = torch.tensor(
        [item["token_length"] for item in batch], dtype=torch.long
    )

    texts = [item["text"] for item in batch]

    return {
        "features": features_padded,         # (B, n_mels, max_time)
        "tokens": tokens_padded,             # (B, max_token_len)
        "feature_lengths": feature_lengths,  # (B,)
        "token_lengths": token_lengths,      # (B,)
        "texts": texts,
    }
