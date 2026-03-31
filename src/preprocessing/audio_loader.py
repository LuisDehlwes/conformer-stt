"""Audio loading and resampling utilities."""

import io

import numpy as np
import soundfile as sf
import torch
import librosa


class AudioLoader:
    """Loads audio files and converts them to a standard format."""

    def __init__(self, target_sample_rate: int = 16000):
        self.target_sample_rate = target_sample_rate

    def load(self, audio_path: str) -> tuple[torch.Tensor, int]:
        """Load an audio file and return (waveform, sample_rate).

        Returns a mono waveform resampled to target_sample_rate.
        """
        data, sr = sf.read(audio_path, dtype="float32")

        # Convert to mono if multi-channel
        if data.ndim > 1:
            data = data.mean(axis=1)

        # Resample if needed
        if sr != self.target_sample_rate:
            data = librosa.resample(data, orig_sr=sr, target_sr=self.target_sample_rate)

        waveform = torch.from_numpy(data).unsqueeze(0)  # (1, num_samples)
        return waveform, self.target_sample_rate

    def load_from_bytes(self, audio_bytes: bytes) -> tuple[torch.Tensor, int]:
        """Load audio from raw bytes."""
        buffer = io.BytesIO(audio_bytes)
        data, sr = sf.read(buffer, dtype="float32")

        if data.ndim > 1:
            data = data.mean(axis=1)

        if sr != self.target_sample_rate:
            data = librosa.resample(data, orig_sr=sr, target_sr=self.target_sample_rate)

        waveform = torch.from_numpy(data).unsqueeze(0)
        return waveform, self.target_sample_rate

    def get_duration(self, waveform: torch.Tensor) -> float:
        """Get duration of waveform in seconds."""
        return waveform.shape[-1] / self.target_sample_rate
