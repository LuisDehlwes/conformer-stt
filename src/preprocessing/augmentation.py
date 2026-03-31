"""Data augmentation for speech recognition training."""

import random
import torch
import torchaudio


class SpecAugment:
    """SpecAugment: frequency and time masking on spectrograms.

    Reference: Park et al., "SpecAugment", 2019
    """

    def __init__(
        self,
        freq_masks: int = 2,
        freq_mask_width: int = 27,
        time_masks: int = 10,
        time_mask_ratio: float = 0.05,
    ):
        self.freq_masks = freq_masks
        self.freq_mask_width = freq_mask_width
        self.time_masks = time_masks
        self.time_mask_ratio = time_mask_ratio

    def __call__(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """Apply SpecAugment to a spectrogram.

        Args:
            spectrogram: Tensor of shape (n_mels, time_steps)

        Returns:
            Augmented spectrogram of the same shape.
        """
        spec = spectrogram.clone()
        n_mels, n_time = spec.shape

        # Frequency masking
        for _ in range(self.freq_masks):
            f = random.randint(0, min(self.freq_mask_width, n_mels - 1))
            f0 = random.randint(0, n_mels - f)
            spec[f0:f0 + f, :] = 0.0

        # Time masking
        max_time_mask = max(1, int(n_time * self.time_mask_ratio))
        for _ in range(self.time_masks):
            t = random.randint(0, min(max_time_mask, n_time - 1))
            t0 = random.randint(0, n_time - t)
            spec[:, t0:t0 + t] = 0.0

        return spec


class SpeedPerturb:
    """Randomly change the speed of audio waveforms."""

    def __init__(self, factors: list[float] | None = None, sample_rate: int = 16000):
        self.factors = factors or [0.9, 1.0, 1.1]
        self.sample_rate = sample_rate

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        """Apply random speed perturbation.

        Args:
            waveform: Tensor of shape (1, num_samples) or (num_samples,)

        Returns:
            Speed-perturbed waveform.
        """
        factor = random.choice(self.factors)
        if factor == 1.0:
            return waveform

        squeeze = False
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
            squeeze = True

        effects = [["speed", str(factor)], ["rate", str(self.sample_rate)]]
        augmented, _ = torchaudio.sox_effects.apply_effects_tensor(
            waveform, self.sample_rate, effects, channels_first=True
        )

        if squeeze:
            augmented = augmented.squeeze(0)
        return augmented
