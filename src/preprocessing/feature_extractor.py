"""Feature extraction: Mel spectrogram and related transforms."""

import torch
import torchaudio


class FeatureExtractor:
    """Extracts log-mel spectrogram features from audio waveforms."""

    def __init__(
        self,
        sample_rate: int = 16000,
        n_mels: int = 80,
        n_fft: int = 512,
        win_length: int = 400,
        hop_length: int = 160,
        f_min: float = 20.0,
        f_max: float = 8000.0,
        normalize: bool = True,
    ):
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.normalize = normalize

        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max,
            power=2.0,
        )

    def extract(self, waveform: torch.Tensor) -> torch.Tensor:
        """Extract log-mel spectrogram features.

        Args:
            waveform: Audio tensor of shape (1, num_samples) or (num_samples,)

        Returns:
            Log-mel spectrogram of shape (n_mels, time_steps)
        """
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        # Compute mel spectrogram
        mel_spec = self.mel_transform(waveform)  # (1, n_mels, time)

        # Log compression
        log_mel = torch.log(mel_spec + 1e-9)

        # Remove batch dim
        log_mel = log_mel.squeeze(0)  # (n_mels, time)

        # Normalize per utterance
        if self.normalize:
            log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-6)

        return log_mel

    def get_output_length(self, input_length: int) -> int:
        """Calculate output time steps for a given input sample length."""
        return (input_length - self.win_length) // self.hop_length + 1

    @classmethod
    def from_config(cls, config: dict) -> "FeatureExtractor":
        """Create FeatureExtractor from a config dict."""
        feat_cfg = config.get("features", config)
        return cls(
            sample_rate=feat_cfg.get("sample_rate", 16000),
            n_mels=feat_cfg.get("n_mels", 80),
            n_fft=feat_cfg.get("n_fft", 512),
            win_length=feat_cfg.get("win_length", 400),
            hop_length=feat_cfg.get("hop_length", 160),
            f_min=feat_cfg.get("f_min", 20.0),
            f_max=feat_cfg.get("f_max", 8000.0),
            normalize=feat_cfg.get("normalize", True),
        )
