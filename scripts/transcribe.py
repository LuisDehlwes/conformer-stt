"""Transcribe a single audio file.

Usage:
    python scripts/transcribe.py --audio path/to/audio.wav
    python scripts/transcribe.py --audio path/to/audio.wav --checkpoint models/checkpoints/best_model.pt
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import yaml
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocessing.audio_loader import AudioLoader
from src.preprocessing.feature_extractor import FeatureExtractor
from src.preprocessing.tokenizer import CharTokenizer
from src.model.model import STTModel
from src.decoding.greedy import GreedyDecoder
from src.decoding.beam_search import BeamSearchDecoder
from src.postprocessing.normalization import TextNormalizer
from src.postprocessing.capitalization import TrueCase
from src.postprocessing.punctuation import PunctuationRestorer

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("transcribe")


def main():
    parser = argparse.ArgumentParser(description="Transcribe audio file")
    parser.add_argument(
        "--audio", type=str, required=True, help="Path to audio file"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="models/checkpoints/best_model.pt",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--model_config", type=str, default="config/model_config.yaml"
    )
    parser.add_argument(
        "--decoder", type=str, choices=["greedy", "beam"], default="greedy"
    )
    parser.add_argument("--beam_width", type=int, default=20)
    parser.add_argument("--lm_path", type=str, default=None)
    args = parser.parse_args()

    # Load config
    with open(args.model_config, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build components
    tokenizer = CharTokenizer(lowercase=True)
    config.setdefault("model", {}).setdefault("decoder", {})["vocab_size"] = len(tokenizer)

    audio_loader = AudioLoader(target_sample_rate=16000)
    feature_extractor = FeatureExtractor.from_config(config)

    # Load model
    model = STTModel.from_config(config)

    if Path(args.checkpoint).exists():
        checkpoint = torch.load(
            args.checkpoint, map_location=device, weights_only=False
        )
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
        logger.info(f"Loaded model from {args.checkpoint}")
    else:
        logger.warning(f"No checkpoint found at {args.checkpoint}, using random weights")

    model.to(device)
    model.eval()

    # Decoder
    if args.decoder == "beam":
        decoder = BeamSearchDecoder(
            tokenizer=tokenizer,
            beam_width=args.beam_width,
            lm_path=args.lm_path,
        )
    else:
        decoder = GreedyDecoder(tokenizer)

    # Postprocessing
    normalizer = TextNormalizer()
    capitalizer = TrueCase()
    punctuator = PunctuationRestorer()

    # Load and process audio
    start_time = time.time()

    waveform, sr = audio_loader.load(args.audio)
    duration = audio_loader.get_duration(waveform)

    features = feature_extractor.extract(waveform)
    features = features.unsqueeze(0).to(device)  # (1, n_mels, T)
    feature_lengths = torch.tensor([features.shape[2]], dtype=torch.long).to(device)

    # Inference
    with torch.no_grad():
        log_probs, output_lengths = model(features, feature_lengths)

    texts = decoder.decode(log_probs, output_lengths)
    raw_text = texts[0] if texts else ""

    # Postprocessing
    text = normalizer.normalize(raw_text)
    text = punctuator.restore(text)
    text = capitalizer.apply(text)

    processing_time = time.time() - start_time

    print(f"\n{'=' * 60}")
    print(f"Audio:     {args.audio}")
    print(f"Duration:  {duration:.2f}s")
    print(f"Time:      {processing_time:.3f}s (RTF: {processing_time / duration:.2f})")
    print(f"Raw:       {raw_text}")
    print(f"Text:      {text}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
