"""Export model to ONNX format for optimized inference.

Usage:
    python scripts/export_model.py --checkpoint models/checkpoints/best_model.pt --output models/exported/stt_model.onnx
"""

import argparse
import logging
import sys
from pathlib import Path

import yaml
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocessing.tokenizer import CharTokenizer
from src.model.model import STTModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("export")


def main():
    parser = argparse.ArgumentParser(description="Export STT model to ONNX")
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint"
    )
    parser.add_argument(
        "--model_config", type=str, default="config/model_config.yaml"
    )
    parser.add_argument(
        "--output", type=str, default="models/exported/stt_model.onnx"
    )
    args = parser.parse_args()

    with open(args.model_config, "r") as f:
        config = yaml.safe_load(f)

    tokenizer = CharTokenizer(lowercase=True)
    config.setdefault("model", {}).setdefault("decoder", {})["vocab_size"] = len(tokenizer)

    # Load model
    model = STTModel.from_config(config)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    logger.info(f"Model parameters: {model.count_parameters():,}")

    # Create dummy input
    n_mels = config.get("features", {}).get("n_mels", 80)
    dummy_features = torch.randn(1, n_mels, 200)  # ~2 seconds of audio
    dummy_lengths = torch.tensor([200], dtype=torch.long)

    # Export
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        (dummy_features, dummy_lengths),
        str(output_path),
        input_names=["features", "feature_lengths"],
        output_names=["log_probs", "output_lengths"],
        dynamic_axes={
            "features": {0: "batch_size", 2: "time"},
            "feature_lengths": {0: "batch_size"},
            "log_probs": {0: "batch_size", 1: "time"},
            "output_lengths": {0: "batch_size"},
        },
        opset_version=18,
    )

    logger.info(f"Exported ONNX model to {output_path}")
    logger.info(f"File size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
