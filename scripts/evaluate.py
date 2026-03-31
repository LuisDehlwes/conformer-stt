"""Evaluate a trained STT model on a test set.

Usage:
    python scripts/evaluate.py --checkpoint models/checkpoints/best_model.pt
    python scripts/evaluate.py --checkpoint models/checkpoints/best_model.pt --test_manifest data/manifests/test.json
"""

import argparse
import logging
import sys
from pathlib import Path

import yaml
import torch
import jiwer
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocessing.audio_loader import AudioLoader
from src.preprocessing.feature_extractor import FeatureExtractor
from src.preprocessing.tokenizer import CharTokenizer
from src.preprocessing.dataset import STTDataset, collate_fn
from src.model.model import STTModel
from src.decoding.greedy import GreedyDecoder
from src.decoding.beam_search import BeamSearchDecoder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("evaluate")


def main():
    parser = argparse.ArgumentParser(description="Evaluate STT Model")
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint"
    )
    parser.add_argument(
        "--model_config", type=str, default="config/model_config.yaml"
    )
    parser.add_argument(
        "--test_manifest", type=str, default="data/manifests/test.json"
    )
    parser.add_argument(
        "--decoder", type=str, choices=["greedy", "beam"], default="greedy"
    )
    parser.add_argument("--beam_width", type=int, default=20)
    parser.add_argument("--lm_path", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
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
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    logger.info(f"Loaded model from {args.checkpoint}")
    logger.info(f"Model parameters: {model.count_parameters():,}")

    # Decoder
    if args.decoder == "beam":
        decoder = BeamSearchDecoder(
            tokenizer=tokenizer,
            beam_width=args.beam_width,
            lm_path=args.lm_path,
        )
    else:
        decoder = GreedyDecoder(tokenizer)

    # Dataset
    test_dataset = STTDataset(
        manifest_path=args.test_manifest,
        audio_loader=audio_loader,
        feature_extractor=feature_extractor,
        tokenizer=tokenizer,
        augment=None,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    logger.info(f"Test samples: {len(test_dataset)}")

    # Evaluate
    all_refs = []
    all_hyps = []

    with torch.no_grad():
        for batch in test_loader:
            features = batch["features"].to(device)
            feature_lengths = batch["feature_lengths"].to(device)

            log_probs, output_lengths = model(features, feature_lengths)
            hypotheses = decoder.decode(log_probs, output_lengths)

            for i, hyp in enumerate(hypotheses):
                ref = batch["texts"][i].lower()
                all_refs.append(ref)
                all_hyps.append(hyp)

    # Compute metrics
    wer = jiwer.wer(all_refs, all_hyps)
    cer = jiwer.cer(all_refs, all_hyps)

    logger.info(f"{'=' * 50}")
    logger.info(f"Results on {args.test_manifest}:")
    logger.info(f"  WER: {wer:.2%}")
    logger.info(f"  CER: {cer:.2%}")
    logger.info(f"  Samples: {len(all_refs)}")
    logger.info(f"{'=' * 50}")

    # Print some examples
    logger.info("\nExample predictions:")
    for i in range(min(5, len(all_refs))):
        logger.info(f"  REF: {all_refs[i]}")
        logger.info(f"  HYP: {all_hyps[i]}")
        logger.info("")


if __name__ == "__main__":
    main()
