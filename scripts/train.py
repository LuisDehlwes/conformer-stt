"""Train the STT model.

Usage:
    python scripts/train.py --model_config config/model_config.yaml --train_config config/train_config.yaml
    python scripts/train.py --resume models/checkpoints/checkpoint_epoch_10.pt
"""

import argparse
import logging
import sys
from pathlib import Path

import yaml
import torch
from torch.utils.data import DataLoader

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocessing.audio_loader import AudioLoader
from src.preprocessing.feature_extractor import FeatureExtractor
from src.preprocessing.tokenizer import CharTokenizer
from src.preprocessing.augmentation import SpecAugment
from src.preprocessing.dataset import STTDataset, collate_fn
from src.model.model import STTModel
from src.training.trainer import Trainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("train")


def main():
    parser = argparse.ArgumentParser(description="Train STT Model")
    parser.add_argument(
        "--model_config",
        type=str,
        default="config/model_config.yaml",
        help="Path to model config",
    )
    parser.add_argument(
        "--train_config",
        type=str,
        default="config/train_config.yaml",
        help="Path to training config",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    args = parser.parse_args()

    # Load configs
    with open(args.model_config, "r") as f:
        model_config = yaml.safe_load(f)
    with open(args.train_config, "r") as f:
        train_config = yaml.safe_load(f)

    # Merge configs
    config = {**model_config, **train_config}

    train_cfg = config.get("training", {})
    data_cfg = config.get("data", {})
    aug_cfg = config.get("augmentation", {})

    # Set seed
    seed = train_cfg.get("seed", 42)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Build components
    tokenizer = CharTokenizer(
        lowercase=config.get("tokenizer", {}).get("lowercase", True)
    )

    # Update vocab size in model config
    config.setdefault("model", {}).setdefault("decoder", {})["vocab_size"] = len(tokenizer)

    audio_loader = AudioLoader(
        target_sample_rate=config.get("features", {}).get("sample_rate", 16000)
    )
    feature_extractor = FeatureExtractor.from_config(config)

    # Augmentation (only for training)
    train_augment = None
    spec_aug_cfg = aug_cfg.get("spec_augment", {})
    if spec_aug_cfg.get("enabled", False):
        train_augment = SpecAugment(
            freq_masks=spec_aug_cfg.get("freq_masks", 2),
            freq_mask_width=spec_aug_cfg.get("freq_mask_width", 27),
            time_masks=spec_aug_cfg.get("time_masks", 10),
            time_mask_ratio=spec_aug_cfg.get("time_mask_width", 0.05),
        )

    # Datasets
    train_manifest = data_cfg.get("train_manifest", "data/manifests/train.json")
    val_manifest = data_cfg.get("val_manifest", "data/manifests/val.json")

    logger.info(f"Loading training data from {train_manifest}")
    train_dataset = STTDataset(
        manifest_path=train_manifest,
        audio_loader=audio_loader,
        feature_extractor=feature_extractor,
        tokenizer=tokenizer,
        augment=train_augment,
        min_duration=data_cfg.get("min_duration", 0.5),
        max_duration=data_cfg.get("max_duration", 20.0),
    )

    val_dataset = None
    val_loader = None
    if Path(val_manifest).exists():
        logger.info(f"Loading validation data from {val_manifest}")
        val_dataset = STTDataset(
            manifest_path=val_manifest,
            audio_loader=audio_loader,
            feature_extractor=feature_extractor,
            tokenizer=tokenizer,
            augment=None,  # No augmentation for validation
            min_duration=data_cfg.get("min_duration", 0.5),
            max_duration=data_cfg.get("max_duration", 20.0),
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=train_cfg.get("batch_size", 16),
            shuffle=False,
            num_workers=data_cfg.get("num_workers", 4),
            collate_fn=collate_fn,
            pin_memory=data_cfg.get("pin_memory", True),
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg.get("batch_size", 16),
        shuffle=True,
        num_workers=data_cfg.get("num_workers", 4),
        collate_fn=collate_fn,
        pin_memory=data_cfg.get("pin_memory", True),
    )

    logger.info(f"Training samples: {len(train_dataset)}")
    if val_dataset:
        logger.info(f"Validation samples: {len(val_dataset)}")

    # Build model
    model = STTModel.from_config(config)
    logger.info(f"Model parameters: {model.count_parameters():,}")

    # Trainer
    trainer = Trainer(model=model, tokenizer=tokenizer, config=config)

    # Resume if specified
    if args.resume and Path(args.resume).exists():
        trainer.load_checkpoint(args.resume)

    # Train
    trainer.train(train_loader, val_loader)

    logger.info("Training complete!")


if __name__ == "__main__":
    main()
