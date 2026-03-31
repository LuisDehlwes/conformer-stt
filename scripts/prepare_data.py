"""Download and prepare LibriSpeech dataset for training.

Creates JSON manifest files with audio paths and transcriptions.

Usage:
    python scripts/prepare_data.py --dataset librispeech --output data/manifests
"""

import argparse
import json
import os
from pathlib import Path


def prepare_librispeech(data_root: str, output_dir: str, splits: list[str] | None = None):
    """Create manifest files from LibriSpeech directory structure.

    LibriSpeech structure:
        data_root/
            train-clean-100/
                <speaker_id>/<chapter_id>/<speaker_id>-<chapter_id>-<utterance_id>.flac
                <speaker_id>/<chapter_id>/<speaker_id>-<chapter_id>.trans.txt
    """
    if splits is None:
        splits = ["train-clean-100", "dev-clean", "test-clean"]

    os.makedirs(output_dir, exist_ok=True)

    for split in splits:
        split_path = Path(data_root) / split
        if not split_path.exists():
            print(f"Split not found: {split_path}, skipping...")
            continue

        manifest_name = split.replace("-", "_") + ".json"
        manifest_path = Path(output_dir) / manifest_name
        entries = []

        # Walk through all transcript files
        for trans_file in sorted(split_path.rglob("*.trans.txt")):
            with open(trans_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split(" ", 1)
                    if len(parts) != 2:
                        continue
                    utt_id, text = parts
                    text = text.strip().lower()

                    # Find audio file
                    audio_file = trans_file.parent / f"{utt_id}.flac"
                    if not audio_file.exists():
                        continue

                    # Get duration (approximate from file size if torchaudio not available)
                    try:
                        import torchaudio

                        info = torchaudio.info(str(audio_file))
                        duration = info.num_frames / info.sample_rate
                    except Exception:
                        # Rough estimate: FLAC ~= 500kbps at 16kHz
                        duration = audio_file.stat().st_size / 64000

                    entries.append({
                        "audio_path": str(audio_file.resolve()),
                        "text": text,
                        "duration": round(duration, 2),
                        "speaker_id": utt_id.split("-")[0],
                    })

        # Write manifest (one JSON per line)
        with open(manifest_path, "w", encoding="utf-8") as f:
            for entry in entries:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        print(f"Created {manifest_path}: {len(entries)} utterances")


def prepare_common_voice(data_root: str, output_dir: str, language: str = "en"):
    """Create manifest from Mozilla Common Voice TSV files."""
    import csv

    os.makedirs(output_dir, exist_ok=True)

    for split in ["train", "dev", "test"]:
        tsv_path = Path(data_root) / f"{split}.tsv"
        clips_dir = Path(data_root) / "clips"

        if not tsv_path.exists():
            print(f"TSV not found: {tsv_path}, skipping...")
            continue

        manifest_path = Path(output_dir) / f"cv_{split}.json"
        entries = []

        with open(tsv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                audio_file = clips_dir / row["path"]
                if not audio_file.exists():
                    # Try .mp3 extension
                    audio_file = clips_dir / (row["path"].rsplit(".", 1)[0] + ".mp3")
                    if not audio_file.exists():
                        continue

                text = row.get("sentence", "").strip().lower()
                if not text:
                    continue

                entries.append({
                    "audio_path": str(audio_file.resolve()),
                    "text": text,
                    "duration": 0.0,  # Will be computed during loading
                    "speaker_id": row.get("client_id", "unknown"),
                })

        with open(manifest_path, "w", encoding="utf-8") as f:
            for entry in entries:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        print(f"Created {manifest_path}: {len(entries)} utterances")


def create_dummy_manifest(output_dir: str):
    """Create a small dummy manifest for testing the pipeline."""
    os.makedirs(output_dir, exist_ok=True)

    # Create dummy audio files
    try:
        import torch
        import soundfile as sf

        audio_dir = Path("data/raw/dummy")
        audio_dir.mkdir(parents=True, exist_ok=True)

        dummy_texts = [
            "hello world",
            "this is a test",
            "speech recognition is awesome",
            "the quick brown fox jumps over the lazy dog",
            "how are you doing today",
        ]

        entries = []
        for i, text in enumerate(dummy_texts):
            # Create a short sine wave as dummy audio
            sample_rate = 16000
            duration = len(text.split()) * 0.4 + 0.5  # rough estimate
            num_samples = int(duration * sample_rate)
            waveform = torch.randn(1, num_samples) * 0.1

            audio_path = audio_dir / f"dummy_{i:04d}.wav"
            sf.write(str(audio_path), waveform.squeeze(0).numpy(), sample_rate)

            entries.append({
                "audio_path": str(audio_path.resolve()),
                "text": text,
                "duration": round(duration, 2),
                "speaker_id": "dummy",
            })

        for split_name in ["train", "val", "test"]:
            manifest_path = Path(output_dir) / f"{split_name}.json"
            with open(manifest_path, "w", encoding="utf-8") as f:
                for entry in entries:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            print(f"Created {manifest_path}: {len(entries)} dummy utterances")

    except ImportError as e:
        print(f"torch/torchaudio required to create dummy audio files: {e}")


def main():
    parser = argparse.ArgumentParser(description="Prepare data for STT training")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["librispeech", "common_voice", "dummy"],
        default="dummy",
        help="Dataset to prepare",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="data/raw",
        help="Root directory of raw dataset",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/manifests",
        help="Output directory for manifest files",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="en",
        help="Language code (for Common Voice)",
    )
    args = parser.parse_args()

    if args.dataset == "librispeech":
        prepare_librispeech(args.data_root, args.output)
    elif args.dataset == "common_voice":
        prepare_common_voice(args.data_root, args.output, args.language)
    elif args.dataset == "dummy":
        create_dummy_manifest(args.output)
    else:
        print(f"Unknown dataset: {args.dataset}")


if __name__ == "__main__":
    main()
