"""FastAPI REST API for the STT engine."""

import logging
import time
from pathlib import Path

import torch
import yaml
from fastapi import FastAPI, UploadFile, HTTPException
from pydantic import BaseModel

from ..preprocessing.audio_loader import AudioLoader
from ..preprocessing.feature_extractor import FeatureExtractor
from ..preprocessing.tokenizer import CharTokenizer
from ..model.model import STTModel
from ..decoding.greedy import GreedyDecoder
from ..decoding.beam_search import BeamSearchDecoder
from ..postprocessing.normalization import TextNormalizer
from ..postprocessing.capitalization import TrueCase
from ..postprocessing.punctuation import PunctuationRestorer

logger = logging.getLogger(__name__)


class TranscriptionResponse(BaseModel):
    text: str
    raw_text: str
    duration_seconds: float
    processing_time_seconds: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str


# Global state for loaded model
_engine_state: dict = {}


def create_app(
    model_config_path: str = "config/model_config.yaml",
    inference_config_path: str = "config/inference_config.yaml",
) -> FastAPI:
    """Create and configure the FastAPI application."""

    app = FastAPI(
        title="STT Engine API",
        description="Speech-to-Text transcription service",
        version="0.1.0",
    )

    @app.on_event("startup")
    async def load_model():
        """Load model on startup."""
        try:
            # Load configs
            with open(model_config_path, "r") as f:
                model_config = yaml.safe_load(f)
            with open(inference_config_path, "r") as f:
                inference_config = yaml.safe_load(f)

            inf_cfg = inference_config.get("inference", {})
            dec_cfg = inference_config.get("decoding", {})

            # Device
            device_str = inf_cfg.get("device", "auto")
            if device_str == "auto":
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            else:
                device = torch.device(device_str)

            # Build components
            tokenizer = CharTokenizer(lowercase=True)

            model = STTModel.from_config(model_config)
            model_path = inf_cfg.get("model_path", "models/checkpoints/best_model.pt")
            if Path(model_path).exists():
                checkpoint = torch.load(
                    model_path, map_location=device, weights_only=False
                )
                if "model_state_dict" in checkpoint:
                    model.load_state_dict(checkpoint["model_state_dict"])
                else:
                    model.load_state_dict(checkpoint)
                logger.info(f"Loaded model from {model_path}")
            else:
                logger.warning(
                    f"No model found at {model_path}, using random weights"
                )

            model.to(device)
            model.eval()

            audio_loader = AudioLoader(target_sample_rate=16000)
            feature_extractor = FeatureExtractor.from_config(inference_config)

            # Decoder
            decode_type = dec_cfg.get("type", "greedy")
            if decode_type == "greedy":
                decoder = GreedyDecoder(tokenizer)
            else:
                decoder = BeamSearchDecoder(
                    tokenizer=tokenizer,
                    beam_width=dec_cfg.get("beam_width", 20),
                    lm_path=dec_cfg.get("lm_path"),
                    lm_alpha=dec_cfg.get("lm_alpha", 0.5),
                    lm_beta=dec_cfg.get("lm_beta", 1.0),
                )

            # Postprocessing
            normalizer = TextNormalizer()
            capitalizer = TrueCase()
            punctuator = PunctuationRestorer()

            _engine_state.update({
                "model": model,
                "tokenizer": tokenizer,
                "audio_loader": audio_loader,
                "feature_extractor": feature_extractor,
                "decoder": decoder,
                "normalizer": normalizer,
                "capitalizer": capitalizer,
                "punctuator": punctuator,
                "device": device,
                "loaded": True,
            })

            logger.info(
                f"STT Engine ready on {device} "
                f"({model.count_parameters():,} parameters)"
            )

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            _engine_state["loaded"] = False

    @app.get("/health", response_model=HealthResponse)
    async def health_check():
        return HealthResponse(
            status="ok" if _engine_state.get("loaded") else "error",
            model_loaded=_engine_state.get("loaded", False),
            device=str(_engine_state.get("device", "unknown")),
        )

    @app.post("/transcribe", response_model=TranscriptionResponse)
    async def transcribe(audio: UploadFile):
        """Transcribe an uploaded audio file."""
        if not _engine_state.get("loaded"):
            raise HTTPException(status_code=503, detail="Model not loaded")

        # Validate file type
        if audio.content_type and not audio.content_type.startswith("audio/"):
            # Allow common types without strict checking
            pass

        start_time = time.time()

        try:
            audio_bytes = await audio.read()
            loader: AudioLoader = _engine_state["audio_loader"]
            waveform, sr = loader.load_from_bytes(audio_bytes)
        except Exception as e:
            raise HTTPException(
                status_code=400, detail=f"Failed to load audio: {e}"
            )

        duration = loader.get_duration(waveform)

        # Extract features
        extractor: FeatureExtractor = _engine_state["feature_extractor"]
        features = extractor.extract(waveform)  # (n_mels, T)
        features = features.unsqueeze(0)  # (1, n_mels, T)
        feature_lengths = torch.tensor([features.shape[2]], dtype=torch.long)

        device = _engine_state["device"]
        features = features.to(device)
        feature_lengths = feature_lengths.to(device)

        # Inference
        model: STTModel = _engine_state["model"]
        with torch.no_grad():
            log_probs, output_lengths = model(features, feature_lengths)

        # Decode
        decoder = _engine_state["decoder"]
        if hasattr(decoder, "decode"):
            texts = decoder.decode(log_probs, output_lengths)
            raw_text = texts[0] if texts else ""
        else:
            raw_text = ""

        # Postprocessing
        normalizer: TextNormalizer = _engine_state["normalizer"]
        capitalizer: TrueCase = _engine_state["capitalizer"]
        punctuator: PunctuationRestorer = _engine_state["punctuator"]

        text = normalizer.normalize(raw_text)
        text = punctuator.restore(text)
        text = capitalizer.apply(text)

        processing_time = time.time() - start_time

        return TranscriptionResponse(
            text=text,
            raw_text=raw_text,
            duration_seconds=round(duration, 2),
            processing_time_seconds=round(processing_time, 3),
        )

    return app
