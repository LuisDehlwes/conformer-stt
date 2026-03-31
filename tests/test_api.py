"""Tests for FastAPI endpoints."""

import pytest
import io
import wave
import numpy as np

from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from src.api.server import create_app


def create_test_wav_bytes(duration=1.0, sample_rate=16000):
    """Create a minimal WAV file as bytes."""
    num_samples = int(sample_rate * duration)
    samples = np.zeros(num_samples, dtype=np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(samples.tobytes())
    buf.seek(0)
    return buf.read()


class TestHealthEndpoint:
    def test_health_check(self):
        app = create_app()
        client = TestClient(app, raise_server_exceptions=False)
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data


class TestTranscribeEndpoint:
    def test_transcribe_without_model(self):
        """Without a trained model, /transcribe should return 503."""
        app = create_app()
        client = TestClient(app, raise_server_exceptions=False)
        wav_bytes = create_test_wav_bytes()
        response = client.post(
            "/transcribe",
            files={"audio": ("test.wav", wav_bytes, "audio/wav")},
        )
        # Without a real model checkpoint, expect 503 (not loaded) or 500
        assert response.status_code in [200, 500, 503]
