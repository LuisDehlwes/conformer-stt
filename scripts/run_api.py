"""Launch the STT API server.

Usage:
    python scripts/run_api.py
    python scripts/run_api.py --host 0.0.0.0 --port 8000
"""

import argparse
import sys
from pathlib import Path

import uvicorn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.api.server import create_app


def main():
    parser = argparse.ArgumentParser(description="Run STT API Server")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--model_config", type=str, default="config/model_config.yaml"
    )
    parser.add_argument(
        "--inference_config", type=str, default="config/inference_config.yaml"
    )
    args = parser.parse_args()

    app = create_app(
        model_config_path=args.model_config,
        inference_config_path=args.inference_config,
    )

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
