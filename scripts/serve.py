"""FastAPI serving entry point.

Usage:
    python scripts/serve.py

Configuration is read from configs/default.yaml (all config in one place).

Override model paths via CLI flags if needed:
    python scripts/serve.py --student-tower outputs/checkpoints/student_tower_best.keras \\
                            --scholarship-tower outputs/checkpoints/scholarship_tower_best.keras

Before starting, pulls latest data + model artifacts from HuggingFace (checks both repos).
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import signal
import sys

import uvicorn

from scripts.hf_sync import pull_data_artifacts, pull_model_artifacts
from src.serving.inference_engine import InferenceEngine
from src.serving.api import create_app


def parse_args():
    parser = argparse.ArgumentParser(description="Start recommendation serving API")
    parser.add_argument(
        "--student-tower",
        type=str,
        default=None,
        help="Override student tower checkpoint path (defaults from config)",
    )
    parser.add_argument(
        "--scholarship-tower",
        type=str,
        default=None,
        help="Override scholarship tower checkpoint path (defaults from config)",
    )
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    return parser.parse_args()


def main():
    args = parse_args()

    # Pull data + model artifacts from HuggingFace before starting server (blocks until done)
    print("Pulling data artifacts...")
    pull_data_artifacts(config_path=args.config)
    print("Pulling model artifacts...")
    pull_model_artifacts(config_path=args.config)

    # Build and initialize inference engine (loads models, warms up SBERT, caches scholarships)
    # Pass None for paths not provided via CLI — InferenceEngine will resolve defaults from config
    print(f"Initializing InferenceEngine (config: {args.config}) ...")
    engine = InferenceEngine(
        student_tower_path=args.student_tower,
        scholarship_tower_path=args.scholarship_tower,
        config_path=args.config,
    )
    engine.initialize()

    # Create FastAPI app bound to this engine instance
    app = create_app(engine)

    # Start uvicorn server with config-driven host/port
    print(f"Starting server on {engine.serving_config.server_host}:{engine.serving_config.server_port}")

    def shutdown(signum, frame):
        print("\nShutting down gracefully...")
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    try:
        uvicorn.run(
            app,
            host=engine.serving_config.server_host,
            port=engine.serving_config.server_port,
        )
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
