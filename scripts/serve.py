"""FastAPI serving entry point.

Usage:
    python scripts/serve.py

Configuration is read from configs/serving.yaml (environment-specific settings)
and configs/default.yaml (model architecture + data paths).

Override model paths via CLI flags if needed:
    python scripts/serve.py --student-tower outputs/checkpoints/student_tower_best.keras \\
                            --scholarship-tower outputs/checkpoints/scholarship_tower_best.keras
"""
import argparse

import uvicorn

from src.serving.inference_engine import InferenceEngine, ServingConfig
from src.serving.api import create_app


def parse_args():
    parser = argparse.ArgumentParser(description="Start recommendation serving API")
    parser.add_argument(
        "--student-tower",
        type=str,
        default=None,
        help="Override student tower checkpoint path (defaults from configs/serving.yaml)",
    )
    parser.add_argument(
        "--scholarship-tower",
        type=str,
        default=None,
        help="Override scholarship tower checkpoint path (defaults from configs/serving.yaml)",
    )
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--serving-config", type=str, default="configs/serving.yaml")
    return parser.parse_args()


def main():
    args = parse_args()

    # Load serving configuration (environment-specific settings)
    serving_cfg = ServingConfig(args.serving_config)

    # Allow CLI flags to override config file values
    student_tower_path = args.student_tower or serving_cfg.student_tower_path
    scholarship_tower_path = args.scholarship_tower or serving_cfg.scholarship_tower_path

    # Build and initialize inference engine (loads models, warms up SBERT, caches scholarships)
    print(f"Initializing InferenceEngine (environment: {serving_cfg.environment}) ...")
    engine = InferenceEngine(
        student_tower_path=student_tower_path,
        scholarship_tower_path=scholarship_tower_path,
        config_path=args.config,
        serving_config_path=args.serving_config,
    )
    engine.initialize()

    # Create FastAPI app bound to this engine instance
    app = create_app(engine)

    # Start uvicorn server with config-driven host/port
    print(f"Starting server on {serving_cfg.server_host}:{serving_cfg.server_port}")
    uvicorn.run(
        app,
        host=serving_cfg.server_host,
        port=serving_cfg.server_port,
    )


if __name__ == "__main__":
    main()