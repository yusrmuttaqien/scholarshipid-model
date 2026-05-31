"""CLI tool for initiating model retraining with new CSV data.

Usage:
    python scripts/retrain.py \
        --students students_new.csv \
        --scholarships scholarships_new.csv \
        --feedbacks feedbacks_new.csv \
        --config configs/default.yaml

This loads the trained models, merges new data, and triggers training in-process.
After successful retraining, artifacts are pushed to HuggingFace.

Useful for local testing before deploying to production.
"""
import argparse
import sys

import yaml

from scripts.hf_sync import push_data_artifacts, push_model_artifacts
from src.serving.inference_engine import (
    InferenceEngine,
    _parse_csv_with_json,
    _STUDENT_JSON_COLS,
    _SCHOLARSHIP_JSON_COLS,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Retrain model with new CSV data")
    parser.add_argument("--students", type=str, help="Path to students CSV file")
    parser.add_argument("--scholarships", type=str, help="Path to scholarships CSV file")
    parser.add_argument("--feedbacks", type=str, help="Path to feedbacks CSV file")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    return parser.parse_args()


def main():
    args = parse_args()

    # Validate at least one file provided
    if not any([args.students, args.scholarships, args.feedbacks]):
        print("Error: At least one CSV file is required (--students, --scholarships, or --feedbacks)")
        sys.exit(1)

    # Load config to get paths
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Initialize engine (loads models from checkpoints)
    engine = InferenceEngine(
        student_tower_path=cfg["models"]["student_tower"],
        scholarship_tower_path=cfg["models"]["scholarship_tower"],
        config_path=args.config,
    )
    engine.initialize()

    # Read and parse CSV files using shared parser
    students_csv = None
    scholarships_csv = None
    feedbacks_csv = None

    if args.students:
        with open(args.students) as f:
            students_csv = _parse_csv_with_json(f.read(), _STUDENT_JSON_COLS)
        print(f"Loaded {len(students_csv)} student records")

    if args.scholarships:
        with open(args.scholarships) as f:
            scholarships_csv = _parse_csv_with_json(f.read(), _SCHOLARSHIP_JSON_COLS)
        print(f"Loaded {len(scholarships_csv)} scholarship records")

    if args.feedbacks:
        with open(args.feedbacks) as f:
            feedbacks_csv = _parse_csv_with_json(f.read(), [])
        print(f"Loaded {len(feedbacks_csv)} feedback records")

    # Run retraining
    result = engine.retrain_from_csvs(
        students_csv_text=students_csv,
        scholarships_csv_text=scholarships_csv,
        feedbacks_csv_text=feedbacks_csv,
    )

    if result.get("status") == "done":
        print("\n✅ Retraining completed successfully!")

        # Push updated data + model artifacts to HuggingFace
        print("\nPushing data artifacts...")
        push_data_artifacts(config_path=args.config, message="Auto-push after retraining")
        print("Pushing model artifacts...")
        push_model_artifacts(config_path=args.config, message="Auto-push after retraining")
    else:
        print(f"\n❌ Retraining failed: {result.get('error')}")
        sys.exit(1)


if __name__ == "__main__":
    main()
