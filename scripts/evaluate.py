"""
Entry point untuk evaluasi model pada test set.

Usage:
    python scripts/evaluate.py --config configs/default.yaml
    # or override checkpoint paths:
    python scripts/evaluate.py --config configs/default.yaml \
        --student_checkpoint outputs/checkpoints/student_tower_best.keras \
        --scholarship_checkpoint outputs/checkpoints/scholarship_tower_best.keras
"""
import argparse
import yaml
import tensorflow as tf

from src.models.student_tower import L2Normalize
from src.evaluators import Evaluator
from src.utils.data_loader import load_data, load_precomputed_features


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",                  type=str, default="configs/default.yaml")
    parser.add_argument("--student_checkpoint",      type=str, default=None)
    parser.add_argument("--scholarship_checkpoint",  type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Resolve checkpoint paths from CLI or config defaults
    student_checkpoint = args.student_checkpoint if args.student_checkpoint else cfg["models"]["student_tower"]
    scholarship_checkpoint = args.scholarship_checkpoint if args.scholarship_checkpoint else cfg["models"]["scholarship_tower"]

    # ── Load features ─────────────────────────────────────────────────────────
    _, _, test_df = load_data(cfg)
    (stu_struct, sch_struct, stu_text_emb, sch_text_emb,
     stu_id_to_idx, sch_id_to_idx) = load_precomputed_features(cfg)

    sch_ids = list(sch_id_to_idx.keys())

    # ── Load model dari format .keras ─────────────────────────────────────────
    custom_objects = {"L2Normalize": L2Normalize}
    student_tower     = tf.keras.models.load_model(
        student_checkpoint, custom_objects=custom_objects)
    scholarship_tower = tf.keras.models.load_model(
        scholarship_checkpoint, custom_objects=custom_objects)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    evaluator = Evaluator(k=cfg["evaluation"]["k_values"][0])
    metrics   = evaluator.compute_metrics(
        df=test_df,
        student_tower=student_tower,
        scholarship_tower=scholarship_tower,
        stu_struct=stu_struct,
        sch_struct=sch_struct,
        stu_text_emb=stu_text_emb,
        sch_text_emb=sch_text_emb,
        stu_id_to_idx=stu_id_to_idx,
        sch_ids=sch_ids,
    )

    print("\n── Test Set Evaluation ─────────────────────────────")
    for key, val in metrics.items():
        print(f"  {key:<12}: {val:.4f}")


if __name__ == "__main__":
    main()
