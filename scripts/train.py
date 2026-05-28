"""
Entry point untuk training Two-Tower model.

Usage:
    python scripts/train.py --config configs/default.yaml
"""
import argparse
import os
import yaml
import numpy as np
import tensorflow as tf

from src.models import build_student_tower, build_scholarship_tower
from src.trainers import Trainer
from src.trainers.trainer import BestModelCallback
from src.evaluators import Evaluator
from src.utils.data_loader import load_data, load_precomputed_features, make_dataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    tf.random.set_seed(cfg["experiment"]["seed"])
    np.random.seed(cfg["experiment"]["seed"])

    # ── Load data ─────────────────────────────────────────────────────────────
    train_df, val_df, test_df = load_data(cfg)

    (stu_struct, sch_struct, stu_text_emb, sch_text_emb,
     stu_id_to_idx, sch_id_to_idx) = load_precomputed_features(cfg)

    sch_ids = list(sch_id_to_idx.keys())
    feedback_weights = cfg["feedback_weights"]
    batch_size       = cfg["training"]["batch_size"]

    train_ds = make_dataset(train_df, stu_struct, sch_struct, stu_text_emb, sch_text_emb,
                            stu_id_to_idx, sch_id_to_idx, feedback_weights,
                            batch_size=batch_size, shuffle=True)
    val_ds   = make_dataset(val_df,   stu_struct, sch_struct, stu_text_emb, sch_text_emb,
                            stu_id_to_idx, sch_id_to_idx, feedback_weights,
                            batch_size=batch_size, shuffle=False)

    # ── Build model ───────────────────────────────────────────────────────────
    student_tower     = build_student_tower(cfg["model"]["student_tower"]["input_dim"])
    scholarship_tower = build_scholarship_tower(cfg["model"]["scholarship_tower"]["input_dim"])

    optimizer = tf.keras.optimizers.Adam(learning_rate=cfg["training"]["learning_rate"])
    trainer   = Trainer(
        student_tower=student_tower,
        scholarship_tower=scholarship_tower,
        optimizer=optimizer,
        temperature=cfg["model"]["temperature"],
        checkpoint_dir=cfg["output"]["checkpoint_dir"],
    )
    k         = cfg["evaluation"]["k_values"][0]
    evaluator = Evaluator(k=k)

    # ── Custom Callback ───────────────────────────────────────────────────────
    callback = BestModelCallback(
        student_tower=student_tower,
        scholarship_tower=scholarship_tower,
        checkpoint_dir=cfg["output"]["checkpoint_dir"],
        monitor=f"val_recall@{k}",
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    epochs = cfg["training"]["epochs"]
    print(f"\nTraining {epochs} epochs...\n")

    callback.on_train_begin()
    for epoch in range(1, epochs + 1):
        train_loss = trainer.train_epoch(train_ds)
        val_loss   = trainer.eval_epoch(val_ds)

        metrics = evaluator.compute_metrics(
            df=val_df,
            student_tower=student_tower,
            scholarship_tower=scholarship_tower,
            stu_struct=stu_struct,
            sch_struct=sch_struct,
            stu_text_emb=stu_text_emb,
            sch_text_emb=sch_text_emb,
            stu_id_to_idx=stu_id_to_idx,
            sch_ids=sch_ids,
        )

        logs = {
            "train_loss":       train_loss,
            "val_loss":         val_loss,
            f"val_recall@{k}":  metrics[f"recall@{k}"],
            f"val_ndcg@{k}":    metrics[f"ndcg@{k}"],
            "val_mrr":          metrics["mrr"],
        }
        callback.on_epoch_end(epoch - 1, logs=logs)

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f} | "
            f"Recall@{k}={metrics[f'recall@{k}']:.4f}  "
            f"NDCG@{k}={metrics[f'ndcg@{k}']:.4f}  "
            f"MRR={metrics['mrr']:.4f}"
        )

    callback.on_train_end()
    print(f"Checkpoints: {cfg['output']['checkpoint_dir']}")


if __name__ == "__main__":
    main()
