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


def _build_log_dir(cfg: dict) -> str:
    """Build TensorBoard log directory path.

    Returns a clean path without trailing underscores.
    """
    tb_cfg = cfg.get("tensorboard", {})
    suffix = tb_cfg.get("suffix", "")
    if suffix:
        return os.path.join(
            cfg["output"]["log_dir"],
            f"tb_{cfg['experiment']['name']}_{suffix}",
        )
    return os.path.join(cfg["output"]["log_dir"], f"tb_{cfg['experiment']['name']}")


def _log_scalar(summary_writer, name, value, step):
    """Write a scalar to TensorBoard summary."""
    with summary_writer.as_default():
        tf.summary.scalar(name, value, step=step)


def _log_embedding_histograms(student_tower, scholarship_tower, summary_writer, step):
    """Log embedding vector distributions from both towers."""
    dummy = tf.constant([[0.0] * 1], dtype=tf.float32)
    try:
        stu_emb = student_tower(dummy, training=False)
        sch_emb = scholarship_tower(dummy, training=False)
        with summary_writer.as_default():
            tf.summary.histogram("embeddings/student_tower", stu_emb, step=step)
            tf.summary.histogram("embeddings/scholarship_tower", sch_emb, step=step)
    except Exception:
        pass  # non-fatal; training continues regardless


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

    # ── TensorBoard setup ───────────────────────────────────────────────────────
    tb_cfg = cfg.get("tensorboard", {})
    tb_enabled = tb_cfg.get("enabled", False)
    summary_writer = None
    if tb_enabled:
        log_dir = _build_log_dir(cfg)
        summary_writer = tf.summary.create_file_writer(log_dir)

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

        # ── TensorBoard logging ───────────────────────────────────────────────
        if tb_enabled and summary_writer is not None:
            epoch_step = epoch  # use epoch number as the step
            _log_scalar(summary_writer, "loss/train_loss", train_loss, epoch_step)
            _log_scalar(summary_writer, "loss/val_loss", val_loss, epoch_step)
            _log_scalar(summary_writer, f"recall/val_recall@{k}", metrics[f"recall@{k}"], epoch_step)
            _log_scalar(summary_writer, f"ndcg/val_ndcg@{k}", metrics[f"ndcg@{k}"], epoch_step)
            _log_scalar(summary_writer, "mrr/val_mrr", metrics["mrr"], epoch_step)

            # Log embedding histograms periodically
            histogram_freq = tb_cfg.get("histogram_freq", 0)
            if histogram_freq > 0 and epoch % histogram_freq == 0:
                _log_embedding_histograms(student_tower, scholarship_tower, summary_writer, epoch_step)

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f} | "
            f"Recall@{k}={metrics[f'recall@{k}']:.4f}  "
            f"NDCG@{k}={metrics[f'ndcg@{k}']:.4f}  "
            f"MRR={metrics['mrr']:.4f}"
        )

    callback.on_train_end()

    # ── Close TensorBoard writer ──────────────────────────────────────────────
    if summary_writer is not None:
        summary_writer.close()
    print(f"Checkpoints: {cfg['output']['checkpoint_dir']}")


if __name__ == "__main__":
    main()
