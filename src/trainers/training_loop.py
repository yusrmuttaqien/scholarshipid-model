"""Shared training loop with validation, TensorBoard logging, and BestModelCallback.

Reusable by both `scripts/train.py` and `/retrain` endpoint in InferenceEngine.
"""
import os
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import tensorflow as tf

from src.trainers.trainer import BestModelCallback, sampled_softmax_loss_weighted


def run_training(
    student_tower: tf.keras.Model,
    scholarship_tower: tf.keras.Model,
    train_stu_feat: np.ndarray,
    train_sch_feat: np.ndarray,
    train_weights: np.ndarray,
    val_stu_feat: np.ndarray,
    val_sch_feat: np.ndarray,
    val_weights: np.ndarray,
    optimizer: tf.keras.optimizers.Optimizer,
    temperature: float,
    epochs: int,
    checkpoint_dir: str,
    log_dir: str,
    tb_enabled: bool = False,
    k: int = 5,
    # ── Validation metrics (optional, for full Recall/NDCG/MRR) ───────
    val_df: Optional[object] = None,
    stu_struct: Optional[np.ndarray] = None,
    sch_struct: Optional[np.ndarray] = None,
    stu_text_emb: Optional[np.ndarray] = None,
    sch_text_emb: Optional[np.ndarray] = None,
    stu_id_to_idx: Optional[dict] = None,
    sch_ids: Optional[list] = None,
) -> dict:
    """Run training loop with validation split, TensorBoard logging, and best-model saving.

    Args:
        student_tower / scholarship_tower: Tower models to finetune.
        train_*_feat / val_*_feat: Feature matrices for train/val splits.
        train_weights / val_weights: Per-sample feedback weights.
        optimizer: Keras optimizer.
        temperature: Contrastive temperature for sampled softmax loss.
        epochs: Number of training epochs.
        checkpoint_dir: Directory to save best model weights.
        log_dir: TensorBoard log directory.
        tb_enabled: Whether to write TensorBoard summaries.
        k: Recall@K value for validation metrics.
        val_df: (optional) Validation DataFrame for Evaluator.compute_metrics().
                If provided, full Recall/NDCG/MRR metrics are computed each epoch.
        stu_struct / sch_struct: Structured feature arrays.
        stu_text_emb / sch_text_emb: Text embedding arrays.
        stu_id_to_idx / sch_ids: ID mappings for retrieval evaluation.

    Returns:
        dict with keys: train_loss, val_loss, and optionally Recall/NDCG/MRR per epoch.
    """
    # ── Setup callback ───────────────────────────────────────────────────────
    os.makedirs(checkpoint_dir, exist_ok=True)
    cb = BestModelCallback(
        student_tower=student_tower,
        scholarship_tower=scholarship_tower,
        checkpoint_dir=checkpoint_dir,
        monitor=f"recall@{k}",  # matches evaluator key (no "val_" prefix)
    )

    # ── Setup TensorBoard ────────────────────────────────────────────────────
    summary_writer = None
    if tb_enabled:
        log_dir = os.path.join(log_dir, f"retrain_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}")
        summary_writer = tf.summary.create_file_writer(log_dir)

    # ── Import evaluator lazily ────────────────────────────────────────────
    from src.evaluators import Evaluator
    evaluator = Evaluator(k=k)

    # ── Training loop ────────────────────────────────────────────────────────
    cb.on_train_begin()
    metrics_history = {}
    for epoch in range(1, epochs + 1):
        # Train step
        with tf.GradientTape() as tape:
            stu_emb = student_tower(train_stu_feat, training=True)
            sch_emb = scholarship_tower(train_sch_feat, training=True)
            train_loss = sampled_softmax_loss_weighted(stu_emb, sch_emb, train_weights, temperature)

        all_vars = (student_tower.trainable_variables +
                    scholarship_tower.trainable_variables)
        grads = tape.gradient(train_loss, all_vars)
        optimizer.apply_gradients(zip(grads, all_vars))

        # Validation step (no gradient computation needed, but keep tape for consistency)
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            val_stu_emb = student_tower(val_stu_feat, training=False)
            val_sch_emb = scholarship_tower(val_sch_feat, training=False)
            val_loss = sampled_softmax_loss_weighted(
                val_stu_emb, val_sch_emb, val_weights, temperature
            )

        # Build logs dict — include retrieval metrics if evaluator data provided
        logs = {
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
        }
        if val_df is not None and stu_struct is not None:
            eval_metrics = evaluator.compute_metrics(
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
            logs.update(eval_metrics)

        cb.on_epoch_end(epoch - 1, logs=logs)

        # TensorBoard logging
        if tb_enabled and summary_writer is not None:
            with summary_writer.as_default():
                tf.summary.scalar("loss/train_loss", train_loss, step=epoch)
                tf.summary.scalar("loss/val_loss", val_loss, step=epoch)
                for mk, mv in logs.items():
                    if mk not in ("train_loss", "val_loss"):
                        tf.summary.scalar(f"{mk}", mv, step=epoch)

        # Print progress — include retrieval metrics when available
        if val_df is not None:
            print(
                f"    Epoch {epoch}/{epochs} | "
                f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f} | "
                f"Recall@{k}={logs.get(f'recall@{k}', 0):.4f}  "
                f"NDCG@{k}={logs.get(f'ndcg@{k}', 0):.4f}  "
                f"MRR={logs.get('mrr', 0):.4f}"
            )
        else:
            print(
                f"    Epoch {epoch}/{epochs} — train_loss={train_loss:.4f}  val_loss={val_loss:.4f}"
            )

        metrics_history[epoch] = dict(logs)

    cb.on_train_end()

    # ── Close TensorBoard writer ─────────────────────────────────────────────
    if summary_writer is not None:
        summary_writer.close()

    return metrics_history