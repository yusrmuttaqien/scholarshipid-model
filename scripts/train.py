"""
Entry point for training Two-Tower model.

Usage:
    python scripts/train.py --config configs/default.yaml
"""
import argparse
import os
import yaml
import numpy as np
import tensorflow as tf

from src.models import build_student_tower, build_scholarship_tower
from src.trainers.training_loop import run_training
from src.utils.data_loader import load_data, load_precomputed_features


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
    train_df, val_df, _test_df = load_data(cfg)

    (stu_struct, sch_struct, stu_text_emb, sch_text_emb,
     stu_id_to_idx, sch_id_to_idx) = load_precomputed_features(cfg)

    sch_ids = list(sch_id_to_idx.keys())
    feedback_weights = cfg["feedback_weights"]

    # ── Build model ───────────────────────────────────────────────────────────
    student_tower     = build_student_tower(cfg["model"]["student_tower"]["input_dim"])
    scholarship_tower = build_scholarship_tower(cfg["model"]["scholarship_tower"]["input_dim"])

    # ── Build full feature matrices for ALL students and scholarships ─────────
    stu_feat_all = np.concatenate([stu_struct, stu_text_emb], axis=1)  # (N_stu, 506)
    sch_feat_all = np.concatenate([sch_struct, sch_text_emb], axis=1)  # (N_sch, 509)

    def _build_sample_features(df):
        """Build feature arrays from a DataFrame of feedback rows."""
        stu_id_map = {sid: i for i, sid in enumerate(stu_id_to_idx.keys())}
        sch_id_map = {sid: i for i, sid in enumerate(sch_id_to_idx.keys())}
        stu_indices = np.array(
            [stu_id_map[sid] for sid in df["student_id"]], dtype=np.int32
        )
        sch_indices = np.array(
            [sch_id_map[sid] for sid in df["scholarship_id"]], dtype=np.int32
        )
        weights = np.array(
            [feedback_weights[ft] for ft in df["feedback_type"]], dtype=np.float32
        )
        train_stu_feat = stu_feat_all[stu_indices]
        train_sch_feat = sch_feat_all[sch_indices]
        return train_stu_feat, train_sch_feat, weights

    train_stu_feat, train_sch_feat, train_weights = _build_sample_features(train_df)
    val_stu_feat, val_sch_feat, val_weights = _build_sample_features(val_df)

    optimizer = tf.keras.optimizers.Adam(learning_rate=cfg["training"]["learning_rate"])
    epochs = cfg["training"]["epochs"]
    k = cfg["evaluation"]["k_values"][0]

    print(f"\nTraining {epochs} epochs...\n")

    # ── Unified training loop with shared run_training() ─────────────────────
    metrics = run_training(
        student_tower=student_tower,
        scholarship_tower=scholarship_tower,
        train_stu_feat=train_stu_feat,
        train_sch_feat=train_sch_feat,
        train_weights=train_weights,
        val_stu_feat=val_stu_feat,
        val_sch_feat=val_sch_feat,
        val_weights=val_weights,
        optimizer=optimizer,
        temperature=cfg["model"]["temperature"],
        epochs=epochs,
        checkpoint_dir=cfg["output"]["checkpoint_dir"],
        log_dir=_build_log_dir(cfg),
        tb_enabled=True,
        k=k,
        # ── Validation metrics (full Recall/NDCG/MRR) ───────────────────
        val_df=val_df,
        stu_struct=stu_struct,
        sch_struct=sch_struct,
        stu_text_emb=stu_text_emb,
        sch_text_emb=sch_text_emb,
        stu_id_to_idx=stu_id_to_idx,
        sch_ids=sch_ids,
    )

    print(f"Checkpoints: {cfg['output']['checkpoint_dir']}")


if __name__ == "__main__":
    main()