"""
Pre-compute scholarship embeddings untuk production serving.

Jalankan setelah training selesai:
    python scripts/export_embeddings.py \
        --scholarship_checkpoint outputs/checkpoints/scholarship_tower_best.weights.h5

Output:
    outputs/embeddings/scholarship_emb.npy   (43, 128) float32
    outputs/embeddings/scholarship_ids.npy   list of scholarship_id strings
"""
import argparse
import os
import numpy as np
import yaml
import tensorflow as tf

from src.models.student_tower import L2Normalize
from src.utils.data_loader import load_precomputed_features


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",                  type=str, default="configs/default.yaml")
    parser.add_argument("--scholarship_checkpoint",  type=str,
                        default="outputs/checkpoints/scholarship_tower_best.keras")
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    output_dir = cfg["output"]["embedding_dir"]
    os.makedirs(output_dir, exist_ok=True)

    # ── Load features ─────────────────────────────────────────────────────────
    (_, sch_struct, _, sch_text_emb,
     _, sch_id_to_idx) = load_precomputed_features(cfg)

    sch_ids = list(sch_id_to_idx.keys())

    # ── Load scholarship tower dari format .keras ─────────────────────────────
    custom_objects    = {"L2Normalize": L2Normalize}
    scholarship_tower = tf.keras.models.load_model(
        args.scholarship_checkpoint, custom_objects=custom_objects)

    # ── Encode all scholarships ───────────────────────────────────────────────
    sch_feat_all = np.concatenate([sch_struct, sch_text_emb], axis=1)  # (43, 509)
    sch_emb      = scholarship_tower(sch_feat_all, training=False).numpy()  # (43, 128)

    # Verify L2 norms
    norms = np.linalg.norm(sch_emb, axis=1)
    print(f"Scholarship embeddings shape: {sch_emb.shape}")
    print(f"L2 norms — min={norms.min():.6f}  max={norms.max():.6f}  (should be ≈1.0)")

    # ── Save ──────────────────────────────────────────────────────────────────
    emb_path = os.path.join(output_dir, "scholarship_emb.npy")
    ids_path = os.path.join(output_dir, "scholarship_ids.npy")

    np.save(emb_path, sch_emb)
    np.save(ids_path, np.array(sch_ids, dtype=object))

    print(f"\nSaved: {emb_path}")
    print(f"Saved: {ids_path}")


if __name__ == "__main__":
    main()
