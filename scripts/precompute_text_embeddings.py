"""
Pre-compute Sentence-BERT text embeddings untuk semua students dan scholarships.

Jalankan sekali sebelum training:
    python scripts/precompute_text_embeddings.py

Output:
    data/features/text_embeddings/students.npy      (20000, 384)
    data/features/text_embeddings/scholarships.npy  (43, 384)
    outputs/embeddings/scholarship_ids.npy           list scholarship_id strings
"""
import argparse
import os
import numpy as np
import pandas as pd

from src.utils.feature_engineering import encode_text


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    return parser.parse_args()


def _needs_recompute(src_path: str, dst_path: str) -> bool:
    """Return True if the source is newer than the destination, or dst doesn't exist."""
    if not os.path.exists(dst_path):
        return True
    return os.path.getmtime(src_path) > os.path.getmtime(dst_path)


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = __import__("yaml").safe_load(f)

    data_root   = cfg["data"]["raw_path"]
    emb_dir     = cfg["data"]["text_embeddings_path"]

    os.makedirs(emb_dir, exist_ok=True)

    STU_EMB_PATH = os.path.join(emb_dir, "students.npy")
    SCH_EMB_PATH = os.path.join(emb_dir, "scholarships.npy")
    SCH_IDS_PATH = cfg["embeddings"]["scholarship_ids"]

    # Source files to check for staleness
    STU_SRC  = os.path.join(data_root, "students.csv")
    SCH_SRC  = os.path.join(data_root, "scholarships.csv")

    students_df     = pd.read_csv(os.path.join(data_root, "students.csv"))
    scholarships_df = pd.read_csv(os.path.join(data_root, "scholarships.csv"))

    print(f"Students    : {len(students_df):,}")
    print(f"Scholarships: {len(scholarships_df)}")

    # ── Student text embeddings ───────────────────────────────────────────────
    if _needs_recompute(STU_SRC, STU_EMB_PATH):
        print("\nEncoding student texts (ini butuh beberapa menit)...")
        stu_texts = (
            students_df["personal_statement"].fillna("") + " " +
            students_df["achievements_narrative"].fillna("") + " " +
            students_df["future_goals"].fillna("")
        ).tolist()
        student_text_emb = encode_text(stu_texts)
        np.save(STU_EMB_PATH, student_text_emb)
        print(f"Saved: {STU_EMB_PATH}  shape={student_text_emb.shape}")
    else:
        print(f"\n[SKIP] {STU_EMB_PATH} (source unchanged)")
        student_text_emb = np.load(STU_EMB_PATH)

    # ── Scholarship text embeddings ───────────────────────────────────────────
    if _needs_recompute(SCH_SRC, SCH_EMB_PATH):
        print("Encoding scholarship texts...")
        sch_texts = (
            scholarships_df["mission_statement"].fillna("") + " " +
            scholarships_df["target_recipient_profile"].fillna("")
        ).tolist()
        scholarship_text_emb = encode_text(sch_texts)
        np.save(SCH_EMB_PATH, scholarship_text_emb)
        print(f"Saved: {SCH_EMB_PATH}  shape={scholarship_text_emb.shape}")
    else:
        print(f"[SKIP] {SCH_EMB_PATH} (source unchanged)")
        scholarship_text_emb = np.load(SCH_EMB_PATH)

    # ── Scholarship IDs ───────────────────────────────────────────────────────
    scholarship_ids = scholarships_df["scholarship_id"].tolist()
    np.save(SCH_IDS_PATH, np.array(scholarship_ids, dtype=object))
    print(f"Saved: {SCH_IDS_PATH}  ({len(scholarship_ids)} IDs)")

    print("\nDone. Shapes:")
    print(f"  students.npy    : {student_text_emb.shape}")
    print(f"  scholarships.npy: {scholarship_text_emb.shape}")


if __name__ == "__main__":
    main()
