"""
Pre-compute Sentence-BERT text embeddings untuk semua students dan scholarships.

Jalankan sekali sebelum training:
    python scripts/precompute_text_embeddings.py

Output:
    data/features/text_embeddings/students.npy      (20000, 384)
    data/features/text_embeddings/scholarships.npy  (43, 384)
    outputs/embeddings/scholarship_ids.npy           list scholarship_id strings
"""
import os
import numpy as np
import pandas as pd

from src.utils.feature_engineering import encode_text

DATA_DIR    = "data/raw"
EMB_DIR     = "data/features/text_embeddings"
OUTPUT_DIR  = "outputs/embeddings"

os.makedirs(EMB_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

STU_EMB_PATH = os.path.join(EMB_DIR, "students.npy")
SCH_EMB_PATH = os.path.join(EMB_DIR, "scholarships.npy")
SCH_IDS_PATH = os.path.join(OUTPUT_DIR, "scholarship_ids.npy")


def main():
    students_df     = pd.read_csv(os.path.join(DATA_DIR, "students.csv"))
    scholarships_df = pd.read_csv(os.path.join(DATA_DIR, "scholarships.csv"))

    print(f"Students    : {len(students_df):,}")
    print(f"Scholarships: {len(scholarships_df)}")

    # ── Student text embeddings ───────────────────────────────────────────────
    if os.path.exists(STU_EMB_PATH):
        print(f"\n[SKIP] {STU_EMB_PATH} sudah ada.")
        student_text_emb = np.load(STU_EMB_PATH)
    else:
        print("\nEncoding student texts (ini butuh beberapa menit)...")
        stu_texts = (
            students_df["personal_statement"].fillna("") + " " +
            students_df["achievements_narrative"].fillna("") + " " +
            students_df["future_goals"].fillna("")
        ).tolist()
        student_text_emb = encode_text(stu_texts)
        np.save(STU_EMB_PATH, student_text_emb)
        print(f"Saved: {STU_EMB_PATH}  shape={student_text_emb.shape}")

    # ── Scholarship text embeddings ───────────────────────────────────────────
    if os.path.exists(SCH_EMB_PATH):
        print(f"[SKIP] {SCH_EMB_PATH} sudah ada.")
        scholarship_text_emb = np.load(SCH_EMB_PATH)
    else:
        print("Encoding scholarship texts...")
        sch_texts = (
            scholarships_df["mission_statement"].fillna("") + " " +
            scholarships_df["target_recipient_profile"].fillna("")
        ).tolist()
        scholarship_text_emb = encode_text(sch_texts)
        np.save(SCH_EMB_PATH, scholarship_text_emb)
        print(f"Saved: {SCH_EMB_PATH}  shape={scholarship_text_emb.shape}")

    # ── Scholarship IDs ───────────────────────────────────────────────────────
    scholarship_ids = scholarships_df["scholarship_id"].tolist()
    np.save(SCH_IDS_PATH, np.array(scholarship_ids, dtype=object))
    print(f"Saved: {SCH_IDS_PATH}  ({len(scholarship_ids)} IDs)")

    print("\nDone. Shapes:")
    print(f"  students.npy    : {student_text_emb.shape}")
    print(f"  scholarships.npy: {scholarship_text_emb.shape}")


if __name__ == "__main__":
    main()
