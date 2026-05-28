"""
Data loading dan tf.data.Dataset builder untuk Two-Tower training.
"""
import json
import numpy as np
import pandas as pd
import tensorflow as tf

from src.utils.feature_engineering import encode_student, encode_scholarship

AUTOTUNE = tf.data.AUTOTUNE
SEED = 42


# ── Load raw data ─────────────────────────────────────────────────────────────

def load_data(cfg: dict):
    """Baca feedback.csv, sort by timestamp, split 70/15/15.

    Returns:
        train_df, val_df, test_df: DataFrame dengan kolom
            [student_id, scholarship_id, feedback_type, timestamp]
    """
    raw_path = cfg["data"]["raw_path"]
    feedback_df = pd.read_csv(f"{raw_path}/feedback.csv")
    feedback_df = feedback_df.sort_values("timestamp").reset_index(drop=True)

    n = len(feedback_df)
    n_train = int(cfg["data"]["train_split"] * n)
    n_val   = int(cfg["data"]["val_split"] * n)

    train_df = feedback_df.iloc[:n_train]
    val_df   = feedback_df.iloc[n_train : n_train + n_val]
    test_df  = feedback_df.iloc[n_train + n_val :]

    print(f"Data split — train:{len(train_df):,}  val:{len(val_df):,}  test:{len(test_df):,}")
    return train_df, val_df, test_df


# ── Pre-compute structured features ──────────────────────────────────────────

def load_precomputed_features(cfg: dict):
    """Encode semua student + scholarship structured features dan load text embeddings.

    Returns:
        stu_struct  : np.ndarray (N_stu, 122)
        sch_struct  : np.ndarray (N_sch, 125)
        stu_text_emb: np.ndarray (N_stu, 384)
        sch_text_emb: np.ndarray (N_sch, 384)
        stu_id_to_idx: dict {student_id: row_index}
        sch_id_to_idx: dict {scholarship_id: row_index}
    """
    raw_path = cfg["data"]["raw_path"]
    emb_path = cfg["data"]["text_embeddings_path"]

    students_df     = pd.read_csv(f"{raw_path}/students.csv")
    scholarships_df = pd.read_csv(f"{raw_path}/scholarships.csv")

    # Parse JSON columns
    for col in ["language_proficiency", "olympiad_subjects", "target_countries"]:
        students_df[col] = students_df[col].apply(
            lambda x: json.loads(x) if isinstance(x, str) else x
        )
    for col in ["eligible_nationalities", "eligible_degree_levels",
                "eligible_high_school_tracks", "eligible_fields",
                "language_requirements", "selection_criteria"]:
        scholarships_df[col] = scholarships_df[col].apply(
            lambda x: json.loads(x) if isinstance(x, str) else x
        )

    # Encode structured features
    print("Encoding structured features...")
    stu_struct = np.array(
        [encode_student(r) for _, r in students_df.iterrows()], dtype=np.float32
    )
    sch_struct = np.array(
        [encode_scholarship(r) for _, r in scholarships_df.iterrows()], dtype=np.float32
    )

    # Load pre-computed text embeddings
    stu_text_emb = np.load(f"{emb_path}/students.npy").astype(np.float32)
    sch_text_emb = np.load(f"{emb_path}/scholarships.npy").astype(np.float32)

    stu_id_to_idx = {sid: i for i, sid in enumerate(students_df["student_id"])}
    sch_id_to_idx = {sid: i for i, sid in enumerate(scholarships_df["scholarship_id"])}

    print(f"  stu_struct : {stu_struct.shape}  stu_text_emb : {stu_text_emb.shape}")
    print(f"  sch_struct : {sch_struct.shape}  sch_text_emb : {sch_text_emb.shape}")

    return stu_struct, sch_struct, stu_text_emb, sch_text_emb, stu_id_to_idx, sch_id_to_idx


# ── Build tf.data.Dataset ─────────────────────────────────────────────────────

def make_dataset(
    df: pd.DataFrame,
    stu_struct: np.ndarray,
    sch_struct: np.ndarray,
    stu_text_emb: np.ndarray,
    sch_text_emb: np.ndarray,
    stu_id_to_idx: dict,
    sch_id_to_idx: dict,
    feedback_weights: dict,
    batch_size: int = 256,
    shuffle: bool = False,
) -> tf.data.Dataset:
    """Build tf.data.Dataset yielding (stu_feat, sch_feat, weights) triples.

    stu_feat shape: (batch, 506)  — concat(struct, text_emb)
    sch_feat shape: (batch, 509)
    weights  shape: (batch,)      — feedback type weights
    """
    stu_indices = np.array([stu_id_to_idx[sid] for sid in df["student_id"]], dtype=np.int32)
    sch_indices = np.array([sch_id_to_idx[sid] for sid in df["scholarship_id"]], dtype=np.int32)
    weights     = np.array([feedback_weights[ft] for ft in df["feedback_type"]], dtype=np.float32)

    stu_feat = np.concatenate([stu_struct[stu_indices], stu_text_emb[stu_indices]], axis=1)
    sch_feat = np.concatenate([sch_struct[sch_indices], sch_text_emb[sch_indices]], axis=1)

    ds = tf.data.Dataset.from_tensor_slices((stu_feat, sch_feat, weights))
    if shuffle:
        ds = ds.shuffle(buffer_size=10_000, seed=SEED)
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(AUTOTUNE)
    return ds
