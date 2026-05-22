"""Two-Tower Recommendation System — Training Script (MVP)

Simple training pipeline using TensorFlow/Keras.
Loads CSV datasets, preprocesses features in pandas,
and trains a two-tower neural network for relevance scoring.

Usage:
    python train.py
    python train.py --epochs 50 --batch-size 256
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ============================================================
# Constants
# ============================================================

DATASET_DIR = "datasets_two_tower"
OUTPUT_DIR = "models_two_tower"
EPOCHS = 50
BATCH_SIZE = 256
EMBEDDING_DIM = 128
LEARNING_RATE = 1e-3
SEED = 42

# Feature columns (skip text fields for MVP)
STUDENT_CATEGORICAL = [
    "nationality", "high_school_track", "school_tier",
    "family_income_category", "intended_career_track", "olympiad_level",
]
STUDENT_NUMERICAL = [
    "age", "overall_report_card_average", "math_score",
    "english_score", "major_subject_average",
    "leadership_experience_count", "volunteer_experience_count",
    "competition_wins_count",
]
STUDENT_BOOLEAN = [
    "willing_to_return_home", "from_underrepresented_region",
    "needs_full_funding", "can_self_fund_living",
]

SCHOLARSHIP_CATEGORICAL = [
    "host_region", "preferred_school_tier",
    "career_track_preference", "max_family_income_category",
]
SCHOLARSHIP_NUMERICAL = [
    "min_age", "max_age", "min_report_card_average",
    "min_major_subject_average", "funding_monthly_stipend",
    "funding_coverage_count",
]
SCHOLARSHIP_BOOLEAN = [
    "requires_financial_need", "requires_return_home_country",
    "funding_covers_tuition", "funding_covers_living",
    "funding_covers_airfare", "funding_covers_insurance",
    "funding_is_full_funding",
]

# List vector dimensions
LIST_COUNTRY_DIM = 27
LIST_TRACK_DIM = 5
LIST_FIELD_DIM = 14
LIST_VECTOR_DIM = LIST_COUNTRY_DIM + LIST_TRACK_DIM + LIST_FIELD_DIM  # 46


# ============================================================
# Custom Layer: Cosine Similarity (kept for compatibility)
# ============================================================

try:
    from tensorflow.keras.saving import register_keras_serializable as _register
except ImportError:
    def _register(package=None):
        def decorator(cls):
            return cls
        return decorator

@_register(package="ScholarshipID")
class CosineSimilarity(layers.Layer):
    """Cosine similarity + sigmoid → relevance score in [0, 1]."""

    def call(self, inputs, training=None):
        student_emb, scholarship_emb = inputs
        student_norm = tf.nn.l2_normalize(student_emb, axis=-1)
        scholarship_norm = tf.nn.l2_normalize(scholarship_emb, axis=-1)
        cosine = tf.reduce_sum(student_norm * scholarship_norm, axis=-1, keepdims=True)
        return tf.sigmoid(cosine)

    def get_config(self):
        return super().get_config()


# ============================================================
# Tower Builders — use packed inputs for efficiency
# ============================================================

def build_student_tower(input_dim: int, embedding_dim: int = 128) -> keras.Model:
    """Student tower with deeper architecture."""
    inp = layers.Input(shape=(input_dim,), name="student_inputs")
    x = layers.Dense(256, activation='relu')(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    embedding = layers.Dense(embedding_dim, activation='relu', name="student_embedding")(x)
    return keras.Model(inputs=inp, outputs=embedding, name="student_tower")


def build_scholarship_tower(input_dim: int, embedding_dim: int = 128) -> keras.Model:
    """Scholarship tower with deeper architecture."""
    inp = layers.Input(shape=(input_dim,), name="scholarship_inputs")
    x = layers.Dense(256, activation='relu')(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    embedding = layers.Dense(embedding_dim, activation='relu', name="scholarship_embedding")(x)
    return keras.Model(inputs=inp, outputs=embedding, name="scholarship_tower")


def build_two_tower_model(
    student_input_dim: int,
    scholarship_input_dim: int,
    embedding_dim: int = 128,
) -> keras.Model:
    """Assemble two-tower model with cosine similarity output."""
    student_tower = build_student_tower(student_input_dim, embedding_dim)
    scholarship_tower = build_scholarship_tower(scholarship_input_dim, embedding_dim)

    student_inputs = student_tower.inputs
    sch_inputs = scholarship_tower.inputs

    student_emb = student_tower(student_inputs)
    sch_emb = scholarship_tower(sch_inputs)

    similarity = CosineSimilarity(name="cosine_similarity")([student_emb, sch_emb])
    return keras.Model(
        inputs=[student_inputs[0], sch_inputs[0]],
        outputs=similarity,
        name="two_tower",
    )


# ============================================================
# Feature Preprocessing (pandas-based)
# ============================================================

def _encode_categorical(df: pd.DataFrame, col: str) -> tuple[np.ndarray, dict]:
    """Label-encode a categorical column. Returns array and reverse mapping."""
    col_vals = df[col].astype(str).fillna("unknown")
    unique_vals = sorted(col_vals.unique())
    mapping = {v: i + 1 for i, v in enumerate(unique_vals)}
    arr = np.zeros(len(df), dtype=np.int32)
    for val, idx in mapping.items():
        mask = col_vals == val
        arr[mask] = idx
    return arr, mapping


def _encode_list_field(json_str, all_values):
    """Encode a list field (JSON string) as binary vector."""
    vec = np.zeros(len(all_values), dtype=np.float32)
    if not json_str or json_str == "[]":
        return vec
    try:
        items = json.loads(json_str)
        if isinstance(items, str):
            items = [items]
        for item in items:
            if isinstance(item, str) and item in all_values:
                vec[all_values.index(item)] = 1.0
    except (json.JSONDecodeError, TypeError):
        pass
    return vec


def _parse_language_proficiency(json_str):
    """Parse language proficiency JSON into 12-dim vector."""
    vec = np.zeros(12, dtype=np.float32)
    try:
        records = json.loads(json_str) if isinstance(json_str, str) else json_str
    except (json.JSONDecodeError, TypeError):
        return vec
    if not isinstance(records, list):
        records = [records]
    language_tests = ["toefl", "ielts", "topik", "jlpt", "delf", "hsk"]
    for record in records:
        if not isinstance(record, dict):
            continue
        test_type = record.get("test_type", "").lower()
        if test_type in language_tests:
            idx = language_tests.index(test_type)
            score = float(record.get("score", 0.0))
            vec[idx * 2] = max(vec[idx * 2], score)
            vec[idx * 2 + 1] = 1.0
    return vec


def _make_list_vectors(scholarships_df, col_name, all_values, offset, dim):
    """Build list vector for a column, placing at offset with given dim."""
    vecs = np.zeros((len(scholarships_df), dim), dtype=np.float32)
    for i, s in enumerate(scholarships_df[col_name].tolist()):
        vec = _encode_list_field(s, all_values)
        vecs[i] = vec[:dim]
    return vecs


def prepare_features(students_df, scholarships_df):
    """Preprocess both DataFrames into packed numpy arrays.

    Returns:
        student_X: (n_students, n_student_features)
        sch_X: (n_scholarships, n_scholarship_features)
        student_mappings: dict of mapping info
        sch_mappings: dict of mapping info
    """
    # --- Students ---
    stu_parts = []

    for col in STUDENT_CATEGORICAL:
        arr, _ = _encode_categorical(students_df, col)
        stu_parts.append(arr[:, None])  # (N, 1)

    for col in STUDENT_NUMERICAL:
        stu_parts.append(students_df[col].values.astype(np.float32)[:, None])

    for col in STUDENT_BOOLEAN:
        stu_parts.append(students_df[col].astype(float).fillna(0).values[:, None])

    # Language vector (already (N, 12))
    lang_vecs = np.stack(
        [_parse_language_proficiency(s) for s in students_df["language_proficiency"].tolist()]
    )
    stu_parts.append(lang_vecs)

    student_X = np.hstack(stu_parts).astype(np.float32)

    # --- Scholarships ---
    sch_parts = []

    for col in SCHOLARSHIP_CATEGORICAL:
        arr, _ = _encode_categorical(scholarships_df, col)
        sch_parts.append(arr[:, None])  # (N, 1)

    for col in SCHOLARSHIP_NUMERICAL:
        sch_parts.append(scholarships_df[col].values.astype(np.float32)[:, None])

    for col in SCHOLARSHIP_BOOLEAN:
        sch_parts.append(scholarships_df[col].astype(float).fillna(0).values[:, None])

    # List vector: 27 countries + 5 tracks + 14 fields = 46
    all_countries = [
        "china", "india", "indonesia", "japan", "malaysia",
        "philippines", "singapore", "south_korea", "thailand", "vietnam",
        "france", "germany", "netherlands", "sweden", "uk",
        "switzerland", "canada", "usa", "argentina", "brazil",
        "chile", "egypt", "kenya", "morocco", "nigeria",
        "south_africa", "australia", "new_zealand",
    ]
    all_tracks = ["science", "social_studies", "languages", "religion", "vocational"]
    all_fields = [
        "computer_science", "engineering", "medicine", "business",
        "economics", "law", "education", "arts_humanities",
        "social_sciences", "agriculture", "mathematics", "physics",
        "chemistry", "biology",
    ]
    all_list_values = all_countries + all_tracks + all_fields

    list_vec = np.zeros((len(scholarships_df), LIST_VECTOR_DIM), dtype=np.float32)

    # eligible_nationalities → offset 0 (countries)
    nationalities_vecs = _make_list_vectors(scholarships_df, "eligible_nationalities", all_list_values, 0, LIST_COUNTRY_DIM)
    list_vec[:, :LIST_COUNTRY_DIM] = nationalities_vecs

    # eligible_high_school_tracks → offset 27 (tracks)
    tracks_vecs = _make_list_vectors(scholarships_df, "eligible_high_school_tracks", all_list_values, LIST_COUNTRY_DIM, LIST_TRACK_DIM)
    list_vec[:, LIST_COUNTRY_DIM:LIST_COUNTRY_DIM + LIST_TRACK_DIM] += tracks_vecs

    # eligible_fields → offset 32 (fields)
    fields_vecs = _make_list_vectors(scholarships_df, "eligible_fields", all_list_values, LIST_COUNTRY_DIM + LIST_TRACK_DIM, LIST_FIELD_DIM)
    list_vec[:, LIST_COUNTRY_DIM + LIST_TRACK_DIM:] += fields_vecs

    sch_parts.append(list_vec)
    sch_X = np.hstack(sch_parts).astype(np.float32)

    student_mappings = {col: _encode_categorical(students_df, col)[1] for col in STUDENT_CATEGORICAL}
    sch_mappings = {col: _encode_categorical(scholarships_df, col)[1] for col in SCHOLARSHIP_CATEGORICAL}

    return student_X, sch_X, student_mappings, sch_mappings


# ============================================================
# Training Loop with Monitoring
# ============================================================

def train(args):
    """Train one-tower merged model on recommendation pairs."""
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    # --- Load data ---
    print(f"Loading data from {DATASET_DIR}/...")
    students = pd.read_csv(f"{DATASET_DIR}/students.csv")
    scholarships = pd.read_csv(f"{DATASET_DIR}/scholarships.csv")
    pairs = pd.read_csv(f"{DATASET_DIR}/pairs.csv")

    print(f"  Students: {len(students)}, Scholarships: {len(scholarships)}, Pairs: {len(pairs)}")

    # --- Preprocess features ---
    print("Preprocessing features...")
    student_X, sch_X, student_mappings, sch_mappings = prepare_features(
        students, scholarships
    )

    print(f"  Student features: {student_X.shape}")
    print(f"  Scholarship features: {sch_X.shape}")

    # Build pairs feature arrays
    stu_id_to_idx = {sid: i for i, sid in enumerate(students["student_id"])}
    sch_id_to_idx = {sid: i for i, sid in enumerate(scholarships["scholarship_id"])}

    # --- Time-based split (70/15/15) ---
    pairs_sorted = pairs.sort_values("timestamp")
    n = len(pairs_sorted)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)

    train_pairs = pairs_sorted.iloc[:train_end]
    val_pairs = pairs_sorted.iloc[train_end:val_end]
    test_pairs = pairs_sorted.iloc[val_end:]

    # Map IDs to indices
    train_stu_idx = np.array([stu_id_to_idx[sid] for sid in train_pairs["student_id"]], dtype=np.int32)
    train_sch_idx = np.array([sch_id_to_idx[sid] for sid in train_pairs["scholarship_id"]], dtype=np.int32)
    train_labels = train_pairs["relevance_score"].values.astype(np.float32)

    val_stu_idx = np.array([stu_id_to_idx[sid] for sid in val_pairs["student_id"]], dtype=np.int32)
    val_sch_idx = np.array([sch_id_to_idx[sid] for sid in val_pairs["scholarship_id"]], dtype=np.int32)
    val_labels = val_pairs["relevance_score"].values.astype(np.float32)

    test_stu_idx = np.array([stu_id_to_idx[sid] for sid in test_pairs["student_id"]], dtype=np.int32)
    test_sch_idx = np.array([sch_id_to_idx[sid] for sid in test_pairs["scholarship_id"]], dtype=np.int32)
    test_labels = test_pairs["relevance_score"].values.astype(np.float32)

    # --- Build datasets (two-tower: separate student/scholarship features) ---
    print("Building datasets...")
    train_dataset = tf.data.Dataset.from_tensor_slices((
        (student_X[train_stu_idx], sch_X[train_sch_idx]), train_labels
    )).batch(args.batch_size).prefetch(tf.data.AUTOTUNE)
    val_dataset = tf.data.Dataset.from_tensor_slices((
        (student_X[val_stu_idx], sch_X[val_sch_idx]), val_labels
    )).batch(args.batch_size).prefetch(tf.data.AUTOTUNE)
    test_dataset = tf.data.Dataset.from_tensor_slices((
        (student_X[test_stu_idx], sch_X[test_sch_idx]), test_labels
    )).batch(args.batch_size)

    # --- Build model ---
    student_input_dim = student_X.shape[1]
    scholarship_input_dim = sch_X.shape[1]
    print(f"\nBuilding two-tower model (student_input={student_input_dim}, sch_input={scholarship_input_dim})...")
    model = build_two_tower_model(student_input_dim, scholarship_input_dim, EMBEDDING_DIM)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="mse",
        metrics=["mae"],
    )
    model.summary()

    # --- Callbacks ---
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_mae", patience=5, restore_best_weights=True, mode="min"
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_mae", factor=0.5, patience=3, min_lr=1e-6
        ),
        keras.callbacks.TensorBoard(
            log_dir=os.path.join(OUTPUT_DIR, "logs"), histogram_freq=1
        ),
    ]

    # --- Train ---
    print(f"\nTraining for {args.epochs} epochs...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    checkpoint_path = os.path.join(OUTPUT_DIR, "best_model.keras")
    callbacks.append(keras.callbacks.ModelCheckpoint(checkpoint_path, monitor="val_mae", save_best_only=True))

    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=1,
    )

    # --- Evaluate on test set ---
    print("\n" + "=" * 60)
    print("EVALUATION ON TEST SET")
    print("=" * 60)
    test_results = model.evaluate(test_dataset, verbose=1)
    print(f"\nTest Loss: {test_results[0]:.4f}")
    print(f"Test MAE:  {test_results[1]:.4f}")

    # --- Class-level metrics ---
    test_preds = []
    test_true = []
    for (stu_inputs, sch_inputs), y in test_dataset:
        preds = model.predict((stu_inputs, sch_inputs), verbose=0)
        test_preds.append(preds.flatten())
        test_true.append(y.numpy().flatten())

    test_preds = np.concatenate(test_preds)
    test_true = np.concatenate(test_true)

    match_mask = test_true >= 0.7
    inbetween_mask = (test_true >= 0.3) & (test_true < 0.7)
    notmatch_mask = test_true < 0.3

    def compute_metrics(true, pred):
        rmse = np.sqrt(np.mean((true - pred) ** 2))
        mae = np.mean(np.abs(true - pred))
        return rmse, mae

    for name, mask in [("Match", match_mask), ("In-Between", inbetween_mask), ("Not Match", notmatch_mask)]:
        if mask.sum() > 0:
            rmse, mae = compute_metrics(test_true[mask], test_preds[mask])
            print(f"  {name:12s} (n={mask.sum():6d}): RMSE={rmse:.4f}, MAE={mae:.4f}")

    # --- Binary accuracy ---
    binary_true = (test_true >= 0.5).astype(int)
    binary_pred = (test_preds >= 0.5).astype(int)
    accuracy = np.mean(binary_true == binary_pred)
    print(f"\nBinary Accuracy (threshold=0.5): {accuracy * 100:.2f}%")

    # --- Save model ---
    final_path = os.path.join(OUTPUT_DIR, "two_tower_model.keras")
    model.save(final_path)
    print(f"\nModel saved to {final_path}")
    print(f"Checkpoint saved to {checkpoint_path}")
    print(f"TensorBoard logs in {OUTPUT_DIR}/logs/")

    # --- Save mappings for inference ---
    import pickle
    mappings = {"student": student_mappings, "scholarship": sch_mappings}
    with open(os.path.join(OUTPUT_DIR, "mappings.pkl"), "wb") as f:
        pickle.dump(mappings, f)
    print(f"Mappings saved to {OUTPUT_DIR}/mappings.pkl")

    return model, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train two-tower recommendation model")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()
    train(args)
