"""Two-Tower Recommendation System — Data Pipeline

Loads CSV datasets and produces tf.data.Dataset objects for efficient training.
Handles feature engineering: StringLookup, normalization, language vector encoding,
and list field binary encoding.
"""

import json
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf


# ============================================================
# Constants (must match MODEL_BUILDING_GUIDE.md)
# ============================================================

LANGUAGE_TESTS = ["toefl", "ielts", "topik", "jlpt", "delf", "hsk"]

ALL_COUNTRIES = [
    "china", "india", "indonesia", "japan", "malaysia",
    "philippines", "singapore", "south_korea", "thailand", "vietnam",
    "france", "germany", "netherlands", "sweden", "uk",
    "switzerland", "canada", "usa", "argentina", "brazil",
    "chile", "egypt", "kenya", "morocco", "nigeria",
    "south_africa", "australia", "new_zealand",
]

ALL_HIGH_SCHOOL_TRACKS = [
    "science", "social_studies", "languages", "religion", "vocational",
]

ALL_MAJOR_FIELDS = [
    "computer_science", "engineering", "medicine", "business",
    "economics", "law", "education", "arts_humanities",
    "social_sciences", "agriculture", "mathematics", "physics",
    "chemistry", "biology",
]

# Feature column definitions
STUDENT_CATEGORICAL_COLS = [
    "nationality", "high_school_track", "school_tier",
    "family_income_category", "intended_career_track", "olympiad_level",
]
STUDENT_NUMERICAL_COLS = [
    "age", "overall_report_card_average", "math_score",
    "english_score", "major_subject_average",
    "leadership_experience_count", "volunteer_experience_count",
    "competition_wins_count",
]
STUDENT_BOOLEAN_COLS = [
    "willing_to_return_home", "from_underrepresented_region",
    "needs_full_funding", "can_self_fund_living",
]

SCHOLARSHIP_CATEGORICAL_COLS = [
    "host_region", "preferred_school_tier",
    "career_track_preference", "max_family_income_category",
]
SCHOLARSHIP_NUMERICAL_COLS = [
    "min_age", "max_age", "min_report_card_average",
    "min_major_subject_average", "funding_monthly_stipend",
    "funding_coverage_count",
]
SCHOLARSHIP_BOOLEAN_COLS = [
    "requires_financial_need", "requires_return_home_country",
    "funding_covers_tuition", "funding_covers_living",
    "funding_covers_airfare", "funding_covers_insurance",
    "funding_is_full_funding",
]


def _parse_language_proficiency(json_str: str) -> np.ndarray:
    """Parse language proficiency JSON into a 12-dim vector."""
    vec = np.zeros(12, dtype=np.float32)
    try:
        records = json.loads(json_str) if isinstance(json_str, str) else json_str
    except (json.JSONDecodeError, TypeError):
        return vec
    if not isinstance(records, list):
        records = [records]
    for record in records:
        if not isinstance(record, dict):
            continue
        test_type = record.get("test_type", "").lower()
        if test_type in LANGUAGE_TESTS:
            idx = LANGUAGE_TESTS.index(test_type)
            score = float(record.get("score", 0.0))
            vec[idx * 2] = max(vec[idx * 2], score)
            vec[idx * 2 + 1] = 1.0
    return vec


def _encode_list_field(json_str: str, all_values: List[str]) -> np.ndarray:
    """Encode a list field as a binary vector."""
    vec = np.zeros(len(all_values), dtype=np.float32)
    try:
        items = json.loads(json_str) if isinstance(json_str, str) else json_str
    except (json.JSONDecodeError, TypeError):
        return vec
    if not isinstance(items, list):
        items = [items]
    for item in items:
        if item in all_values:
            idx = all_values.index(item)
            vec[idx] = 1.0
    return vec


def load_raw_data(
    students_csv: str = "./datasets_two_tower/students.csv",
    scholarships_csv: str = "./datasets_two_tower/scholarships.csv",
    pairs_csv: str = "./datasets_two_tower/pairs.csv",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load raw CSV data into pandas DataFrames."""
    students_df = pd.read_csv(students_csv)
    scholarships_df = pd.read_csv(scholarships_csv)
    pairs_df = pd.read_csv(pairs_csv)
    return students_df, scholarships_df, pairs_df


def split_pairs_time_based(
    pairs_df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split pairs by timestamp (already sorted in generator)."""
    pairs_df = pairs_df.sort_values("timestamp").reset_index(drop=True)
    n = len(pairs_df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    train_df = pairs_df.iloc[:train_end].copy()
    val_df = pairs_df.iloc[train_end:val_end].copy()
    test_df = pairs_df.iloc[val_end:].copy()
    return train_df, val_df, test_df


def _build_string_lookups(
    train_df: pd.DataFrame,
    categorical_cols: List[str],
) -> Dict[str, tf.keras.layers.StringLookup]:
    """Build StringLookup layers for the given categorical columns."""
    lookups = {}
    for col in categorical_cols:
        lookup = tf.keras.layers.StringLookup(
            mask_token=None,
            num_oov_indices=1,
            output_mode="int",
        )
        values = train_df[col].dropna().unique().tolist()
        lookup.adapt(tf.constant(values, dtype=tf.string))
        lookups[col] = lookup
    return lookups


def _build_normalizers(
    train_df: pd.DataFrame,
    numerical_cols: List[str],
) -> Dict[str, tf.keras.layers.Normalization]:
    """Build Normalization layers for the given numerical columns."""
    normalizers = {}
    for col in numerical_cols:
        values = train_df[col].dropna().values.astype(np.float32)
        normalizer = tf.keras.layers.Normalization(axis=None)
        normalizer.adapt(tf.constant(values))
        normalizers[col] = normalizer
    return normalizers


def _row_to_input_list(
    row: pd.Series,
) -> List[tf.Tensor]:
    """Convert a merged DataFrame row into a flat list of input tensors.

    The order MUST match the order of Input layers in the combined model:
        Student: 6 categorical, 8 numerical, 4 boolean, 1 language_vector
        Scholarship: 4 categorical, 6 numerical, 7 boolean, 1 list_vector
    """
    tensors = []

    # Student categorical (6)
    for col in STUDENT_CATEGORICAL_COLS:
        val = str(row[f"student_{col}"]) if pd.notna(row.get(f"student_{col}")) else ""
        tensors.append(tf.constant(val, dtype=tf.string))

    # Student numerical (8)
    for col in STUDENT_NUMERICAL_COLS:
        val = float(row.get(f"student_{col}", 0.0)) if pd.notna(row.get(f"student_{col}")) else 0.0
        tensors.append(tf.constant(val, dtype=tf.float32))

    # Student boolean (4)
    for col in STUDENT_BOOLEAN_COLS:
        val = float(row.get(f"student_{col}", 0.0)) if pd.notna(row.get(f"student_{col}")) else 0.0
        tensors.append(tf.constant(val, dtype=tf.float32))

    # Student language vector (12-dim)
    tensors.append(
        tf.constant(
            _parse_language_proficiency(row.get("student_language_proficiency", "[]")),
            dtype=tf.float32,
        )
    )

    # Scholarship categorical (4)
    for col in SCHOLARSHIP_CATEGORICAL_COLS:
        val = str(row[f"scholarship_{col}"]) if pd.notna(row.get(f"scholarship_{col}")) else ""
        tensors.append(tf.constant(val, dtype=tf.string))

    # Scholarship numerical (6)
    for col in SCHOLARSHIP_NUMERICAL_COLS:
        val = float(row.get(f"scholarship_{col}", 0.0)) if pd.notna(row.get(f"scholarship_{col}")) else 0.0
        tensors.append(tf.constant(val, dtype=tf.float32))

    # Scholarship boolean (7)
    for col in SCHOLARSHIP_BOOLEAN_COLS:
        val = float(row.get(f"scholarship_{col}", 0.0)) if pd.notna(row.get(f"scholarship_{col}")) else 0.0
        tensors.append(tf.constant(val, dtype=tf.float32))

    # Scholarship list vector (46-dim: 27 countries + 5 tracks + 14 fields)
    tensors.append(
        tf.constant(
            np.concatenate([
                _encode_list_field(row.get("scholarship_eligible_nationalities", "[]"), ALL_COUNTRIES),
                _encode_list_field(row.get("scholarship_eligible_high_school_tracks", "[]"), ALL_HIGH_SCHOOL_TRACKS),
                _encode_list_field(row.get("scholarship_eligible_fields", "[]"), ALL_MAJOR_FIELDS),
            ]),
            dtype=tf.float32,
        )
    )

    return tensors


def create_datasets(
    students_csv: str = "./datasets_two_tower/students.csv",
    scholarships_csv: str = "./datasets_two_tower/scholarships.csv",
    pairs_csv: str = "./datasets_two_tower/pairs.csv",
    batch_size: int = 2048,
    shuffle_buffer: int = 100_000,
    prefetch: int = tf.data.AUTOTUNE,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, Dict]:
    """Create train, validation, and test tf.data.Datasets.

    Each dataset yields (input_list, relevance_score) where input_list is a flat
    list of tensors matching the combined model's input order.

    Returns:
        (train_ds, val_ds, test_ds, preprocessors)
    """
    students_df, scholarships_df, pairs_df = load_raw_data(
        students_csv, scholarships_csv, pairs_csv
    )

    student_index = students_df.set_index("student_id")
    scholarship_index = scholarships_df.set_index("scholarship_id")

    train_pairs, val_pairs, test_pairs = split_pairs_time_based(pairs_df)

    def merge_features(pairs: pd.DataFrame) -> pd.DataFrame:
        merged = pairs.copy()
        merged = merged.merge(
            student_index.add_prefix("student_"),
            left_on="student_id", right_index=True, how="left",
        )
        merged = merged.merge(
            scholarship_index.add_prefix("scholarship_"),
            left_on="scholarship_id", right_index=True, how="left",
        )
        return merged

    train_merged = merge_features(train_pairs)
    val_merged = merge_features(val_pairs)
    test_merged = merge_features(test_pairs)

    # Build preprocessor layers
    print("Building StringLookup layers from training data...")
    _student_lookups_pf = _build_string_lookups(
        train_merged, [f"student_{c}" for c in STUDENT_CATEGORICAL_COLS]
    )
    student_lookups = {c: _student_lookups_pf[f"student_{c}"] for c in STUDENT_CATEGORICAL_COLS}

    _scholar_lookups_pf = _build_string_lookups(
        train_merged, [f"scholarship_{c}" for c in SCHOLARSHIP_CATEGORICAL_COLS]
    )
    scholarship_lookups = {c: _scholar_lookups_pf[f"scholarship_{c}"] for c in SCHOLARSHIP_CATEGORICAL_COLS}

    print("Building Normalization layers from training data...")
    _student_norm_pf = _build_normalizers(
        train_merged, [f"student_{c}" for c in STUDENT_NUMERICAL_COLS]
    )
    student_normalizers = {c: _student_norm_pf[f"student_{c}"] for c in STUDENT_NUMERICAL_COLS}

    _scholar_norm_pf = _build_normalizers(
        train_merged, [f"scholarship_{c}" for c in SCHOLARSHIP_NUMERICAL_COLS]
    )
    scholarship_normalizers = {c: _scholar_norm_pf[f"scholarship_{c}"] for c in SCHOLARSHIP_NUMERICAL_COLS}

    list_vector_dim = len(ALL_COUNTRIES) + len(ALL_HIGH_SCHOOL_TRACKS) + len(ALL_MAJOR_FIELDS)  # 46

    preprocessors = {
        "student_lookups": student_lookups,
        "scholarship_lookups": scholarship_lookups,
        "student_normalizers": student_normalizers,
        "scholarship_normalizers": scholarship_normalizers,
        "student_categorical_cols": STUDENT_CATEGORICAL_COLS,
        "student_numerical_cols": STUDENT_NUMERICAL_COLS,
        "student_boolean_cols": STUDENT_BOOLEAN_COLS,
        "scholarship_categorical_cols": SCHOLARSHIP_CATEGORICAL_COLS,
        "scholarship_numerical_cols": SCHOLARSHIP_NUMERICAL_COLS,
        "scholarship_boolean_cols": SCHOLARSHIP_BOOLEAN_COLS,
        "list_vector_dim": list_vector_dim,
        "language_vector_dim": len(LANGUAGE_TESTS) * 2,  # 12
    }

    def dataframe_to_dataset(df: pd.DataFrame, shuffle: bool = False) -> tf.data.Dataset:
        def gen():
            for _, row in df.iterrows():
                inputs = _row_to_input_list(row)
                relevance = tf.constant(
                    float(row["relevance_score"]) if pd.notna(row["relevance_score"]) else 0.0,
                    dtype=tf.float32,
                )
                yield (tuple(inputs), relevance)

        # Build output signature from first row
        first_inputs = _row_to_input_list(df.iloc[0])
        input_specs = tuple(
            tf.TensorSpec(shape=t.shape if t.shape.rank > 0 else (), dtype=t.dtype, name="")
            for t in first_inputs
        )

        output_signature = (input_specs, tf.TensorSpec(shape=(), dtype=tf.float32))

        ds = tf.data.Dataset.from_generator(gen, output_signature=output_signature)
        if shuffle:
            ds = ds.shuffle(shuffle_buffer)
        ds = ds.batch(batch_size, drop_remainder=False).prefetch(prefetch)
        return ds

    print(f"Training pairs:   {len(train_merged):,}")
    print(f"Validation pairs: {len(val_merged):,}")
    print(f"Test pairs:       {len(test_merged):,}")

    train_ds = dataframe_to_dataset(train_merged, shuffle=True)
    val_ds = dataframe_to_dataset(val_merged, shuffle=False)
    test_ds = dataframe_to_dataset(test_merged, shuffle=False)

    return train_ds, val_ds, test_ds, preprocessors