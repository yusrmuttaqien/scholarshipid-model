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
    students_df = pd.read_csv(students_csv)
    scholarships_df = pd.read_csv(scholarships_csv)
    pairs_df = pd.read_csv(pairs_csv)
    return students_df, scholarships_df, pairs_df


def split_pairs_time_based(
    pairs_df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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
    lookups = {}
    for col in categorical_cols:
        lookup = tf.keras.layers.StringLookup(
            mask_token=None,
            num_oov_indices=1,
            output_mode="int",
        )
        values = train_df[col].dropna().unique().tolist()
        if not values:
            values = [""]
        lookup.adapt(tf.constant(values, dtype=tf.string))
        lookups[col] = lookup
    return lookups


def _build_normalizers(
    train_df: pd.DataFrame,
    numerical_cols: List[str],
) -> Dict[str, tf.keras.layers.Normalization]:
    normalizers = {}
    for col in numerical_cols:
        values = train_df[col].dropna().values.astype(np.float32)
        if len(values) == 0:
            values = np.array([0.0], dtype=np.float32)
        normalizer = tf.keras.layers.Normalization(axis=None)
        normalizer.adapt(tf.constant(values))
        normalizers[col] = normalizer
    return normalizers


def _prepare_tensor_slices(merged_df: pd.DataFrame) -> Tuple[Tuple[np.ndarray, ...], np.ndarray]:
    """Pre-compute all input tensors as numpy arrays from merged DataFrame.

    This replaces the Python generator with bulk numpy operations,
    enabling from_tensor_slices() which is much faster at training time.

    Returns:
        (input_tensors_tuple, relevance_array)
    """
    n = len(merged_df)

    # Pre-allocate arrays for each input slot
    # Student categorical (6) — these are strings, store as object arrays
    student_cat_arrays = []
    for col in STUDENT_CATEGORICAL_COLS:
        arr = np.empty(n, dtype=object)
        for i, (_, row) in enumerate(merged_df.iterrows()):
            val = row.get(f"student_{col}")
            arr[i] = str(val) if pd.notna(val) else ""
        student_cat_arrays.append(arr)

    # Student numerical (8)
    student_num_arrays = []
    for col in STUDENT_NUMERICAL_COLS:
        arr = np.zeros(n, dtype=np.float32)
        for i, (_, row) in enumerate(merged_df.iterrows()):
            val = row.get(f"student_{col}")
            arr[i] = float(val) if pd.notna(val) else 0.0
        student_num_arrays.append(arr)

    # Student boolean (4)
    student_bool_arrays = []
    for col in STUDENT_BOOLEAN_COLS:
        arr = np.zeros(n, dtype=np.float32)
        for i, (_, row) in enumerate(merged_df.iterrows()):
            val = row.get(f"student_{col}")
            arr[i] = float(val) if pd.notna(val) else 0.0
        student_bool_arrays.append(arr)

    # Student language vector (12-dim)
    lang_vectors = np.zeros((n, 12), dtype=np.float32)
    for i, (_, row) in enumerate(merged_df.iterrows()):
        lang_vectors[i] = _parse_language_proficiency(
            row.get("student_language_proficiency", "[]")
        )

    # Scholarship categorical (4)
    scholar_cat_arrays = []
    for col in SCHOLARSHIP_CATEGORICAL_COLS:
        arr = np.empty(n, dtype=object)
        for i, (_, row) in enumerate(merged_df.iterrows()):
            val = row.get(f"scholarship_{col}")
            arr[i] = str(val) if pd.notna(val) else ""
        scholar_cat_arrays.append(arr)

    # Scholarship numerical (6)
    scholar_num_arrays = []
    for col in SCHOLARSHIP_NUMERICAL_COLS:
        arr = np.zeros(n, dtype=np.float32)
        for i, (_, row) in enumerate(merged_df.iterrows()):
            val = row.get(f"scholarship_{col}")
            arr[i] = float(val) if pd.notna(val) else 0.0
        scholar_num_arrays.append(arr)

    # Scholarship boolean (7)
    scholar_bool_arrays = []
    for col in SCHOLARSHIP_BOOLEAN_COLS:
        arr = np.zeros(n, dtype=np.float32)
        for i, (_, row) in enumerate(merged_df.iterrows()):
            val = row.get(f"scholarship_{col}")
            arr[i] = float(val) if pd.notna(val) else 0.0
        scholar_bool_arrays.append(arr)

    # Scholarship list vector (46-dim)
    list_dim = len(ALL_COUNTRIES) + len(ALL_HIGH_SCHOOL_TRACKS) + len(ALL_MAJOR_FIELDS)  # 46
    list_vectors = np.zeros((n, list_dim), dtype=np.float32)
    for i, (_, row) in enumerate(merged_df.iterrows()):
        list_vectors[i] = np.concatenate([
            _encode_list_field(row.get("scholarship_eligible_nationalities", "[]"), ALL_COUNTRIES),
            _encode_list_field(row.get("scholarship_eligible_high_school_tracks", "[]"), ALL_HIGH_SCHOOL_TRACKS),
            _encode_list_field(row.get("scholarship_eligible_fields", "[]"), ALL_MAJOR_FIELDS),
        ])

    # Relevance scores
    relevance = np.zeros(n, dtype=np.float32)
    for i, (_, row) in enumerate(merged_df.iterrows()):
        val = row.get("relevance_score")
        relevance[i] = float(val) if pd.notna(val) else 0.0

    # Combine in the order: student(6 cat, 8 num, 4 bool, 1 lang) + scholarship(4 cat, 6 num, 7 bool, 1 list)
    input_tensors = (
        *student_cat_arrays,   # 6
        *student_num_arrays,   # 8
        *student_bool_arrays,  # 4
        lang_vectors,          # 1 (12-dim)
        *scholar_cat_arrays,   # 4
        *scholar_num_arrays,   # 6
        *scholar_bool_arrays,  # 7
        list_vectors,          # 1 (46-dim)
    )

    return input_tensors, relevance


def _make_tensor_spec_from_array(arr: np.ndarray) -> tf.TensorSpec:
    """Create a TensorSpec from a numpy array."""
    if arr.dtype == object:
        # String input: scalar shape
        return tf.TensorSpec(shape=(), dtype=tf.string)
    elif len(arr.shape) == 1:
        # 1D numeric: scalar shape
        return tf.TensorSpec(shape=(), dtype=tf.float32)
    else:
        # Multi-dim: preserve last dim
        return tf.TensorSpec(shape=(arr.shape[1],), dtype=tf.float32)


def create_datasets(
    students_csv: str = "./datasets_two_tower/students.csv",
    scholarships_csv: str = "./datasets_two_tower/scholarships.csv",
    pairs_csv: str = "./datasets_two_tower/pairs.csv",
    batch_size: int = 2048,
    shuffle_buffer: int = 100_000,
    prefetch: int = tf.data.AUTOTUNE,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, Dict]:
    """Create train, validation, and test tf.data.Datasets.

    Uses from_tensor_slices for fast tensor-based data loading.

    Each dataset yields (input_tuple, relevance_score) where input_tuple is a
    flat tuple of tensors matching the combined model's input order.

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
        # Pre-compute all numpy arrays
        input_tensors, relevance = _prepare_tensor_slices(df)

        # Build TensorSpecs
        input_specs = tuple(_make_tensor_spec_from_array(arr) for arr in input_tensors)
        output_signature = (input_specs, tf.TensorSpec(shape=(), dtype=tf.float32))

        # Create dataset from tensor slices — MUCH faster than from_generator
        ds = tf.data.Dataset.from_tensor_slices((input_tensors, relevance))

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