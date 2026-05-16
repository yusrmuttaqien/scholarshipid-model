"""Two-Tower Recommendation System — Inference Pipeline

Three-stage pipeline for producing ranked scholarship recommendations
for a given student:

    Stage 1: Hard Filter — Deterministic eligibility checks
    Stage 2: Two-Tower NN — Soft similarity scoring
    Stage 3: Text Bonus — TF-IDF similarity bonus
"""

import json
import pickle
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from data import (
    ALL_COUNTRIES,
    ALL_HIGH_SCHOOL_TRACKS,
    ALL_MAJOR_FIELDS,
    LANGUAGE_TESTS,
    _parse_language_proficiency,
    _encode_list_field,
    create_inference_data,
)
from model import CosineSimilarity, WeightedMSE


# ============================================================
# Stage 1: Hard Filter
# ============================================================


def _check_nationality(student_nationality: str, eligible_nationalities: list) -> bool:
    """Check if student nationality is in eligible list."""
    if not eligible_nationalities:
        return True
    return student_nationality in eligible_nationalities


def _check_age(student_age: float, min_age: float, max_age: float) -> bool:
    """Check if student age is within scholarship range."""
    if pd.isna(min_age) or pd.isna(max_age):
        return True
    return min_age <= student_age <= max_age


def _check_report_card(student_avg: float, min_avg: float) -> bool:
    """Check if student report card average meets minimum."""
    if pd.isna(min_avg):
        return True
    return student_avg >= min_avg


def _check_major_subject(student_avg: float, min_avg: float) -> bool:
    """Check if student major subject average meets minimum."""
    if pd.isna(min_avg):
        return True
    return student_avg >= min_avg


def _check_language_requirements(
    student_lang: list,
    scholarship_lang_reqs: list,
) -> bool:
    """Check if student meets all mandatory language requirements.

    Args:
        student_lang: List of dicts with test_type and score.
        scholarship_lang_reqs: List of dicts with test_type, min_score, is_mandatory.

    Returns:
        True if all mandatory requirements are met.
    """
    if not scholarship_lang_reqs:
        return True

    # Build student's best scores
    student_best = {}
    for entry in student_lang:
        if isinstance(entry, dict):
            ttype = entry.get("test_type", "")
            score = float(entry.get("score", 0.0))
            if ttype:
                student_best[ttype] = max(student_best.get(ttype, 0.0), score)

    for req in scholarship_lang_reqs:
        if isinstance(req, str):
            try:
                req = json.loads(req)
            except (json.JSONDecodeError, TypeError):
                continue
        if not isinstance(req, dict):
            continue

        test_type = req.get("test_type", "")
        min_score = float(req.get("min_score", 0.0))
        is_mandatory = req.get("is_mandatory", False)

        if is_mandatory and test_type:
            if test_type not in student_best:
                return False
            if student_best[test_type] < min_score:
                return False

    return True


def _check_return_home(
    requires_return: bool,
    student_willing: bool,
) -> bool:
    """Check return home requirement."""
    if not requires_return:
        return True
    return student_willing


def _check_financial_need(
    requires_financial_need: bool,
    student_income: str,
    max_income_category: str,
) -> bool:
    """Check financial need requirement.

    If scholarship requires financial need, student income must be
    <= max_family_income_category.
    """
    if not requires_financial_need:
        return True

    income_order = ["very_low", "low", "middle", "upper_middle", "high"]
    student_idx = income_order.index(student_income) if student_income in income_order else 99
    max_idx = income_order.index(max_income_category) if max_income_category in income_order else 99

    return student_idx <= max_idx


def hard_filter(
    student_row: pd.Series,
    scholarships_df: pd.DataFrame,
) -> pd.DataFrame:
    """Apply deterministic hard filters to scholarships.

    Args:
        student_row: A pandas Series with student features.
        scholarships_df: DataFrame of all scholarships.

    Returns:
        Filtered DataFrame of eligible scholarships only.
    """
    eligible = []

    for _, sch in scholarships_df.iterrows():
        # Parse list fields
        eligible_nationalities = []
        try:
            raw = sch.get("eligible_nationalities", "[]")
            eligible_nationalities = json.loads(raw) if isinstance(raw, str) else raw
        except (json.JSONDecodeError, TypeError):
            eligible_nationalities = []

        if not _check_nationality(student_row.get("nationality", ""), eligible_nationalities):
            continue

        if not _check_age(
            float(student_row.get("age", 0)),
            float(sch.get("min_age", 0)),
            float(sch.get("max_age", 99)),
        ):
            continue

        if not _check_report_card(
            float(student_row.get("overall_report_card_average", 0)),
            float(sch.get("min_report_card_average", 0)),
        ):
            continue

        if not _check_major_subject(
            float(student_row.get("major_subject_average", 0)),
            float(sch.get("min_major_subject_average", 0)),
        ):
            continue

        # Language requirements
        student_lang = []
        try:
            raw = student_row.get("language_proficiency", "[]")
            student_lang = json.loads(raw) if isinstance(raw, str) else raw
        except (json.JSONDecodeError, TypeError):
            student_lang = []

        sch_lang_reqs = []
        try:
            raw = sch.get("language_requirements", "[]")
            sch_lang_reqs = json.loads(raw) if isinstance(raw, str) else raw
        except (json.JSONDecodeError, TypeError):
            sch_lang_reqs = []

        if not _check_language_requirements(student_lang, sch_lang_reqs):
            continue

        if not _check_return_home(
            bool(sch.get("requires_return_home_country", False)),
            bool(student_row.get("willing_to_return_home", True)),
        ):
            continue

        if not _check_financial_need(
            bool(sch.get("requires_financial_need", False)),
            str(student_row.get("family_income_category", "")),
            str(sch.get("max_family_income_category", "high")),
        ):
            continue

        eligible.append(sch)

    if not eligible:
        return pd.DataFrame()

    return pd.DataFrame(eligible)


# ============================================================
# Stage 2: Two-Tower NN Scoring
# ============================================================


def _build_scholarship_batch(
    scholarships_df: pd.DataFrame,
    preprocessors: Dict,
) -> Dict[str, tf.Tensor]:
    """Convert a DataFrame of scholarships into model input dict.

    Args:
        scholarships_df: DataFrame of scholarship features.
        preprocessors: Dict from training containing lookups/normalizers.

    Returns:
        Dict of scholarship feature tensors, each with batch dimension.
    """
    features = {}

    # Categorical
    for col in preprocessors["scholarship_categorical_cols"]:
        values = []
        for _, row in scholarships_df.iterrows():
            val = str(row.get(col, "")) if pd.notna(row.get(col)) else ""
            values.append(val)
        features[col] = tf.constant(values, dtype=tf.string)

    # Numerical
    for col in preprocessors["scholarship_numerical_cols"]:
        values = []
        for _, row in scholarships_df.iterrows():
            val = float(row.get(col, 0.0)) if pd.notna(row.get(col)) else 0.0
            values.append(val)
        features[col] = tf.constant(values, dtype=tf.float32)

    # Boolean
    for col in preprocessors["scholarship_boolean_cols"]:
        values = []
        for _, row in scholarships_df.iterrows():
            val = float(row.get(col, 0.0)) if pd.notna(row.get(col)) else 0.0
            values.append(val)
        features[col] = tf.constant(values, dtype=tf.float32)

    # List vector
    list_vectors = []
    for _, row in scholarships_df.iterrows():
        vec = np.concatenate([
            _encode_list_field(
                row.get("eligible_nationalities", "[]"),
                ALL_COUNTRIES,
            ),
            _encode_list_field(
                row.get("eligible_high_school_tracks", "[]"),
                ALL_HIGH_SCHOOL_TRACKS,
            ),
            _encode_list_field(
                row.get("eligible_fields", "[]"),
                ALL_MAJOR_FIELDS,
            ),
        ])
        list_vectors.append(vec)

    features["list_vector"] = tf.constant(
        np.array(list_vectors, dtype=np.float32),
        dtype=tf.float32,
    )

    return features


def nn_score(
    model: keras.Model,
    student_features: Dict[str, tf.Tensor],
    scholarships_df: pd.DataFrame,
    preprocessors: Dict,
) -> np.ndarray:
    """Compute NN similarity scores for all eligible scholarships.

    Args:
        model: Trained two-tower model.
        student_features: Dict of student feature tensors (single sample).
        scholarships_df: DataFrame of eligible scholarships.
        preprocessors: Dict from training.

    Returns:
        Array of predicted relevance scores, shape (num_scholarships,).
    """
    if scholarships_df.empty:
        return np.array([])

    # Build scholarship batch
    sch_features = _build_scholarship_batch(scholarships_df, preprocessors)

    # Repeat student features for each scholarship
    num_scholarships = len(scholarships_df)

    # Build flattened input list (order must match build_model)
    inputs = []

    # Student inputs (repeated for each scholarship)
    for col in preprocessors["student_categorical_cols"]:
        val = student_features[col]
        inputs.append(tf.repeat(tf.expand_dims(val, 0), num_scholarships, axis=0))

    for col in preprocessors["student_numerical_cols"]:
        val = student_features[col]
        inputs.append(tf.repeat(tf.expand_dims(val, 0), num_scholarships, axis=0))

    for col in preprocessors["student_boolean_cols"]:
        val = student_features[col]
        inputs.append(tf.repeat(tf.expand_dims(val, 0), num_scholarships, axis=0))

    inputs.append(
        tf.repeat(
            tf.expand_dims(student_features["language_vector"], 0),
            num_scholarships,
            axis=0,
        )
    )

    # Scholarship inputs
    for col in preprocessors["scholarship_categorical_cols"]:
        inputs.append(sch_features[col])

    for col in preprocessors["scholarship_numerical_cols"]:
        inputs.append(sch_features[col])

    for col in preprocessors["scholarship_boolean_cols"]:
        inputs.append(sch_features[col])

    inputs.append(sch_features["list_vector"])

    # Predict
    predictions = model.predict(inputs, verbose=0)
    return predictions.flatten()


# ============================================================
# Stage 3: Text Similarity Bonus
# ============================================================


def compute_text_similarity_bonus(
    student_texts: Dict[str, str],
    scholarship_texts: Dict[str, str],
    max_bonus: float = 0.1,
) -> float:
    """Compute TF-IDF text similarity bonus between student and scholarship texts.

    Pairs compared:
        - personal_statement vs mission_statement
        - future_goals vs target_recipient_profile

    Args:
        student_texts: Dict with 'personal_statement', 'future_goals'.
        scholarship_texts: Dict with 'mission_statement', 'target_recipient_profile'.
        max_bonus: Maximum bonus to add (default 0.1).

    Returns:
        Bonus score in [0.0, max_bonus].
    """
    pairs = [
        (student_texts.get("personal_statement", ""),
         scholarship_texts.get("mission_statement", "")),
        (student_texts.get("future_goals", ""),
         scholarship_texts.get("target_recipient_profile", "")),
    ]

    bonuses = []
    for text_a, text_b in pairs:
        if not text_a or not text_b:
            bonuses.append(0.0)
            continue

        try:
            vectorizer = TfidfVectorizer(stop_words="english")
            tfidf_matrix = vectorizer.fit_transform([text_a, text_b])
            sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            bonuses.append(float(sim))
        except Exception:
            bonuses.append(0.0)

    # Average similarity across pairs, scaled to max_bonus
    avg_sim = np.mean(bonuses) if bonuses else 0.0
    bonus = avg_sim * max_bonus

    return min(bonus, max_bonus)


# ============================================================
# Full Inference Pipeline
# ============================================================


def recommend(
    student_id: str,
    model: keras.Model,
    preprocessors: Dict,
    students_df: pd.DataFrame,
    scholarships_df: pd.DataFrame,
    top_k: int = 20,
    text_bonus_enabled: bool = True,
    text_bonus_max: float = 0.1,
) -> pd.DataFrame:
    """Run full 3-stage inference pipeline for a single student.

    Args:
        student_id: Student identifier.
        model: Trained two-tower model.
        preprocessors: Dict from training containing lookups/normalizers.
        students_df: DataFrame of all students.
        scholarships_df: DataFrame of all scholarships.
        top_k: Number of top recommendations to return.
        text_bonus_enabled: Whether to apply Stage 3 text bonus.
        text_bonus_max: Maximum text bonus value.

    Returns:
        DataFrame with columns:
            scholarship_id, name, nn_score, text_bonus, final_score
        Sorted by final_score descending, limited to top_k.
    """
    # Look up student
    if student_id not in students_df.index:
        raise ValueError(f"Student {student_id} not found in dataset")

    student_row = students_df.loc[student_id]
    if isinstance(student_row, pd.DataFrame):
        student_row = student_row.iloc[0]

    # ── Stage 1: Hard Filter ────────────────────────────────
    print(f"[Stage 1] Applying hard filters for student {student_id}...")
    eligible_df = hard_filter(student_row, scholarships_df)
    print(f"  {len(eligible_df)} of {len(scholarships_df)} scholarships passed hard filters")

    if eligible_df.empty:
        print("  No eligible scholarships found.")
        return pd.DataFrame(
            columns=["scholarship_id", "name", "nn_score", "text_bonus", "final_score"]
        )

    # Build student features for NN
    student_features = _build_student_features(student_row, preprocessors)

    # ── Stage 2: Two-Tower NN ───────────────────────────────
    print(f"[Stage 2] Computing NN similarity scores...")
    nn_scores = nn_score(model, student_features, eligible_df, preprocessors)

    if len(nn_scores) == 0:
        print("  No scores computed.")
        return pd.DataFrame(
            columns=["scholarship_id", "name", "nn_score", "text_bonus", "final_score"]
        )

    eligible_df = eligible_df.copy()
    eligible_df["nn_score"] = nn_scores

    # ── Stage 3: Text Similarity Bonus ──────────────────────
    if text_bonus_enabled:
        print(f"[Stage 3] Computing text similarity bonus...")
        text_bonuses = []
        student_texts = {
            "personal_statement": str(student_row.get("personal_statement", "")),
            "future_goals": str(student_row.get("future_goals", "")),
        }

        for _, sch_row in eligible_df.iterrows():
            scholarship_texts = {
                "mission_statement": str(sch_row.get("mission_statement", "")),
                "target_recipient_profile": str(sch_row.get("target_recipient_profile", "")),
            }
            bonus = compute_text_similarity_bonus(
                student_texts, scholarship_texts, max_bonus=text_bonus_max
            )
            text_bonuses.append(bonus)

        eligible_df["text_bonus"] = text_bonuses
    else:
        eligible_df["text_bonus"] = 0.0

    # Final score: clamp to [0, 1]
    eligible_df["final_score"] = (eligible_df["nn_score"] + eligible_df["text_bonus"]).clip(0.0, 1.0)

    # Sort and return top-K
    result = eligible_df.sort_values("final_score", ascending=False).head(top_k)

    return result[["scholarship_id", "name", "nn_score", "text_bonus", "final_score"]]


def _build_student_features(
    student_row: pd.Series,
    preprocessors: Dict,
) -> Dict[str, tf.Tensor]:
    """Build student features dict from a pandas Series.

    Args:
        student_row: A pandas Series with student features.
        preprocessors: Dict from training.

    Returns:
        Dict of student feature tensors (single sample).
    """
    features = {}

    for col in preprocessors["student_categorical_cols"]:
        val = str(student_row.get(col, "")) if pd.notna(student_row.get(col)) else ""
        features[col] = tf.constant(val, dtype=tf.string)

    for col in preprocessors["student_numerical_cols"]:
        val = float(student_row.get(col, 0.0)) if pd.notna(student_row.get(col)) else 0.0
        features[col] = tf.constant(val, dtype=tf.float32)

    for col in preprocessors["student_boolean_cols"]:
        val = float(student_row.get(col, 0.0)) if pd.notna(student_row.get(col)) else 0.0
        features[col] = tf.constant(val, dtype=tf.float32)

    features["language_vector"] = tf.constant(
        _parse_language_proficiency(
            student_row.get("language_proficiency", "[]")
        ),
        dtype=tf.float32,
    )

    return features


# ============================================================
# CLI Entry Point
# ============================================================


def main():
    """CLI entry point for inference.

    Usage:
        python inference.py STU_000001 --top_k 20
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Two-Tower Recommendation Inference"
    )
    parser.add_argument("student_id", type=str, help="Student ID (e.g., STU_000001)")
    parser.add_argument(
        "--model", type=str, default="./two_tower_model.keras",
        help="Path to trained model",
    )
    parser.add_argument(
        "--preprocessors", type=str, default="./two_tower_model_preprocessors.pkl",
        help="Path to preprocessors pickle",
    )
    parser.add_argument(
        "--students", type=str, default="./datasets_two_tower/students.csv",
        help="Path to students CSV",
    )
    parser.add_argument(
        "--scholarships", type=str, default="./datasets_two_tower/scholarships.csv",
        help="Path to scholarships CSV",
    )
    parser.add_argument(
        "--top_k", type=int, default=20,
        help="Number of top recommendations",
    )
    parser.add_argument(
        "--no-text-bonus", action="store_true",
        help="Disable text similarity bonus",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Two-Tower Recommendation System — Inference")
    print("=" * 60)

    # Load data
    print(f"\nLoading students from {args.students}...")
    students_df = pd.read_csv(args.students).set_index("student_id")

    print(f"Loading scholarships from {args.scholarships}...")
    scholarships_df = pd.read_csv(args.scholarships)

    # Load model
    print(f"Loading model from {args.model}...")
    model = keras.models.load_model(
        args.model,
        custom_objects={
            "CosineSimilarity": CosineSimilarity,
            "WeightedMSE": WeightedMSE,
        },
    )

    # Load preprocessors
    print(f"Loading preprocessors from {args.preprocessors}...")
    with open(args.preprocessors, "rb") as f:
        preprocessors = pickle.load(f)

    # Run inference
    print(f"\nGenerating recommendations for {args.student_id}...")
    print("=" * 60)

    result = recommend(
        student_id=args.student_id,
        model=model,
        preprocessors=preprocessors,
        students_df=students_df,
        scholarships_df=scholarships_df,
        top_k=args.top_k,
        text_bonus_enabled=not args.no_text_bonus,
    )

    print(f"\nTop {len(result)} Recommendations for {args.student_id}:")
    print("=" * 60)
    print(f"{'Rank':<5} {'Scholarship ID':<20} {'Name':<50} {'NN Score':<10} {'Bonus':<8} {'Final':<8}")
    print("-" * 100)

    for rank, (_, row) in enumerate(result.iterrows(), 1):
        name_trunc = str(row["name"])[:47] + "..." if len(str(row["name"])) > 50 else str(row["name"])
        print(
            f"{rank:<5} {str(row['scholarship_id']):<20} "
            f"{name_trunc:<50} "
            f"{row['nn_score']:.4f}   "
            f"{row['text_bonus']:.4f}  "
            f"{row['final_score']:.4f}"
        )

    print("=" * 60)


if __name__ == "__main__":
    main()