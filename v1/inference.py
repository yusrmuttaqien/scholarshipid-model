"""Recommendation System — Inference Engine (v1)

Two-stage pipeline:
  Stage 1: Hard filter (deterministic eligibility checks)
  Stage 2: Two-tower neural network scoring

Usage as CLI:
    python inference.py --student-id STU_000001 --top-k 5

Usage as library:
    from v1.inference import InferenceEngine
    engine = InferenceEngine()
    recommendations = engine.recommend(student_id, student, scholarships)
"""

import argparse
import json
import pickle
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional, Dict, Any

import numpy as np
import pandas as pd
import warnings

# Suppress harmless Keras input structure warnings (model was saved with named inputs)
warnings.filterwarnings("ignore", message=".*structure of `inputs` doesn't match.*")

# Add project root to path so `from src.schemas import ...` works
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ============================================================
# Constants — must match train.py exactly
# ============================================================

_SCRIPT_DIR = Path(__file__).resolve().parent
_MODEL_DIR = _SCRIPT_DIR / "models"

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

LIST_COUNTRY_DIM = 27
LIST_TRACK_DIM = 5
LIST_FIELD_DIM = 14
LIST_VECTOR_DIM = LIST_COUNTRY_DIM + LIST_TRACK_DIM + LIST_FIELD_DIM  # 46

ALL_COUNTRIES = [
    "china", "india", "indonesia", "japan", "malaysia",
    "philippines", "singapore", "south_korea", "thailand", "vietnam",
    "france", "germany", "netherlands", "sweden", "uk",
    "switzerland", "canada", "usa", "argentina", "brazil",
    "chile", "egypt", "kenya", "morocco", "nigeria",
    "south_africa", "australia", "new_zealand",
]
ALL_TRACKS = ["science", "social_studies", "languages", "religion", "vocational"]
ALL_FIELDS = [
    "computer_science", "engineering", "medicine", "business",
    "economics", "law", "education", "arts_humanities",
    "social_sciences", "agriculture", "mathematics", "physics",
    "chemistry", "biology",
]
ALL_LIST_VALUES = ALL_COUNTRIES + ALL_TRACKS + ALL_FIELDS

# Income category ordering for comparison
INCOME_ORDER = ["very_low", "low", "middle", "upper_middle", "high"]


# ============================================================
# Custom Layer (must match train.py)
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
# Tower Builders (must match train.py)
# ============================================================

def _build_student_tower(input_dim: int, embedding_dim: int = 64) -> keras.Model:
    inp = layers.Input(shape=(input_dim,), dtype=tf.float32, name="student_inputs")
    x = layers.Dense(128, activation="relu", name="student_dense_128")(inp)
    x = layers.BatchNormalization(name="student_bn_128")(x)
    x = layers.Dense(64, activation="relu", name="student_dense_64")(x)
    x = layers.BatchNormalization(name="student_bn_64")(x)
    embedding = layers.Dense(embedding_dim, activation=None, name="student_embedding")(x)
    return keras.Model(inputs=inp, outputs=embedding, name="student_tower")


def _build_scholarship_tower(input_dim: int, embedding_dim: int = 64) -> keras.Model:
    inp = layers.Input(shape=(input_dim,), dtype=tf.float32, name="scholarship_inputs")
    x = layers.Dense(128, activation="relu", name="scholarship_dense_128")(inp)
    x = layers.BatchNormalization(name="scholarship_bn_128")(x)
    x = layers.Dense(64, activation="relu", name="scholarship_dense_64")(x)
    x = layers.BatchNormalization(name="scholarship_bn_64")(x)
    embedding = layers.Dense(embedding_dim, activation=None, name="scholarship_embedding")(x)
    return keras.Model(inputs=inp, outputs=embedding, name="scholarship_tower")


# ============================================================
# Feature Preprocessing — must match train.py exactly
# ============================================================

def _encode_categorical(df: pd.DataFrame, col: str) -> tuple[np.ndarray, dict]:
    """Label-encode a categorical column."""
    col_vals = df[col].astype(str).fillna("unknown")
    unique_vals = sorted(col_vals.unique())
    mapping = {v: i + 1 for i, v in enumerate(unique_vals)}
    arr = np.zeros(len(df), dtype=np.int32)
    for val, idx in mapping.items():
        mask = col_vals == val
        arr[mask] = idx
    return arr, mapping


def _encode_list_field(json_str: str, all_values: list) -> np.ndarray:
    """Encode a JSON list field as binary vector."""
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


def _parse_language_proficiency(json_str: Optional[str]) -> np.ndarray:
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


def _prepare_student_features(student, student_mapping: dict) -> np.ndarray:
    """Convert a single Student object to packed feature vector."""
    parts = []

    # Categorical fields
    for col in STUDENT_CATEGORICAL:
        val = str(getattr(student, col, "") or "unknown")
        mapping = student_mapping[col]
        encoded = 0 if val not in mapping else mapping[val]
        parts.append(np.array([encoded], dtype=np.int32))

    # Numerical fields
    for col in STUDENT_NUMERICAL:
        val = getattr(student, col, 0.0) or 0.0
        parts.append(np.array([float(val)], dtype=np.float32))

    # Boolean fields (defaults to False if missing)
    for col in STUDENT_BOOLEAN:
        val = getattr(student, col, False)
        parts.append(np.array([1.0 if val else 0.0], dtype=np.float32))

    # Language proficiency vector
    lang_field = getattr(student, "language_proficiency", [])
    if isinstance(lang_field, str):
        lang_vec = _parse_language_proficiency(lang_field)
    elif isinstance(lang_field, list) and len(lang_field) > 0:
        first_entry = lang_field[0]
        if hasattr(first_entry, "test_type"):
            # List of LanguageProficiency dataclasses → convert to dicts
            clean_records = [asdict(lp) for lp in lang_field]
            lang_vec = _parse_language_proficiency(json.dumps(clean_records))
        else:
            # Already a list of dicts or strings
            lang_vec = _parse_language_proficiency(json.dumps(lang_field))
    else:
        lang_vec = np.zeros(12, dtype=np.float32)
    parts.append(lang_vec)

    return np.hstack(parts).astype(np.float32)


def prepare_scholarship_features(scholarships, scholarship_mapping: dict) -> np.ndarray:
    """Convert a list of Scholarship objects to packed feature matrix."""
    df = pd.DataFrame([asdict(s) for s in scholarships])

    # Flatten nested fields
    if "language_requirements" in df.columns:
        df["language_requirements"] = df["language_requirements"].apply(
            lambda x: json.dumps(x) if isinstance(x, list) else str(x)
        )
    if "selection_criteria" in df.columns:
        df["selection_criteria"] = df["selection_criteria"].apply(
            lambda x: json.dumps(asdict(x)) if hasattr(x, 'asdict') else str(x)
        )

    sch_parts = []

    for col in SCHOLARSHIP_CATEGORICAL:
        arr, _ = _encode_categorical(df, col)
        sch_parts.append(arr[:, None])

    for col in SCHOLARSHIP_NUMERICAL:
        if col in df.columns:
            sch_parts.append(df[col].values.astype(np.float32)[:, None])
        else:
            sch_parts.append(np.zeros((len(scholarships), 1), dtype=np.float32))

    for col in SCHOLARSHIP_BOOLEAN:
        if col in df.columns:
            sch_parts.append(df[col].astype(float).fillna(0).values[:, None])
        else:
            sch_parts.append(np.zeros((len(scholarships), 1), dtype=np.float32))

    # List vector — matches original train.py behavior (which has a subtle bug where
    # _make_list_vectors ignores the `offset` param and always takes vec[:dim])
    list_vec = np.zeros((len(scholarships), LIST_VECTOR_DIM), dtype=np.float32)

    for i, s in enumerate(scholarships):
        eligible_nat = getattr(s, "eligible_nationalities", [])
        if isinstance(eligible_nat, str):
            eligible_nat = json.loads(eligible_nat) if eligible_nat else []
        nat_vec = _encode_list_field(json.dumps(eligible_nat), ALL_LIST_VALUES)
        list_vec[i, :LIST_COUNTRY_DIM] = nat_vec[:LIST_COUNTRY_DIM]

        eligible_tracks = getattr(s, "eligible_high_school_tracks", [])
        if isinstance(eligible_tracks, str):
            eligible_tracks = json.loads(eligible_tracks) if eligible_tracks else []
        track_vec = _encode_list_field(json.dumps(eligible_tracks), ALL_LIST_VALUES)
        list_vec[i, LIST_COUNTRY_DIM:LIST_COUNTRY_DIM + LIST_TRACK_DIM] += track_vec[:LIST_TRACK_DIM]

        eligible_fields = getattr(s, "eligible_fields", [])
        if isinstance(eligible_fields, str):
            eligible_fields = json.loads(eligible_fields) if eligible_fields else []
        field_vec = _encode_list_field(json.dumps(eligible_fields), ALL_LIST_VALUES)
        list_vec[i, LIST_COUNTRY_DIM + LIST_TRACK_DIM:] += field_vec[:LIST_FIELD_DIM]

    sch_parts.append(list_vec)
    return np.hstack(sch_parts).astype(np.float32)


# ============================================================
# Hard Filter (Stage 1 from README)
# ============================================================

def _check_eligible_nationalities(student, scholarship) -> bool:
    """Student nationality must be in eligible_nationalities."""
    student_nat = getattr(student, "nationality", "").lower().strip()
    if not student_nat:
        return False
    eligible = getattr(scholarship, "eligible_nationalities", [])
    if isinstance(eligible, str):
        try:
            eligible = json.loads(eligible)
        except (json.JSONDecodeError, TypeError):
            eligible = [eligible]
    for nat in eligible:
        if str(nat).lower().strip() == student_nat:
            return True
    return False


def _check_age(student, scholarship) -> bool:
    """Student age within [min_age, max_age]."""
    age = getattr(student, "age", 0)
    min_age = getattr(scholarship, "min_age", 0)
    max_age = getattr(scholarship, "max_age", 100)
    return min_age <= age <= max_age


def _check_degree_level(student, scholarship) -> bool:
    """Student target degree level must be in eligible levels."""
    target = str(getattr(student, "target_degree_level", "")).lower()
    eligible = getattr(scholarship, "eligible_degree_levels", [])
    if isinstance(eligible, str):
        try:
            eligible = json.loads(eligible)
        except (json.JSONDecodeError, TypeError):
            eligible = [eligible]
    return target in [str(e).lower() for e in eligible]


def _check_high_school_track(student, scholarship) -> bool:
    """Student high school track must be eligible."""
    track = str(getattr(student, "high_school_track", "")).lower()
    eligible = getattr(scholarship, "eligible_high_school_tracks", [])
    if isinstance(eligible, str):
        try:
            eligible = json.loads(eligible)
        except (json.JSONDecodeError, TypeError):
            eligible = [eligible]
    return track in [str(e).lower() for e in eligible]


def _check_language_requirements(student, scholarship) -> bool:
    """Student must meet mandatory language requirements."""
    student_langs = getattr(student, "language_proficiency", [])
    if isinstance(student_langs, str):
        try:
            student_langs = json.loads(student_langs)
        except (json.JSONDecodeError, TypeError):
            student_langs = []
    elif isinstance(student_langs, list) and len(student_langs) > 0:
        first = student_langs[0]
        if hasattr(first, "test_type"):
            student_langs = [asdict(lp) for lp in student_langs]

    reqs = getattr(scholarship, "language_requirements", [])
    if not reqs:
        return True

    # Build lookup: test_type → max score from student
    lang_lookup = {}
    for entry in student_langs:
        if isinstance(entry, dict):
            tt = entry.get("test_type", "").lower()
            sc = float(entry.get("score", 0))
            lang_lookup[tt] = max(lang_lookup.get(tt, 0), sc)

    for req in (reqs if isinstance(reqs, list) else []):
        if isinstance(req, dict):
            test_type = req.get("test_type", "").lower()
            min_score = float(req.get("min_score", 0))
            is_mandatory = bool(req.get("is_mandatory", True))
        else:
            test_type = getattr(req, "test_type", "").lower()
            min_score = float(getattr(req, "min_score", 0))
            is_mandatory = bool(getattr(req, "is_mandatory", True))

        if not is_mandatory:
            continue
        student_max = lang_lookup.get(test_type, 0)
        if student_max < min_score:
            return False
    return True


def _check_return_home(student, scholarship) -> bool:
    """If scholarship requires return-home, student must be willing."""
    requires = getattr(scholarship, "requires_return_home_country", False)
    if not requires:
        return True
    return bool(getattr(student, "willing_to_return_home", True))


def _check_financial_need(student, scholarship) -> bool:
    """If scholarship requires financial need, student's income must be low enough."""
    requires = getattr(scholarship, "requires_financial_need", False)
    if not requires:
        return True

    max_cat = str(getattr(scholarship, "max_family_income_category", "high")).lower()
    student_cat = str(getattr(student, "family_income_category", "middle")).lower()

    max_idx = INCOME_ORDER.index(max_cat) if max_cat in INCOME_ORDER else len(INCOME_ORDER) - 1
    student_idx = INCOME_ORDER.index(student_cat) if student_cat in INCOME_ORDER else len(INCOME_ORDER) - 1

    return student_idx <= max_idx


def _check_academic_thresholds(student, scholarship) -> bool:
    """Minimum report card and major subject averages."""
    min_rc = getattr(scholarship, "min_report_card_average", 0)
    min_major = getattr(scholarship, "min_major_subject_average", 0)
    return (float(getattr(student, "overall_report_card_average", 0)) >= min_rc and
            float(getattr(student, "major_subject_average", 0)) >= min_major)


def _check_eligible_fields(student, scholarship) -> bool:
    """Student's intended career track should align with eligible fields."""
    eligible = getattr(scholarship, "eligible_fields", [])
    if not eligible:
        return True
    # This is a soft check — we don't hard-filter on this
    # The two-tower model will capture this signal
    return True


def _check_high_school_track_eligible(student, scholarship) -> bool:
    """Student's high school track must be eligible."""
    track = str(getattr(student, "high_school_track", "")).lower()
    tracks = getattr(scholarship, "eligible_high_school_tracks", [])
    if isinstance(tracks, str):
        try:
            tracks = json.loads(tracks)
        except (json.JSONDecodeError, TypeError):
            tracks = [tracks]
    return track in [str(t).lower() for t in tracks]


def apply_hard_filter(student, scholarship) -> bool:
    """Run all hard filter checks. Returns True if eligible.

    Note: return-home obligation and strict financial need are NOT hard filters.
    These are preferences captured by the two-tower model in Stage 2 — filtering them
    out too aggressively would leave too few candidates for ranking.
    """
    checks = [
        _check_eligible_nationalities,
        _check_age,
        _check_degree_level,
        _check_high_school_track,
        _check_language_requirements,
        _check_academic_thresholds,
    ]
    return all(check(student, scholarship) for check in checks)


# ============================================================
# Inference Engine
# ============================================================

@dataclass
class Recommendation:
    rank: int
    scholarship_id: str
    scholarship_name: str
    relevance_score: float
    reason: Optional[str] = None


@dataclass
class InferenceResult:
    student_id: str
    student_name: Optional[str]
    recommendations: List[Recommendation]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "student_id": self.student_id,
            "student_name": self.student_name,
            "recommendations": [
                asdict(r) for r in self.recommendations
            ],
        }


class InferenceEngine:
    """Loads trained model and provides recommendation inference."""

    def __init__(self, model_dir: Optional[Path] = None):
        """Initialize inference engine with trained model and mappings."""
        if model_dir is None:
            model_dir = _MODEL_DIR

        self.model_path = model_dir / "best_model.keras"
        self.mappings_path = model_dir / "mappings.pkl"
        self._load()

    def _load(self):
        self._load_model()
        self._load_mappings()

    def _load_model(self):
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model not found at {self.model_path}. "
                "Run `python train.py` first."
            )
        custom_objects = {"CosineSimilarity": CosineSimilarity}
        self._model = keras.models.load_model(str(self.model_path), custom_objects=custom_objects)
        print(f"Loaded model from {self.model_path}")

    def _load_mappings(self):
        if not self.mappings_path.exists():
            raise FileNotFoundError(
                f"Mappings not found at {self.mappings_path}. "
                "Run `python train.py` first."
            )
        with open(self.mappings_path, "rb") as f:
            self._mappings = pickle.load(f)

    def recommend(
        self,
        student_id: str,
        student,
        scholarships: list,
        top_k: int = 5,
        use_hard_filter: bool = True,
    ) -> InferenceResult:
        """Run two-stage recommendation pipeline.

        Args:
            student_id: Unique student identifier.
            student: Student dataclass object.
            scholarships: List of Scholarship objects.
            top_k: Number of recommendations to return.
            use_hard_filter: If True, apply Stage 1 hard filter before model scoring.
                Set False to let the two-tower model handle all filtering via learned scores.

        Returns:
            InferenceResult with ranked recommendations.
        """
        # Stage 1: Hard filter (optional)
        if use_hard_filter:
            eligible = [
                s for s in scholarships
                if apply_hard_filter(student, s)
            ]
        else:
            eligible = list(scholarships)

        filtered_ids = {s.scholarship_id for s in eligible}

        # If no scholarships pass the hard filter (or empty input), return empty result
        if not eligible:
            return InferenceResult(
                student_id=student_id,
                student_name=getattr(student, "school_name", None),
                recommendations=[],
            )

        # Stage 2: Two-tower scoring
        student_features = _prepare_student_features(student, self._mappings["student"])
        sch_features = prepare_scholarship_features(eligible, self._mappings["scholarship"])

        # Repeat student features to match batch size of scholarships (broadcast inference)
        n_candidates = len(eligible)
        student_batch = np.repeat(student_features[None, :], n_candidates, axis=0)

        scores = self._model.predict(
            [student_batch, sch_features],
            verbose=0,
        ).flatten()

        # Sort by score descending
        ranked_indices = np.argsort(-scores)

        recommendations = []
        for rank, idx in enumerate(ranked_indices[:top_k], start=1):
            s = eligible[idx]
            recommendations.append(Recommendation(
                rank=rank,
                scholarship_id=s.scholarship_id,
                scholarship_name=s.name,
                relevance_score=float(round(scores[int(idx)], 4)),
                reason=None,  # Can be enhanced later with feature attribution
            ))

        return InferenceResult(
            student_id=student_id,
            student_name=getattr(student, "school_name", None),
            recommendations=recommendations,
        )


# ============================================================
# CLI Entry Point
# ============================================================

def _print_recommendations(result: InferenceResult):
    """Pretty-print recommendations to stdout."""
    print(f"\nRecommendations for {result.student_id}")
    if result.student_name:
        print(f"  School: {result.student_name}")
    print("-" * 70)

    for rec in result.recommendations:
        score_str = f"{rec.relevance_score:.4f}"
        print(f"  #{rec.rank}  {rec.scholarship_name:<35s}  Score: {score_str}")


def _load_students_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def _load_scholarships_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def main():
    parser = argparse.ArgumentParser(description="Run recommendation inference")
    parser.add_argument("--student-id", required=True, help="Student ID (e.g., STU_000001)")
    parser.add_argument("--top-k", type=int, default=5, help="Number of recommendations")
    parser.add_argument("--models-dir", type=str, default=None, help="Path to models directory")
    parser.add_argument(
        "--no-hard-filter",
        action="store_true",
        help="Disable Stage 1 hard filter — let the model handle all filtering",
    )
    args = parser.parse_args()

    engine = InferenceEngine(model_dir=Path(args.models_dir) if args.models_dir else None)

    # Load data
    datasets_dir = _SCRIPT_DIR / "datasets"
    students_df = pd.read_csv(datasets_dir / "students.csv")
    scholarships_df = pd.read_csv(datasets_dir / "scholarships.csv")

    # Find student by ID
    mask = students_df["student_id"] == args.student_id
    if not mask.any():
        print(f"Student {args.student_id} not found in dataset.")
        return

    stu_row = students_df[mask].iloc[0]

    # Reconstruct Student object (simplified — uses flat CSV data)
    from src.schemas import Student, LanguageProficiency
    lang_records = json.loads(stu_row["language_proficiency"]) if isinstance(stu_row.get("language_proficiency"), str) else []
    student = Student(
        student_id=stu_row["student_id"],
        nationality=stu_row["nationality"],
        age=int(stu_row["age"]),
        current_degree_level="high_school",
        target_degree_level="bachelors",
        high_school_track=stu_row["high_school_track"],
        school_name=stu_row["school_name"],
        overall_report_card_average=float(stu_row["overall_report_card_average"]),
        math_score=float(stu_row["math_score"]),
        english_score=float(stu_row["english_score"]),
        major_subject_average=float(stu_row["major_subject_average"]),
        language_proficiency=[LanguageProficiency(**lp) for lp in lang_records],
        olympiad_level=stu_row.get("olympiad_level", "none"),
        olympiad_subjects=json.loads(stu_row.get("olympiad_subjects", "[]")) if isinstance(stu_row.get("olympiad_subjects"), str) else stu_row.get("olympiad_subjects", []),
        leadership_experience_count=int(stu_row.get("leadership_experience_count", 0)),
        volunteer_experience_count=int(stu_row.get("volunteer_experience_count", 0)),
        competition_wins_count=int(stu_row.get("competition_wins_count", 0)),
        school_tier=stu_row.get("school_tier", "unknown"),
        family_income_category=stu_row.get("family_income_category", "middle"),
        from_underrepresented_region=bool(stu_row.get("from_underrepresented_region", False)),
        intended_career_track=stu_row.get("intended_career_track", ""),
        willing_to_return_home=bool(stu_row.get("willing_to_return_home", True)),
        target_countries=json.loads(stu_row.get("target_countries", "[]")) if isinstance(stu_row.get("target_countries"), str) else stu_row.get("target_countries", []),
        personal_statement=stu_row.get("personal_statement", ""),
        achievements_narrative=stu_row.get("achievements_narrative", ""),
        future_goals=stu_row.get("future_goals", ""),
        needs_full_funding=bool(stu_row.get("needs_full_funding", False)),
        can_self_fund_living=bool(stu_row.get("can_self_fund_living", False)),
    )

    # Reconstruct Scholarship objects from CSV rows
    from src.schemas import Scholarship, LanguageRequirement, SelectionCriteria, FundingCoverage
    scholarships = []
    for _, row in scholarships_df.iterrows():
        eligible_nat = json.loads(row["eligible_nationalities"]) if isinstance(row.get("eligible_nationalities"), str) else []
        eligible_deg = json.loads(row["eligible_degree_levels"]) if isinstance(row.get("eligible_degree_levels"), str) else []
        eligible_tracks = json.loads(row["eligible_high_school_tracks"]) if isinstance(row.get("eligible_high_school_tracks"), str) else []
        eligible_fields = json.loads(row["eligible_fields"]) if isinstance(row.get("eligible_fields"), str) else []
        lang_reqs_data = json.loads(row["language_requirements"]) if isinstance(row.get("language_requirements"), str) else []
        sel_crit_data = json.loads(row["selection_criteria"]) if isinstance(row.get("selection_criteria"), str) else {}

        scholarships.append(Scholarship(
            scholarship_id=row["scholarship_id"],
            name=row["name"],
            eligible_nationalities=eligible_nat,
            min_age=int(row["min_age"]),
            max_age=int(row["max_age"]),
            eligible_degree_levels=eligible_deg,
            eligible_high_school_tracks=eligible_tracks,
            eligible_fields=eligible_fields,
            preferred_school_tier=row.get("preferred_school_tier", "unknown"),
            min_report_card_average=float(row["min_report_card_average"]),
            min_major_subject_average=float(row["min_major_subject_average"]),
            language_requirements=[LanguageRequirement(**lr) for lr in lang_reqs_data],
            requires_financial_need=bool(row.get("requires_financial_need", False)),
            max_family_income_category=row.get("max_family_income_category", "high"),
            host_country=row.get("host_country", ""),
            host_region=row.get("host_region", ""),
            selection_criteria=SelectionCriteria(**sel_crit_data) if sel_crit_data else SelectionCriteria(),
            funding_coverage=FundingCoverage(
                covers_tuition=bool(row.get("funding_covers_tuition", False)),
                covers_living_expense=bool(row.get("funding_covers_living", False)),
                covers_airfare=bool(row.get("funding_covers_airfare", False)),
                covers_insurance=bool(row.get("funding_covers_insurance", False)),
                monthly_stipend=float(row.get("funding_monthly_stipend", 0.0)),
            ),
            career_track_preference=row.get("career_track_preference"),
            requires_return_home_country=bool(row.get("requires_return_home_country", False)),
            mission_statement=row.get("mission_statement", ""),
            target_recipient_profile=row.get("target_recipient_profile", ""),
        ))

    # Run inference
    result = engine.recommend(
        args.student_id, student, scholarships,
        top_k=args.top_k,
        use_hard_filter=not args.no_hard_filter,
    )

    _print_recommendations(result)


if __name__ == "__main__":
    main()
