"""Recommendation System — Inference Engine (v1)

Two-stage pipeline:
  Stage 1: Hard filter (deterministic eligibility checks)
  Stage 2: Two-tower neural network scoring

Usage as CLI:
    python inference.py --input student.json --top-k 5
    python inference.py --student-id STU_000001 --top-k 5

Usage as library:
    from v1.inference import InferenceEngine
    engine = InferenceEngine()
    recommendations = engine.recommend(student, scholarships)

Schema-based portability:
    The engine loads schema.json (saved by train.py) which defines column names,
    types, and list vector structure. No hardcoded constants needed — inference
    adapts automatically to whatever schema the model was trained with.
"""

import argparse
import json
import pickle
import sys
from dataclasses import asdict, dataclass
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
# Schema-based Feature Encoding
# ============================================================

def _load_schema(model_dir: Path) -> dict:
    """Load schema.json from model directory."""
    schema_path = model_dir / "schema.json"
    if not schema_path.exists():
        raise FileNotFoundError(
            f"Schema not found at {schema_path}. "
            "Run `python train.py` first."
        )
    with open(schema_path) as f:
        return json.load(f)


def _encode_categorical_from_schema(value, mapping: dict) -> int:
    """Encode a categorical value using schema mapping."""
    val = str(value).strip() if value else "unknown"
    return mapping.get(val, 0)


def _parse_language_proficiency(json_str: Optional[str], language_tests: Optional[list] = None) -> np.ndarray:
    """Parse language proficiency JSON into fixed-dim vector."""
    vec = np.zeros(12, dtype=np.float32)
    if language_tests is None:
        # Fallback for backwards compatibility
        language_tests = ["toefl", "ielts", "topik", "jlpt", "delf", "hsk"]
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
        if test_type in language_tests:
            idx = language_tests.index(test_type)
            score = float(record.get("score", 0.0))
            vec[idx * 2] = max(vec[idx * 2], score)
            vec[idx * 2 + 1] = 1.0
    return vec


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


def prepare_student_features(student_dict: dict, student_schema: dict) -> np.ndarray:
    """Convert a student dict to packed feature vector using schema.

    Args:
        student_dict: Dict with student fields matching schema keys.
        student_schema: Schema from schema.json for students.

    Returns:
        Packed numpy array of shape (feature_dim,)
    """
    parts = []

    # Categorical fields — uses encoding mapping from training
    for col in student_schema["categorical"]:
        val = str(student_dict.get(col, "") or "unknown")
        encoded = _encode_categorical_from_schema(val, student_schema["_mappings"].get(col, {}))
        parts.append(np.array([encoded], dtype=np.int32))

    # Numerical fields
    for col in student_schema["numerical"]:
        val = float(student_dict.get(col, 0.0) or 0.0)
        parts.append(np.array([val], dtype=np.float32))

    # Boolean fields (defaults to False if missing)
    for col in student_schema["boolean"]:
        val = bool(student_dict.get(col, False))
        parts.append(np.array([1.0 if val else 0.0], dtype=np.float32))

    # Language proficiency vector (always 12-dim)
    lang_field = student_dict.get("language_proficiency", [])
    lang_tests = student_schema.get("language_tests")
    if isinstance(lang_field, str):
        lang_vec = _parse_language_proficiency(lang_field, language_tests=lang_tests)
    elif isinstance(lang_field, list):
        lang_vec = _parse_language_proficiency(json.dumps(lang_field), language_tests=lang_tests)
    else:
        lang_vec = np.zeros(12, dtype=np.float32)
    parts.append(lang_vec)

    return np.hstack(parts).astype(np.float32)


def prepare_scholarship_features(scholarships_list: list, sch_schema: dict) -> np.ndarray:
    """Convert scholarship dicts to packed feature matrix using schema.

    Args:
        scholarships_list: List of dicts with scholarship fields.
        sch_schema: Schema from schema.json for scholarships.

    Returns:
        Packed numpy array of shape (n_scholarships, feature_dim)
    """
    parts = []

    # Categorical fields
    for col in sch_schema["categorical"]:
        values = [str(s.get(col, "")) for s in scholarships_list]
        mapping = sch_schema["_mappings"].get(col, {})
        encoded = np.array([mapping.get(v, 0) for v in values], dtype=np.int32)
        parts.append(encoded[:, None])

    # Numerical fields
    for col in sch_schema["numerical"]:
        vals = [float(s.get(col, 0.0) or 0.0) for s in scholarships_list]
        parts.append(np.array(vals, dtype=np.float32)[:, None])

    # Boolean fields (defaults to False if missing)
    for col in sch_schema["boolean"]:
        vals = [bool(s.get(col, False)) for s in scholarships_list]
        parts.append(np.array(vals, dtype=np.float32)[:, None])

    # List vector: countries + tracks + fields
    all_values = sch_schema["all_list_values"]
    list_vec = np.zeros((len(scholarships_list), sch_schema["list_vector_dim"]), dtype=np.float32)

    for i, s in enumerate(scholarships_list):
        # Eligible nationalities → country positions
        eligible_nat = s.get("eligible_nationalities", [])
        if isinstance(eligible_nat, str):
            try:
                eligible_nat = json.loads(eligible_nat)
            except (json.JSONDecodeError, TypeError):
                eligible_nat = []
        nat_vec = _encode_list_field(json.dumps(eligible_nat), all_values)
        list_vec[i, :sch_schema["list_country_dim"]] = nat_vec[:sch_schema["list_country_dim"]]

        # Eligible tracks → track positions
        eligible_tracks = s.get("eligible_high_school_tracks", [])
        if isinstance(eligible_tracks, str):
            try:
                eligible_tracks = json.loads(eligible_tracks)
            except (json.JSONDecodeError, TypeError):
                eligible_tracks = []
        track_vec = _encode_list_field(json.dumps(eligible_tracks), all_values)
        list_vec[i, sch_schema["list_country_dim"]:sch_schema["list_country_dim"] + sch_schema["list_track_dim"]] += \
            track_vec[:sch_schema["list_track_dim"]]

        # Eligible fields → field positions
        eligible_fields = s.get("eligible_fields", [])
        if isinstance(eligible_fields, str):
            try:
                eligible_fields = json.loads(eligible_fields)
            except (json.JSONDecodeError, TypeError):
                eligible_fields = []
        field_vec = _encode_list_field(json.dumps(eligible_fields), all_values)
        list_vec[i, sch_schema["list_country_dim"] + sch_schema["list_track_dim"]:] += \
            field_vec[:sch_schema["list_field_dim"]]

    parts.append(list_vec)
    return np.hstack(parts).astype(np.float32)


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


def _check_academic_thresholds(student, scholarship) -> bool:
    """Minimum report card and major subject averages."""
    min_rc = getattr(scholarship, "min_report_card_average", 0)
    min_major = getattr(scholarship, "min_major_subject_average", 0)
    return (float(getattr(student, "overall_report_card_average", 0)) >= min_rc and
            float(getattr(student, "major_subject_average", 0)) >= min_major)


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
    """Loads trained model and schema, provides recommendation inference."""

    def __init__(self, model_dir: Optional[Path] = None):
        """Initialize inference engine with trained model and schema.

        Args:
            model_dir: Path to directory containing best_model.keras + schema.json.
                       Defaults to v1/models/
        """
        if model_dir is None:
            model_dir = _SCRIPT_DIR / "models"

        self.model_path = model_dir / "best_model.keras"
        self.schema = _load_schema(model_dir)
        self._mappings_path = model_dir / "mappings.pkl"
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
        if not self._mappings_path.exists():
            raise FileNotFoundError(
                f"Mappings not found at {self._mappings_path}. "
                "Run `python train.py` first."
            )
        with open(self._mappings_path, "rb") as f:
            mappings = pickle.load(f)

        # Attach mappings to schema for feature encoding
        self.schema["student"]["_mappings"] = mappings["student"]
        self.schema["scholarship"]["_mappings"] = mappings["scholarship"]

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
            student: Student dataclass object or dict with matching fields.
            scholarships: List of Scholarship objects or dicts.
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

        # Convert Student/scholarship objects to dicts for feature encoding
        from dataclasses import asdict
        student_dict = asdict(student) if hasattr(student, '__dataclass_fields__') else student
        eligible_dicts = [asdict(s) if hasattr(s, '__dataclass_fields__') else s for s in eligible]

        # Stage 2: Two-tower scoring
        student_features = prepare_student_features(student_dict, self.schema["student"])
        sch_features = prepare_scholarship_features(eligible_dicts, self.schema["scholarship"])

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
    parser.add_argument("--student-id", type=str, help="Student ID (e.g., STU_000001)")
    parser.add_argument("--input", "-i", type=str, help="JSON file with student profile")
    parser.add_argument("--top-k", type=int, default=5, help="Number of recommendations")
    parser.add_argument("--models-dir", type=str, default=None, help="Path to models directory")
    parser.add_argument(
        "--no-hard-filter",
        action="store_true",
        help="Disable Stage 1 hard filter — let the model handle all filtering",
    )
    args = parser.parse_args()

    if not args.student_id and not args.input:
        print("Error: provide --student-id or --input")
        return

    engine = InferenceEngine(model_dir=Path(args.models_dir) if args.models_dir else None)

    # Load data
    datasets_dir = _SCRIPT_DIR / "datasets"
    students_df = pd.read_csv(datasets_dir / "students.csv")
    scholarships_df = pd.read_csv(datasets_dir / "scholarships.csv")

    # Find student by ID or load from JSON file
    if args.input:
        with open(args.input) as f:
            stu_dict = json.load(f)
        student_id = stu_dict.get("student_id", "unknown")
    else:
        mask = students_df["student_id"] == args.student_id
        if not mask.any():
            print(f"Student {args.student_id} not found in dataset.")
            return
        stu_dict = students_df[mask].iloc[0].to_dict()
        student_id = args.student_id

    # Reconstruct Student object (simplified — uses flat CSV/JSON data)
    from src.schemas import Student, LanguageProficiency
    lang_records = json.loads(stu_dict["language_proficiency"]) if isinstance(stu_dict.get("language_proficiency"), str) else []
    student = Student(
        student_id=stu_dict["student_id"],
        nationality=stu_dict["nationality"],
        age=int(stu_dict["age"]),
        current_degree_level="high_school",
        target_degree_level="bachelors",
        high_school_track=stu_dict["high_school_track"],
        school_name=stu_dict.get("school_name", ""),
        overall_report_card_average=float(stu_dict["overall_report_card_average"]),
        math_score=float(stu_dict["math_score"]),
        english_score=float(stu_dict["english_score"]),
        major_subject_average=float(stu_dict["major_subject_average"]),
        language_proficiency=[LanguageProficiency(**lp) for lp in lang_records],
        olympiad_level=stu_dict.get("olympiad_level", "none"),
        olympiad_subjects=json.loads(stu_dict.get("olympiad_subjects", "[]")) if isinstance(stu_dict.get("olympiad_subjects"), str) else stu_dict.get("olympiad_subjects", []),
        leadership_experience_count=int(stu_dict.get("leadership_experience_count", 0)),
        volunteer_experience_count=int(stu_dict.get("volunteer_experience_count", 0)),
        competition_wins_count=int(stu_dict.get("competition_wins_count", 0)),
        school_tier=stu_dict.get("school_tier", "unknown"),
        family_income_category=stu_dict.get("family_income_category", "middle"),
        from_underrepresented_region=bool(stu_dict.get("from_underrepresented_region", False)),
        intended_career_track=stu_dict.get("intended_career_track", ""),
        willing_to_return_home=bool(stu_dict.get("willing_to_return_home", True)),
        target_countries=json.loads(stu_dict.get("target_countries", "[]")) if isinstance(stu_dict.get("target_countries"), str) else stu_dict.get("target_countries", []),
        personal_statement=stu_dict.get("personal_statement", ""),
        achievements_narrative=stu_dict.get("achievements_narrative", ""),
        future_goals=stu_dict.get("future_goals", ""),
        needs_full_funding=bool(stu_dict.get("needs_full_funding", False)),
        can_self_fund_living=bool(stu_dict.get("can_self_fund_living", False)),
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
        student_id, student, scholarships,
        top_k=args.top_k,
        use_hard_filter=not args.no_hard_filter,
    )

    _print_recommendations(result)


if __name__ == "__main__":
    main()
