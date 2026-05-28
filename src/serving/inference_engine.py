"""InferenceEngine: load models, encode students, retrieve top-K scholarships.

Design:
- Student embeddings are computed on-the-fly per request
- Scholarship embeddings are cached and refreshed on demand
- Reuses existing encoding logic from feature_engineering (single source of truth)
"""
from typing import Optional

import json
import numpy as np
import pandas as pd
import tensorflow as tf
import yaml

from src.models.student_tower import L2Normalize
from src.utils.feature_engineering import (
    encode_scholarship,
    encode_student,
    encode_text,
    get_sbert_model,
)


def student_profile_to_csv_schema(student_data: dict) -> dict:
    """Convert flat API student profile to CSV-compatible schema for encode_student.

    The API accepts flat fields (e.g. toefl_score=65), but encode_student expects
    CSV-compatible structures (e.g. language_proficiency=[{"test_type":"toefl","score":65}]).
    """
    row = dict(student_data)

    # Build language_proficiency as list of dicts (expected by encode_student)
    lang_prof = []
    if student_data.get("toefl_score", 0) > 0:
        lang_prof.append({"test_type": "toefl", "score": student_data["toefl_score"]})
    if student_data.get("ielts_score", 0) > 0:
        lang_prof.append({"test_type": "ielts", "score": student_data["ielts_score"]})
    row["language_proficiency"] = lang_prof

    # olympiad_subjects: convert comma-separated string to list if needed
    olympiad_subjects_raw = student_data.get("olympiad_subjects") or []
    if isinstance(olympiad_subjects_raw, str) and olympiad_subjects_raw:
        row["olympiad_subjects"] = [s.strip() for s in olympiad_subjects_raw.split(",")]
    else:
        row["olympiad_subjects"] = olympiad_subjects_raw

    # target_countries: convert comma-separated string to list if needed
    target_countries_raw = student_data.get("target_countries")
    if isinstance(target_countries_raw, str) and target_countries_raw:
        row["target_countries"] = [c.strip() for c in target_countries_raw.split(",")]
    else:
        row["target_countries"] = target_countries_raw

    return row


class ServingConfig:
    """Configuration for the serving layer loaded from configs/serving.yaml."""

    def __init__(self, config_path: str = "configs/serving.yaml"):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        self.environment: str = cfg.get("environment", "local")
        self.data_source: str = cfg.get("data_source", "csv")
        self.csv_path: str = cfg.get("csv_path", "data/raw/scholarships.csv")

        self.student_tower_path: str = cfg["models"]["student_tower"]
        self.scholarship_tower_path: str = cfg["models"]["scholarship_tower"]

        self.refresh_on_the_fly: bool = cfg["refresh"].get("on_the_fly", True)

        self.server_host: str = cfg["server"]["host"]
        self.server_port: int = cfg["server"]["port"]
        self.cors_origins: str = cfg["server"]["cors_origins"]
        self.auth_required: bool = cfg["server"].get("auth_required", False)

        self.precomputed_struct_path: str = cfg["precomputed"]["scholarship_struct"]
        self.precomputed_text_path: str = cfg["precomputed"]["scholarship_text"]

        self.log_level: str = cfg["logging"].get("level", "DEBUG")


class InferenceEngine:
    """Loads trained towers and retrieves top-K scholarships per student.

    Architecture:
        Student Tower:  student profile → student embedding (128-dim, L2-normalized)
        Scholarship cache: pre-computed scholarship embeddings (128-dim, L2-normalized)
        Matching: dot product (equivalent to cosine similarity for L2-normalized vectors)

    Usage:
        engine = InferenceEngine(
            student_tower_path="outputs/checkpoints/student_tower_best.keras",
            scholarship_tower_path="outputs/checkpoints/scholarship_tower_best.keras",
            config_path="configs/default.yaml",
        )
        engine.initialize()

        # Recommend top-5 scholarships for a student
        results = engine.recommend(student_data, k=5)

        # Refresh scholarship cache when new scholarships are added
        engine.refresh_scholarships()
    """

    def __init__(
        self,
        student_tower_path: str,
        scholarship_tower_path: str,
        config_path: str = "configs/default.yaml",
        serving_config_path: str = "configs/serving.yaml",
    ):
        self.student_tower_path = student_tower_path
        self.scholarship_tower_path = scholarship_tower_path
        self.config_path = config_path
        self.serving_config = ServingConfig(serving_config_path)

        # Loaded from config
        self.cfg: Optional[dict] = None

        # Towers (loaded on initialize)
        self.student_tower: Optional[tf.keras.Model] = None
        self.scholarship_tower: Optional[tf.keras.Model] = None

        # Scholarship cache - pre-computed embeddings for fast retrieval
        self._sch_emb: Optional[np.ndarray] = None  # (N, 128) L2-normalized
        self._sch_ids: list = []
        self._sch_metadata: list = []  # Raw scholarship dicts for API responses

    def _load_config(self) -> dict:
        with open(self.config_path) as f:
            return yaml.safe_load(f)

    def initialize(self):
        """Load both towers and build initial scholarship embedding cache."""
        self.cfg = self._load_config()

        custom = {"L2Normalize": L2Normalize}

        # Load student tower — used for encoding students at inference time
        self.student_tower = tf.keras.models.load_model(
            self.student_tower_path, custom_objects=custom
        )
        print(f"Loaded student tower from {self.student_tower_path}")

        # Load scholarship tower — used for encoding scholarships into the cache
        self.scholarship_tower = tf.keras.models.load_model(
            self.scholarship_tower_path, custom_objects=custom
        )
        print(f"Loaded scholarship tower from {self.scholarship_tower_path}")

        # Warm-up SBERT model (lazy-loaded singleton in feature_engineering)
        get_sbert_model()
        print("SBERT model warmed up")

        # Build initial scholarship cache — always recompute on-the-fly
        self.refresh_scholarships()

    # ── Public API ────────────────────────────────────────────────────────

    def recommend(
        self, student_data: dict, k: int = 5
    ) -> list[dict]:
        """Return top-K scholarships for a student profile.

        Args:
            student_data: Student profile dict matching CSV columns
                (nationality, age, high_school_track, etc.)
            k: Number of scholarships to return (default 5)

        Returns:
            List of dicts with scholarship_id, score, rank, and metadata.
        """
        if self.student_tower is None:
            raise RuntimeError("Call initialize() before using recommend()")

        # Convert flat API profile → CSV-compatible schema, then encode
        csv_row = student_profile_to_csv_schema(student_data)
        stu_struct = np.array([encode_student(csv_row)], dtype=np.float32)
        stu_text_raw = self._build_student_text(student_data)
        stu_text_emb = encode_text([stu_text_raw])

        stu_feat = np.concatenate([stu_struct, stu_text_emb], axis=1)

        # Forward through student tower
        stu_emb = self.student_tower(stu_feat, training=False).numpy()  # (1, 128)

        # Dot product vs all cached scholarship embeddings
        sch_scores = self._compute_scores(stu_emb[0])

        # Top-K retrieval
        top_k_idx = np.argsort(-sch_scores)[:k]

        results = []
        for rank, idx in enumerate(top_k_idx, start=1):
            results.append(
                {
                    "scholarship_id": self._sch_ids[idx],
                    "score": float(sch_scores[idx]),
                    "rank": rank,
                    "metadata": self._sch_metadata[idx],
                }
            )

        return results

    def refresh_scholarships(self):
        """Rebuild scholarship embedding cache from CSV on-the-fly.

        Reads the CSV directly, parses JSON columns, encodes structured + text
        features through SBERT and the scholarship tower, then caches embeddings.

        This method always recomputes — it does NOT depend on precomputed .npy files.
        """
        if self.cfg is None:
            self.cfg = self._load_config()

        # Load scholarships directly from CSV
        raw_path = self.cfg["data"]["raw_path"]
        scholarships_df = pd.read_csv(f"{raw_path}/scholarships.csv")

        # Parse JSON columns (language_requirements, selection_criteria)
        # Use json.loads instead of eval() to handle JSON booleans (true/false) properly
        for col in ["language_requirements", "selection_criteria"]:
            if col in scholarships_df.columns:
                def parse_json(val):
                    if isinstance(val, str) and val.strip().startswith(("[", "{")):
                        try:
                            return json.loads(val)
                        except (json.JSONDecodeError, ValueError):
                            return val
                    return val
                scholarships_df[col] = scholarships_df[col].apply(parse_json)

        # Encode each scholarship on-the-fly
        sch_struct_list = []
        sch_text_list = []

        for _, row in scholarships_df.iterrows():
            # Build dict from row
            sch_dict = row.to_dict()
            sch_struct = encode_scholarship(sch_dict)
            sch_struct_list.append(sch_struct)

            # Text embedding from mission_statement + target_recipient_profile
            text_parts = []
            for field in ["mission_statement", "target_recipient_profile"]:
                val = sch_dict.get(field, "")
                if val:
                    text_parts.append(str(val))
            sch_text_raw = " ".join(text_parts) if text_parts else ""
            sch_text_list.append(sch_text_raw)

        # Stack structured features
        sch_struct = np.array(sch_struct_list, dtype=np.float32)

        # Encode text features via SBERT
        sch_text_emb = encode_text(sch_text_list)

        # Concatenate structured + text features, run through scholarship tower
        sch_feat = np.concatenate([sch_struct, sch_text_emb], axis=1)
        self._sch_emb = self.scholarship_tower(
            sch_feat, training=False
        ).numpy()  # (N, 128) L2-normalized

        # Build metadata for API responses
        self._sch_ids = scholarships_df["scholarship_id"].tolist()
        self._sch_metadata = _build_scholarship_metadata(scholarships_df)

        print(f"Scholarship cache refreshed: {len(self._sch_ids)} scholarships "
              f"(embedding shape {self._sch_emb.shape})")

    def add_scholarships(self, new_scholarships: list[dict]) -> int:
        """Encode and cache new scholarships without a full refresh.

        Useful when a single scholarship is added and we want to avoid
        re-encoding the entire catalog.

        Args:
            new_scholarships: List of scholarship dicts matching CSV schema.

        Returns:
            Number of scholarships added.
        """
        if self._sch_emb is None:
            raise RuntimeError(
                "Cache not initialized. Call refresh_scholarships() first."
            )

        # Encode structured features
        new_struct = np.array(
            [encode_scholarship(s) for s in new_scholarships], dtype=np.float32
        )

        # Encode text features via SBERT
        texts = [
            (s.get("mission_statement", "") or "") + " " +
            (s.get("target_recipient_profile", "") or "")
            for s in new_scholarships
        ]
        new_text_emb = encode_text(texts)

        # Run through scholarship tower to get embeddings
        new_feat = np.concatenate([new_struct, new_text_emb], axis=1)
        new_emb = self.scholarship_tower(new_feat, training=False).numpy()

        # Append to cache
        self._sch_emb = np.concatenate([self._sch_emb, new_emb], axis=0)

        for s in new_scholarships:
            self._sch_ids.append(s["scholarship_id"])
            self._sch_metadata.append(_scholarship_to_metadata(s))

        print(f"Added {len(new_scholarships)} scholarships. Total: {len(self._sch_ids)}")
        return len(new_scholarships)

    # ── Private helpers ───────────────────────────────────────────────────

    def _build_student_text(self, student_data: dict) -> str:
        """Build text string for SBERT encoding from student narrative fields."""
        parts = []
        for field in ["personal_statement", "achievements_narrative", "future_goals"]:
            val = student_data.get(field, "")
            if val:
                parts.append(str(val))
        return " ".join(parts)

    def _compute_scores(self, stu_emb: np.ndarray) -> np.ndarray:
        """Dot-product student embedding against cached scholarship embeddings.

        Both sides are L2-normalized, so dot product == cosine similarity.
        """
        scores = self._sch_emb @ stu_emb
        return scores


def _build_scholarship_metadata(df) -> list[dict]:
    """Extract lightweight metadata from scholarship DataFrame for API responses."""
    metadata = []
    for _, row in df.iterrows():
        metadata.append({
            "scholarship_id": row.get("scholarship_id"),
            "host_country": row.get("host_country"),
            "host_region": row.get("host_region"),
            "funding_is_full_funding": bool(row.get("funding_is_full_funding", False)),
        })
    return metadata


def _scholarship_to_metadata(sch: dict) -> dict:
    """Convert scholarship dict to lightweight metadata for API responses."""
    return {
        "scholarship_id": sch.get("scholarship_id"),
        "host_country": sch.get("host_country"),
        "host_region": sch.get("host_region"),
        "funding_is_full_funding": bool(sch.get("funding_is_full_funding", False)),
    }