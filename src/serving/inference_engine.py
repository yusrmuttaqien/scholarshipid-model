"""InferenceEngine: load models, encode students, retrieve top-K scholarships.

Design:
- Student embeddings are computed on-the-fly per request
- Scholarship embeddings are cached and refreshed on demand
- Retraining runs asynchronously in a background thread

Imports from sibling modules:
- config: _print, ServingConfig, STUDENT_JSON_COLS, SCHOLARSHIP_JSON_COLS
- helpers: student_profile_to_csv_schema, _parse_csv_with_json, etc.
"""
import os
import sys
import threading
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd
import tensorflow as tf
import yaml

from src.models.student_tower import L2Normalize
from src.models.scholarship_tower import build_scholarship_tower
from src.models.student_tower import build_student_tower
from src.utils.feature_engineering import (
    encode_scholarship,
    encode_student,
    encode_text,
    get_sbert_model,
)
from src.trainers.training_loop import run_training

from .config import _print, ServingConfig, STUDENT_JSON_COLS, SCHOLARSHIP_JSON_COLS
from .helpers import (
    student_profile_to_csv_schema,
    _parse_csv_with_json,
    _build_scholarship_metadata,
    _scholarship_to_metadata,
)


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
        student_tower_path: Optional[str],
        scholarship_tower_path: Optional[str],
        config_path: str = "configs/default.yaml",
        serving_config_path: str = "configs/serving.yaml",
    ):
        self.serving_config = ServingConfig(serving_config_path)
        self.config_path = config_path

        # Resolve default paths from ServingConfig when not provided via CLI
        self.student_tower_path = student_tower_path or self.serving_config.student_tower_path
        self.scholarship_tower_path = scholarship_tower_path or self.serving_config.scholarship_tower_path

        # Loaded from config
        self.cfg: Optional[dict] = None

        # Towers (loaded on initialize)
        self.student_tower: Optional[tf.keras.Model] = None
        self.scholarship_tower: Optional[tf.keras.Model] = None

        # Scholarship cache - pre-computed embeddings for fast retrieval
        self._sch_emb: Optional[np.ndarray] = None  # (N, 128) L2-normalized
        self._sch_ids: list = []
        self._sch_metadata: list = []  # Raw scholarship dicts for API responses

        # ── Retracking state ────────────────────────────────────────────────
        self._retraining_lock = threading.Lock()
        self._retraining_status: str = "idle"  # idle | training | done | error
        self._retraining_started_at: Optional[datetime] = None
        self._retraining_finished_at: Optional[datetime] = None
        self._retraining_error: Optional[str] = None

    def _load_config(self) -> dict:
        with open(self.config_path) as f:
            return yaml.safe_load(f)

    # ── Retraining helpers ────────────────────────────────────────────────

    def _merge_csvs(
        self, existing_df: pd.DataFrame, new_df: pd.DataFrame, id_col: str
    ) -> pd.DataFrame:
        """Merge new rows with existing data by ID. New overwrites old."""
        if new_df is None or len(new_df) == 0:
            return existing_df

        # Merge and deduplicate by ID (new entries win)
        merged = pd.concat([existing_df, new_df]).drop_duplicates(
            subset=[id_col], keep="last"
        )
        _print(f"  Merged {len(new_df)} new rows → {len(merged)} total (by {id_col})")
        return merged

    def _precompute_text_embeddings(
        self, students_df: pd.DataFrame, scholarships_df: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray]:
        """Precompute text embeddings for students and scholarships.

        Returns:
            stu_text_emb: (N_stu, 384), sch_text_emb: (N_sch, 384)
        """
        _print("  Precomputing student text embeddings...")
        stu_texts = (
            students_df["personal_statement"].fillna("") + " " +
            students_df["achievements_narrative"].fillna("") + " " +
            students_df["future_goals"].fillna("")
        ).tolist()
        stu_text_emb = encode_text(stu_texts)

        _print("  Precomputing scholarship text embeddings...")
        sch_texts = (
            scholarships_df["mission_statement"].fillna("") + " " +
            scholarships_df["target_recipient_profile"].fillna("")
        ).tolist()
        sch_text_emb = encode_text(sch_texts)

        return stu_text_emb, sch_text_emb

    def _train_model(
        self,
        stu_struct: np.ndarray,
        sch_struct: np.ndarray,
        stu_text_emb: np.ndarray,
        sch_text_emb: np.ndarray,
        feedback_df: pd.DataFrame,
        cfg: dict,
        stu_indices: np.ndarray,
        sch_indices: np.ndarray,
        # ── For evaluation metrics (Recall/NDCG/MRR) ─────────────
        stu_id_to_idx: Optional[dict] = None,
        sch_ids: Optional[list] = None,
    ) -> dict:
        """Train the model on merged data using existing weights (finetuning).

        By default uses ALL feedback for training (no chronological split),
        ensuring new/delta entries uploaded via /retrain are seen by the optimizer.
        An optional holdout_fraction can reserve recent data for monitoring only.

        Returns metrics dict with final training results.
        """
        feedback_weights = cfg["feedback_weights"]
        epochs = cfg["training"]["epochs"]
        temp = cfg["model"]["temperature"]
        tb_cfg = cfg.get("tensorboard", {})

        # Build full feature matrices for ALL students and scholarships
        stu_feat_all = np.concatenate([stu_struct, stu_text_emb], axis=1)  # (N_stu, 506)
        sch_feat_all = np.concatenate([sch_struct, sch_text_emb], axis=1)  # (N_sch, 509)

        # Sort feedback by timestamp for consistent ordering
        feedback_df = feedback_df.sort_values("timestamp").reset_index(drop=True)
        n_total = len(feedback_df)

        # ── Optional holdout split ────────────────────────────────────────
        holdout_frac = self.serving_config.retrain_holdout_fraction
        if holdout_frac > 0:
            n_holdout = int(holdout_frac * n_total)
            train_df = feedback_df.iloc[:n_total - n_holdout]
            val_df = feedback_df.iloc[n_total - n_holdout:]
            _print(f"  Feedback split — train:{len(train_df)}  holdout:{len(val_df)}")
        else:
            train_df = feedback_df
            val_df = feedback_df
            _print(f"  Using all {n_total} feedback samples for training (no holdout)")

        # Build per-sample features from feedback rows
        weights_all = np.array(
            [feedback_weights[ft] for ft in feedback_df["feedback_type"]], dtype=np.float32
        )

        stu_feat_all_fb = stu_feat_all[stu_indices]  # (M, 506) where M = len(feedback_df)
        sch_feat_all_fb = sch_feat_all[sch_indices]  # (M, 509)

        if holdout_frac > 0:
            n_train = len(train_df)
            train_weights = weights_all[:n_train]
            val_weights = weights_all[n_train:]
            train_stu_feat = stu_feat_all_fb[:n_train]
            val_stu_feat = stu_feat_all_fb[n_train:]
            train_sch_feat = sch_feat_all_fb[:n_train]
            val_sch_feat = sch_feat_all_fb[n_train:]
        else:
            # No holdout — use all data for both training and evaluation
            train_weights = weights_all
            val_weights = weights_all
            train_stu_feat = stu_feat_all_fb
            val_stu_feat = stu_feat_all_fb
            train_sch_feat = sch_feat_all_fb
            val_sch_feat = sch_feat_all_fb

        # ── Build optimizer (reuse loaded towers) ────────────────────────
        student_tower = self.student_tower
        scholarship_tower = self.scholarship_tower
        # Use lower LR for fine-tuning so we don't destroy pre-trained weights
        finetune_lr = cfg["training"]["learning_rate"] / 10
        optimizer = tf.keras.optimizers.Adam(learning_rate=finetune_lr)

        # ── Train with validation + TensorBoard ──────────────────────────
        _print(f"  Training for {epochs} epochs (finetuning from existing weights)...")

        metrics = run_training(
            student_tower=student_tower,
            scholarship_tower=scholarship_tower,
            train_stu_feat=train_stu_feat,
            train_sch_feat=train_sch_feat,
            train_weights=train_weights,
            val_stu_feat=val_stu_feat,
            val_sch_feat=val_sch_feat,
            val_weights=val_weights,
            optimizer=optimizer,
            temperature=temp,
            epochs=epochs,
            checkpoint_dir=cfg["output"]["checkpoint_dir"],
            log_dir=cfg["output"]["log_dir"],
            tb_enabled=tb_cfg.get("enabled", False),
            k=cfg["evaluation"]["k_values"][0],
            # ── Evaluation data (for Recall/NDCG/MRR) ───────────────
            val_df=val_df,
            stu_struct=stu_struct,
            sch_struct=sch_struct,
            stu_text_emb=stu_text_emb,
            sch_text_emb=sch_text_emb,
            stu_id_to_idx=stu_id_to_idx,
            sch_ids=sch_ids,
        )

        _print(f"  Saved updated weights to {cfg['output']['checkpoint_dir']}")
        return metrics

    def retrain_from_csvs(
        self,
        students_csv_text: Optional[str] = None,
        scholarships_csv_text: Optional[str] = None,
        feedbacks_csv_text: Optional[str] = None,
    ) -> dict:
        """Full training pipeline: merge data → precompute embeddings → train → export.

        This method runs in a background thread and updates _retraining_status.
        """
        with self._retraining_lock:
            if self._retraining_status == "training":
                return {"error": "Retraining already in progress"}

            self._retraining_status = "training"
            self._retraining_started_at = datetime.now(timezone.utc)
            self._retraining_finished_at = None
            self._retraining_error = None

        try:
            _print("=== Retraining started ===")

            # ── 1. Load existing data from disk ────────────────────────────
            raw_path = self.cfg["data"]["raw_path"] if self.cfg else None
            if not raw_path:
                self.cfg = self._load_config()
                raw_path = self.cfg["data"]["raw_path"]

            students_df = pd.read_csv(f"{raw_path}/students.csv")
            scholarships_df = pd.read_csv(f"{raw_path}/scholarships.csv")
            feedbacks_df = pd.read_csv(f"{raw_path}/feedback.csv")

            # ── Normalize existing data to ensure consistent JSON types ────
            students_df = self._normalize_json_columns(students_df, STUDENT_JSON_COLS)
            scholarships_df = self._normalize_json_columns(scholarships_df, SCHOLARSHIP_JSON_COLS)

            # ── 2. Merge with new data from request ────────────────────────
            if students_csv_text:
                new_students = _parse_csv_with_json(
                    students_csv_text, STUDENT_JSON_COLS
                )
                students_df = self._merge_csvs(students_df, new_students, "student_id")

            if scholarships_csv_text:
                new_scholarships = _parse_csv_with_json(
                    scholarships_csv_text, SCHOLARSHIP_JSON_COLS
                )
                scholarships_df = self._merge_csvs(
                    scholarships_df, new_scholarships, "scholarship_id"
                )

            if feedbacks_csv_text:
                new_feedbacks = _parse_csv_with_json(
                    feedbacks_csv_text, []  # feedbacks don't have JSON columns in standard format
                )
                id_col = "feedback_id" if "feedback_id" in new_feedbacks.columns else "timestamp"
                feedbacks_df = self._merge_csvs(feedbacks_df, new_feedbacks, id_col)

            _print(
                f"  Data: {len(students_df)} students, {len(scholarships_df)} scholarships, {len(feedbacks_df)} feedbacks"
            )

            # ── 3. Precompute text embeddings ─────────────────────────────
            stu_text_emb, sch_text_emb = self._precompute_text_embeddings(
                students_df, scholarships_df
            )

            # ── 4. Build structured features for all data ──────────────────
            _print("  Building structured features...")
            stu_struct = np.array(
                [encode_student(r) for _, r in students_df.iterrows()], dtype=np.float32
            )
            sch_struct = np.array(
                [encode_scholarship(r) for _, r in scholarships_df.iterrows()], dtype=np.float32
            )

            # ── 5. Save merged data back to disk ───────────────────────────
            pd.DataFrame(students_df).to_csv(f"{raw_path}/students.csv", index=False)
            pd.DataFrame(scholarships_df).to_csv(f"{raw_path}/scholarships.csv", index=False)
            pd.DataFrame(feedbacks_df).to_csv(f"{raw_path}/feedback.csv", index=False)

            # ── 6. Train model (finetuning from existing weights) ──────────
            cfg = self.cfg if self.cfg else self._load_config()

            # Build indices mapping each feedback row to its position in the full arrays
            stu_id_map = {sid: i for i, sid in enumerate(students_df["student_id"])}
            sch_id_map = {sid: i for i, sid in enumerate(scholarships_df["scholarship_id"])}

            stu_indices = np.array(
                [stu_id_map[sid] for sid in feedbacks_df["student_id"]], dtype=np.int32
            )
            sch_indices = np.array(
                [sch_id_map[sid] for sid in feedbacks_df["scholarship_id"]], dtype=np.int32
            )

            # Build ID mappings for evaluation metrics (Recall/NDCG/MRR)
            stu_id_to_idx = {sid: i for i, sid in enumerate(students_df["student_id"])}
            sch_ids = scholarships_df["scholarship_id"].tolist()

            self._train_model(
                stu_struct, sch_struct, stu_text_emb, sch_text_emb,
                feedbacks_df, cfg, stu_indices, sch_indices,
                stu_id_to_idx=stu_id_to_idx,
                sch_ids=sch_ids,
            )

            # ── 7. Refresh in-memory cache + export to disk ────────────────
            self._refresh_from_df(scholarships_df)
            output_dir = self.cfg["output"]["embedding_dir"]
            os.makedirs(output_dir, exist_ok=True)

            sch_struct_list = []
            for _, row in scholarships_df.iterrows():
                sch_dict = row.to_dict()
                sch_struct_list.append(encode_scholarship(sch_dict))
            sch_struct = np.array(sch_struct_list, dtype=np.float32)

            sch_text_list = []
            for _, row in scholarships_df.iterrows():
                sch_dict = row.to_dict()
                text_parts = []
                for field in ["mission_statement", "target_recipient_profile"]:
                    val = sch_dict.get(field, "")
                    if val:
                        text_parts.append(str(val))
                sch_text_list.append(" ".join(text_parts) if text_parts else "")

            sch_text_emb_all = encode_text(sch_text_list)
            sch_feat = np.concatenate([sch_struct, sch_text_emb_all], axis=1)
            sch_emb = self.scholarship_tower(sch_feat, training=False).numpy()

            sch_ids = scholarships_df["scholarship_id"].tolist()
            np.save(os.path.join(output_dir, "scholarship_emb.npy"), sch_emb)
            np.save(os.path.join(output_dir, "scholarship_ids.npy"), np.array(sch_ids, dtype=object))
            _print(f"  Saved scholarship embeddings to {output_dir}/")

            # ── 8. Update status → done ────────────────────────────────────
            with self._retraining_lock:
                self._retraining_status = "done"
                self._retraining_finished_at = datetime.now(timezone.utc)

            _print("=== Retraining completed ===\n")
            return {"status": "done"}

        except Exception as e:
            with self._retraining_lock:
                self._retraining_status = "error"
                self._retraining_error = str(e)
            print(f"\n=== Retraining FAILED: {e} ===\n", file=sys.stderr, flush=True)
            return {"status": "error", "error": str(e)}

    # ── Public API ────────────────────────────────────────────────────────

    def recommend(self, student_data: dict, k: int = 5) -> list[dict]:
        """Return top-K scholarships for a student profile.

        Args:
            student_data: Student profile dict matching CSV schema
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
        """Rebuild scholarship embedding cache from local CSV file.

        Reads the CSV file from the configured data path, parses JSON columns,
        encodes structured + text features through SBERT and the scholarship tower,
        then caches embeddings.

        This method is used for local development where data is read from files.
        """
        if self.cfg is None:
            self.cfg = self._load_config()

        # Load scholarships directly from CSV file
        raw_path = self.cfg["data"]["raw_path"]
        scholarships_df = pd.read_csv(f"{raw_path}/scholarships.csv")

        # Parse JSON columns (language_requirements, selection_criteria)
        scholarships_df = self._normalize_json_columns(scholarships_df, SCHOLARSHIP_JSON_COLS)

        self._refresh_from_df(scholarships_df)

    def refresh_from_csv(self, csv_text: str):
        """Rebuild scholarship embedding cache from CSV text.

        Parses the CSV text (from an API payload), encodes structured + text
        features through SBERT and the scholarship tower, then caches embeddings.

        This method is used for production where data is pushed via the /refresh endpoint.

        Args:
            csv_text: Raw CSV text content to parse.
        """
        # Parse CSV from string using shared parser
        scholarships_df = _parse_csv_with_json(csv_text, SCHOLARSHIP_JSON_COLS)

        self._refresh_from_df(scholarships_df)

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

        # Encode text features via SBERT — same logic as refresh_scholarships
        texts = []
        for s in new_scholarships:
            text_parts = []
            for field in ["mission_statement", "target_recipient_profile"]:
                val = s.get(field)
                if val:
                    text_parts.append(str(val))
            texts.append(" ".join(text_parts))
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

    def _load_cached_embeddings(self) -> bool:
        """Try to load pre-computed scholarship embeddings from disk.

        Returns True if embeddings were loaded successfully, False otherwise.
        When successful, sets _sch_emb, _sch_ids, and _sch_metadata directly.
        """
        emb_dir = self.cfg["output"]["embedding_dir"]
        sch_emb_path = os.path.join(emb_dir, "scholarship_emb.npy")
        sch_ids_path = os.path.join(emb_dir, "scholarship_ids.npy")

        if not (os.path.exists(sch_emb_path) and os.path.exists(sch_ids_path)):
            _print("  No cached embeddings found — will recompute from CSVs.")
            return False

        try:
            self._sch_emb = np.load(sch_emb_path, allow_pickle=False)
            self._sch_ids = list(np.load(sch_ids_path, allow_pickle=True).tolist())
            self._sch_metadata = []  # Will be built lazily from CSV if needed

            # Build metadata from the saved IDs (minimal — just enough for API)
            self._sch_metadata = [{"scholarship_id": sid} for sid in self._sch_ids]

            _print(f"Loaded cached embeddings: {len(self._sch_emb)} scholarships "
                   f"(shape {self._sch_emb.shape})")
            return True
        except Exception as e:
            print(f"  Failed to load cached embeddings: {e}", file=sys.stderr, flush=True)
            return False

    def initialize(self):
        """Load both towers and build initial scholarship embedding cache.

        Strategy:
        1. Load model weights from disk (always required)
        2. Try loading pre-computed scholarship embeddings from disk
           (skips SBERT + tower inference — much faster cold start)
        3. Fall back to recomputing from CSVs if cached embeddings unavailable
        """
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

        # Try loading pre-computed embeddings (fast path — skips SBERT + tower)
        if self._load_cached_embeddings():
            print("Scholarship cache loaded from disk ✅")
            return  # Done — no need for SBERT warmup or recomputation

        # Fallback: recompute from CSVs (slower, but always works)
        get_sbert_model()
        print("SBERT model warmed up")
        self.refresh_scholarships()

    # ── Private helpers ───────────────────────────────────────────────────

    def _normalize_json_columns(self, df: pd.DataFrame, json_cols: list[str]) -> pd.DataFrame:
        """Normalize JSON columns in a DataFrame using the shared function."""
        from src.utils.feature_engineering import normalize_json_columns as _normalize_json_columns
        return _normalize_json_columns(df, json_cols)

    def get_retraining_status(self) -> dict:
        """Return current retraining status (for /health endpoint)."""
        with self._retraining_lock:
            return {
                "status": self._retraining_status,
                "started_at": self._retraining_started_at.isoformat() if self._retraining_started_at else None,
                "finished_at": self._retraining_finished_at.isoformat() if self._retraining_finished_at else None,
                "error": self._retraining_error,
            }

    def _refresh_from_df(self, scholarships_df: pd.DataFrame) -> None:
        """Build scholarship embedding cache from a DataFrame.

        Shared logic used by refresh_scholarships() and refresh_from_csv().
        """
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