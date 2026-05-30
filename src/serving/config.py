"""Serving configuration and shared constants."""


def _print(msg: str) -> None:
    """Print a message to stdout (flushed for live terminal visibility)."""
    print(msg, flush=True)


# ── Student/Scholarship JSON column schemas (single source of truth) ─────

STUDENT_JSON_COLS = ["language_proficiency", "olympiad_subjects", "target_countries"]
SCHOLARSHIP_JSON_COLS = [
    "eligible_nationalities", "eligible_degree_levels",
    "eligible_high_school_tracks", "eligible_fields",
    "preferred_school_tier", "language_requirements",
    "selection_criteria", "career_track_preference",
]


class ServingConfig:
    """Configuration for the serving layer loaded from configs/serving.yaml."""

    def __init__(self, config_path: str = "configs/serving.yaml"):
        import yaml
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        self.student_tower_path: str = cfg["models"]["student_tower"]
        self.scholarship_tower_path: str = cfg["models"]["scholarship_tower"]

        self.server_host: str = cfg["server"]["host"]
        self.server_port: int = cfg["server"]["port"]
        self.cors_origins: str = cfg["server"]["cors_origins"]
        self.auth_required: bool = cfg["server"].get("auth_required", False)
        self.auth_token: str = cfg["server"].get("auth_token", "")

        # Retraining splits (loaded from retraining section in serving.yaml)
        rt_cfg = cfg.get("retraining", {})
        self.retrain_train_split: float = rt_cfg.get("train_split", 0.70)
        self.retrain_val_split: float = rt_cfg.get("val_split", 0.15)
