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
    """Configuration for the serving layer loaded from configs/default.yaml."""

    def __init__(self, config_path: str = "configs/default.yaml"):
        import yaml
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        self.student_tower_path: str = cfg["models"]["student_tower"]
        self.scholarship_tower_path: str = cfg["models"]["scholarship_tower"]

        # Allow env var override (e.g., HF Spaces sets SERVER_PORT=7860)
        import os
        server_port_env = os.environ.get("SERVER_PORT")
        if server_port_env is not None:
            self.server_host: str = "0.0.0.0"
            self.server_port: int = int(server_port_env)
        else:
            self.server_host: str = cfg["server"]["host"]
            self.server_port: int = cfg["server"]["port"]
        self.cors_origins: str = cfg["server"]["cors_origins"]
        self.auth_required: bool = cfg["server"].get("auth_required", False)
        self.auth_token: str = cfg["server"].get("auth_token", "")

        # Retraining config (loaded from top-level retraining section in default.yaml)
        rt_cfg = cfg.get("retraining", {})
        self.retrain_holdout_fraction: float = rt_cfg.get("holdout_fraction", 0.0)
