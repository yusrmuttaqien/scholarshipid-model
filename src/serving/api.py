"""FastAPI application for scholarship recommendation serving.

Endpoints:
    POST /recommend  — Recommend top-K scholarships for a student profile
    POST /refresh    — Rebuild scholarship embedding cache from CSV (admin)
    POST /retrain    — Retrain model with new data (async, admin)
    GET  /health     — Health check (includes retraining status)

After successful retraining, data + model artifacts are pushed to HuggingFace.
"""
from __future__ import annotations

import threading
from http import HTTPStatus
from typing import Optional

from fastapi import Depends, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

from scripts.hf_sync import push_data_artifacts, push_model_artifacts
from src.serving.inference_engine import InferenceEngine

# Bearer security scheme for auth-protected endpoints
security = HTTPBearer(auto_error=False)


def _get_auth_token(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)) -> Optional[str]:
    """Dependency to extract Bearer token from request headers."""
    return credentials.credentials if credentials else None


# ── Pydantic schemas ──────────────────────────────────────────────────────

class StudentProfile(BaseModel):
    """Student profile for scholarship recommendation.

    Required fields: nationality, age, high_school_track
    All other fields are optional with sensible defaults.
    """

    # ── Required fields ──────────────────────────────────────────────────
    nationality: str = Field(
        ...,
        description="Student's nationality (e.g. 'Indonesia', 'Malaysia')",
        examples=["Indonesia"],
    )
    age: int = Field(
        ...,
        description="Student's age in years",
        examples=[17],
        ge=15,
        le=20,
    )
    high_school_track: str = Field(
        ...,
        description="High school track (e.g. 'science', 'social')",
        examples=["science"],
    )

    # ── Academic performance (optional) ──────────────────────────────────
    overall_report_card_average: float = Field(
        default=0.0,
        description="Overall report card average score",
        examples=[85.0],
    )
    math_score: float = Field(
        default=0.0,
        description="Mathematics exam score",
        examples=[80.0],
    )
    english_score: float = Field(
        default=0.0,
        description="English exam score",
        examples=[75.0],
    )
    major_subject_average: float = Field(
        default=0.0,
        description="Average score in major subjects",
        examples=[82.0],
    )

    # ── Language & competitions (optional) ───────────────────────────────
    toefl_score: float = Field(
        default=0.0,
        description="TOEFL score (0-120)",
        examples=[65.0],
    )
    ielts_score: float = Field(
        default=0.0,
        description="IELTS score (0-9)",
        examples=[7.0],
    )
    language_proficiency: Optional[str] = Field(
        default=None,
        description="Language proficiency level or certification (deprecated, use toefl_score/ielts_score instead)",
        examples=["TOEFL 65"],
    )
    olympiad_level: str = Field(
        default="",
        description="Highest olympiad level achieved",
        examples=["national"],
    )
    olympiad_subjects: Optional[str] = Field(
        default=None,
        description="Olympiad subjects participated in",
        examples=["physics,mathematics"],
    )

    # ── Experience counts (optional) ─────────────────────────────────────
    leadership_experience_count: int = Field(
        default=0,
        description="Number of leadership positions held",
        examples=[3],
    )
    volunteer_experience_count: int = Field(
        default=0,
        description="Number of volunteer activities",
        examples=[5],
    )
    competition_wins_count: int = Field(
        default=0,
        description="Number of competition wins",
        examples=[2],
    )

    # ── Background (optional) ────────────────────────────────────────────
    school_tier: str = Field(
        default="",
        description="School tier classification",
        examples=["accredited_a"],
    )
    family_income_category: str = Field(
        default="",
        description="Family income category",
        examples=["upper_middle"],
    )
    from_underrepresented_region: bool = Field(
        default=False,
        description="Whether student is from an underrepresented region",
    )

    # ── Preferences (optional) ───────────────────────────────────────────
    intended_career_track: str = Field(
        default="",
        description="Intended career or field of study",
        examples=["computer_science"],
    )
    willing_to_return_home: bool = Field(
        default=False,
        description="Willingness to return home after studying abroad",
    )
    target_countries: Optional[str] = Field(
        default=None,
        description="Preferred destination countries",
        examples=["Japan,Germany"],
    )
    needs_full_funding: bool = Field(
        default=False,
        description="Whether full funding is required",
    )
    can_self_fund_living: bool = Field(
        default=False,
        description="Ability to self-fund living expenses",
    )

    # ── Text narratives (optional) ───────────────────────────────────────
    personal_statement: str = Field(
        default="",
        description="Personal statement or motivation letter",
        examples=["Passionate about STEM innovation and technology"],
    )
    achievements_narrative: str = Field(
        default="",
        description="Description of key achievements",
        examples=["National science olympiad participant"],
    )
    future_goals: str = Field(
        default="",
        description="Future career goals and aspirations",
        examples=["Technology entrepreneur building AI solutions"],
    )


class ScholarshipResult(BaseModel):
    """Single scholarship recommendation result.

    The `recommendation` field is currently a placeholder (empty string).
    Future: will contain LLM-generated personalized recommendation for the student.
    """
    scholarship_id: str
    score: float
    rank: int
    recommendation: str = Field(
        default="",
        description="Personalized recommendation explaining why this scholarship is a match",
    )
    metadata: dict


class RecommendationResponse(BaseModel):
    """Response for /recommend endpoint."""
    recommendations: list[ScholarshipResult]
    k: int


class RefreshResponse(BaseModel):
    """Response for /refresh endpoint."""
    status: str = Field(
        description="Operation status",
        examples=["refreshed"],
    )
    total_scholarships: int = Field(
        description="Total number of scholarships in cache after refresh",
        examples=[150],
    )


class HealthResponse(BaseModel):
    """Response for /health endpoint."""
    status: str = Field(description="Service health status")
    student_tower_loaded: bool = Field(description="Whether student tower model is loaded")
    scholarship_tower_loaded: bool = Field(description="Whether scholarship tower model is loaded")
    cached_scholarships: int = Field(description="Number of scholarships in cache")
    retraining: RetrainingInfo = Field(
        description="Current retraining job status and metadata",
    )


class RetrainStartResponse(BaseModel):
    """Response for /retrain endpoint (immediate, async)."""
    status: str
    message: str


class RetrainingInfo(BaseModel):
    """Retraining status info."""
    status: str  # idle | training | done | error
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    error: Optional[str] = None


# ── Application factory ───────────────────────────────────────────────────

def create_app(engine: InferenceEngine) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        engine: A warmed-up InferenceEngine instance.

    Returns:
        Configured FastAPI application.
    """
    app = FastAPI(
        title="ScholarshipID Recommendation API",
        description="Two-tower retrieval model for matching students to scholarships.",
        version="0.1.0",
    )

    # ── CORS Middleware ──────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[engine.serving_config.cors_origins]
            if engine.serving_config.cors_origins != "*"
            else ["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Endpoints ────────────────────────────────────────────────────────

    @app.post("/recommend", response_model=RecommendationResponse)
    async def recommend(
        student: StudentProfile,
        k: int = Query(5, ge=1, le=50),
    ):
        """Return top-K scholarships for the given student profile.

        The student body is encoded through the student tower, then matched
        against cached scholarship embeddings via dot-product (cosine similarity).
        """
        try:
            results = engine.recommend(student.model_dump(), k=k)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

        # Add placeholder recommendation field (future: LLM-generated per student)
        for r in results:
            r["recommendation"] = ""

        return RecommendationResponse(
            recommendations=[ScholarshipResult(**r) for r in results],
            k=k,
        )

    @app.post("/refresh", response_model=RefreshResponse)
    async def refresh(
        auth_token: Optional[str] = Depends(_get_auth_token),
        scholarships: UploadFile = File(..., description="Scholarship data CSV file"),
    ):
        """Rebuild the scholarship embedding cache from an uploaded CSV file.

        Call this endpoint after adding new scholarships to the data source.
        The model will use updated embeddings on the next /recommend call.

        Requires admin authentication when auth_required is enabled.
        """
        # Validate file extension
        if not scholarships.filename or not scholarships.filename.lower().endswith(".csv"):
            raise HTTPException(
                status_code=422,
                detail="Only .csv files are allowed",
            )

        # Auth check
        if engine.serving_config.auth_required:
            if auth_token is None or auth_token != engine.serving_config.auth_token:
                raise HTTPException(
                    status_code=HTTPStatus.UNAUTHORIZED,
                    detail="Invalid or missing authorization token",
                )

        try:
            content = await scholarships.read()
            engine.refresh_from_csv(content.decode("utf-8"))
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

        # Push updated data artifacts to HuggingFace on success (data only)
        try:
            push_data_artifacts(
                config_path=engine.config_path,
                message="Auto-push after /refresh",
            )
        except Exception as e:
            print(f"Warning: HF push failed after refresh: {e}", flush=True)

        return RefreshResponse(
            status="refreshed",
            total_scholarships=len(engine._sch_ids),
        )

    @app.post("/retrain", response_model=RetrainStartResponse)
    async def retrain(
        auth_token: Optional[str] = Depends(_get_auth_token),
        students: Optional[UploadFile] = File(None),
        scholarships: Optional[UploadFile] = File(None),
        feedbacks: Optional[UploadFile] = File(None),
    ):
        """Initiate model retraining with new data.

        Accepts 3 optional CSV files in the request body (multipart/form-data):
        - students: New student records (by student_id)
        - scholarships: New scholarship records (by scholarship_id)
        - feedbacks: New feedback records (by timestamp or feedback_id)

        At least one file is required. Training runs asynchronously in a background thread.
        Requires admin authentication when auth_required is enabled.

        The retraining process:
        1. Merges new data with existing disk data by ID
        2. Saves merged data back to disk
        3. Precomputes text embeddings on full merged datasets
        4. Trains model (finetuning from existing weights) on all feedback data
        5. Exports updated scholarship embeddings

        After completion, call /recommend without refresh — the engine auto-loads
        the latest embeddings from disk.
        """
        # Validate at least one file provided
        if not any([students, scholarships, feedbacks]):
            raise HTTPException(
                status_code=422,
                detail="At least one file is required (students, scholarships, or feedbacks)",
            )

        # Auth check
        if engine.serving_config.auth_required:
            if auth_token is None or auth_token != engine.serving_config.auth_token:
                raise HTTPException(
                    status_code=HTTPStatus.UNAUTHORIZED,
                    detail="Invalid or missing authorization token",
                )

        def _do_retrain():
            csv_parts = {}
            if students and students.filename:
                csv_parts["students"] = students.file.read().decode("utf-8")
            if scholarships and scholarships.filename:
                csv_parts["scholarships"] = scholarships.file.read().decode("utf-8")
            if feedbacks and feedbacks.filename:
                csv_parts["feedbacks"] = feedbacks.file.read().decode("utf-8")

            result = engine.retrain_from_csvs(
                students_csv_text=csv_parts.get("students", None),
                scholarships_csv_text=csv_parts.get("scholarships", None),
                feedbacks_csv_text=csv_parts.get("feedbacks", None),
            )

            # Push updated data + model artifacts to HuggingFace on success
            if result.get("status") == "done":
                try:
                    push_data_artifacts(
                        config_path=engine.config_path,
                        message="Auto-push after API retraining",
                    )
                    push_model_artifacts(
                        config_path=engine.config_path,
                        message="Auto-push after API retraining",
                    )
                except Exception as e:
                    print(f"Warning: HF push failed: {e}", flush=True)

        # Start training in background thread
        threading.Thread(target=_do_retrain, daemon=True).start()

        return RetrainStartResponse(
            status="training_started",
            message="Retraining initiated. Check /health for status.",
        )

    @app.get("/health", response_model=HealthResponse)
    async def health():
        """Health check — returns model loading status and retraining info."""
        retrain_info = engine.get_retraining_status()

        return HealthResponse(
            status="healthy",
            student_tower_loaded=engine.student_tower is not None,
            scholarship_tower_loaded=engine.scholarship_tower is not None,
            cached_scholarships=len(engine._sch_ids),
            retraining=retrain_info,
        )

    return app