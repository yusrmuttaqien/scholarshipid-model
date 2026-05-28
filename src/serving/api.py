"""FastAPI application for scholarship recommendation serving.

Endpoints:
    POST /recommend  — Recommend top-K scholarships for a student profile
    POST /refresh    — Refresh scholarship cache (admin)
    GET  /health     — Health check
"""
from __future__ import annotations

from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from src.serving.inference_engine import InferenceEngine


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
    """Single scholarship recommendation result."""
    scholarship_id: str
    score: float
    rank: int
    metadata: dict


class RecommendationResponse(BaseModel):
    """Response for /recommend endpoint."""
    recommendations: list[ScholarshipResult]
    k: int


class RefreshResponse(BaseModel):
    """Response for /refresh endpoint."""
    status: str
    total_scholarships: int


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

        return RecommendationResponse(
            recommendations=[ScholarshipResult(**r) for r in results],
            k=k,
        )

    @app.post("/refresh", response_model=RefreshResponse)
    async def refresh():
        """Rebuild the scholarship embedding cache.

        Call this endpoint after new scholarships are added to the data source.
        Requires admin authentication in production.
        """
        try:
            engine.refresh_scholarships()
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

        return RefreshResponse(
            status="refreshed",
            total_scholarships=len(engine._sch_ids),
        )

    @app.get("/health")
    async def health():
        """Health check — returns model loading status."""
        return {
            "status": "healthy",
            "student_tower_loaded": engine.student_tower is not None,
            "scholarship_tower_loaded": engine.scholarship_tower is not None,
            "cached_scholarships": len(engine._sch_ids),
        }

    return app