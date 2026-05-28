from dataclasses import dataclass, field
from typing import List, Optional

from .enums import (
    CareerTrack,
    Country,
    DegreeLevel,
    HighSchoolTrack,
    IncomeCategory,
    LanguageTest,
    OlympiadLevel,
    OlympiadSubject,
    SchoolTier,
)


@dataclass
class LanguageProficiency:
    test_type: str
    score: float
    valid_until: Optional[str] = None


@dataclass
class Student:
    # Identity
    student_id: str
    nationality: str  # Country enum value

    # Demographics
    age: int
    current_degree_level: str  # DegreeLevel enum value
    target_degree_level: str   # DegreeLevel enum value

    # Academic Profile
    high_school_track: str  # HighSchoolTrack enum value
    school_name: str
    overall_report_card_average: float
    math_score: float
    english_score: float
    major_subject_average: float

    # Language Proficiency
    language_proficiency: List[LanguageProficiency] = field(default_factory=list)

    # Achievements
    olympiad_level: str = "none"  # OlympiadLevel enum value
    olympiad_subjects: List[str] = field(default_factory=list)  # OlympiadSubject values
    leadership_experience_count: int = 0
    volunteer_experience_count: int = 0
    competition_wins_count: int = 0

    # Background
    school_tier: str = "unknown"            # SchoolTier enum value
    family_income_category: str = "middle"  # IncomeCategory enum value
    from_underrepresented_region: bool = False

    # Career Intent
    intended_career_track: str = ""  # CareerTrack enum value
    willing_to_return_home: bool = True
    target_countries: List[str] = field(default_factory=list)  # Country values

    # Text Fields (for text similarity in two-tower model)
    personal_statement: str = ""
    achievements_narrative: str = ""
    future_goals: str = ""

    # Funding Preferences
    needs_full_funding: bool = False
    can_self_fund_living: bool = False
