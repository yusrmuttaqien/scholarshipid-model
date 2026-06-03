from dataclasses import dataclass, field
from typing import List, Optional

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

@dataclass
class LanguageRequirement:
    test_type: str   # LanguageTest enum value
    min_score: float
    is_mandatory: bool = True

@dataclass
class FundingCoverage:
    covers_tuition: bool = False
    covers_living_expense: bool = False
    covers_airfare: bool = False
    covers_insurance: bool = False
    monthly_stipend: float = 0.0

    @property
    def is_full_funding(self) -> bool:
        return self.covers_tuition and self.covers_living_expense

    @property
    def coverage_count(self) -> int:
        return sum([
            self.covers_tuition,
            self.covers_living_expense,
            self.covers_airfare,
            self.covers_insurance,
        ])

@dataclass
class SelectionCriteria:
    academic: float = 0.2
    leadership: float = 0.15
    olympiad: float = 0.3
    extracurricular: float = 0.15
    essay: float = 0.2

@dataclass
class Scholarship:
    # Identity
    scholarship_id: str
    name: str

    # Eligibility Constraints (Hard Filters)
    eligible_nationalities: List[str]        # Country values
    min_age: int
    max_age: int
    eligible_degree_levels: List[str]        # DegreeLevel values
    eligible_high_school_tracks: List[str]   # HighSchoolTrack values
    eligible_fields: List[str]              # MajorField values
    preferred_school_tier: str              # SchoolTier value
    min_report_card_average: float
    min_major_subject_average: float
    language_requirements: List[LanguageRequirement] = field(default_factory=list)
    requires_financial_need: bool = False
    max_family_income_category: str = "high"  # IncomeCategory value

    # Location
    host_country: str = ""   # Country value
    host_region: str = ""    # HostRegion value

    # Selection Criteria (weights sum to 1.0)
    selection_criteria: SelectionCriteria = field(default_factory=SelectionCriteria)

    # Funding Coverage
    funding_coverage: FundingCoverage = field(default_factory=FundingCoverage)

    # Career Preference
    career_track_preference: Optional[str] = None  # CareerTrack value

    # Service Requirements
    requires_return_home_country: bool = False

    # Text Fields (for text similarity in two-tower model)
    mission_statement: str = ""
    target_recipient_profile: str = ""

@dataclass
class Feedback:
    student_id: str
    scholarship_id: str
    feedback_type: str  # "accepted" | "apply" | "click"
    timestamp: str      # ISO 8601: "YYYY-MM-DDTHH:MM:SSZ"