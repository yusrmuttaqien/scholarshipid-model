"""5-stage relevance scoring pipeline for student–scholarship pairs.

All functions are pure (no class state). Pass `rng` and `countries_by_region`
explicitly so callers control randomness and geography lookups.

Pipeline:
  1. compute_eligibility_multiplier — hard knockouts
  2. score_* functions — component scores
  3. (caller) core = weighted sum via selection_criteria
  4. compute_fit_bonuses — lateral preference signals
  5. compute_relevance_score — combines all stages + noise
"""

from typing import Optional

import numpy as np

from src.schemas import Scholarship, Student

# ── Ordinal lookup tables ────────────────────────────────────────────────────

INCOME_ORDER = ["very_low", "low", "middle", "upper_middle", "high"]

TIER_ORDER = [
    "excellence", "public_a", "private_a",
    "accredited_b", "accredited_c", "unaccredited", "unknown",
]

OLYMPIAD_LEVEL_SCORE = {
    # "none" is 0.30, not 0: lack of olympiad is a low-but-present signal.
    "none": 0.30,
    "school": 0.45,
    "city": 0.60,
    "provincial": 0.75,
    "national": 0.90,
    "international": 1.00,
}

# Olympiad subject → MajorField buckets (cross-relevance with eligible_fields)
OLYMPIAD_TO_FIELDS = {
    "mathematics":      {"mathematics", "computer_science", "engineering", "physics", "economics"},
    "physics":          {"physics", "engineering", "mathematics"},
    "chemistry":        {"chemistry", "medicine", "biology", "engineering"},
    "biology":          {"biology", "medicine", "agriculture"},
    "economics":        {"economics", "business", "social_sciences"},
    "geography":        {"social_sciences", "agriculture"},
    "computer_science": {"computer_science", "engineering", "mathematics"},
    "informatics":      {"computer_science", "engineering", "mathematics"},
    "linguistics":      {"arts_humanities", "education"},
    "astronomy":        {"physics", "mathematics"},
    "history":          {"arts_humanities", "social_sciences", "education"},
    "english_language": {"arts_humanities", "education"},
    "business_studies": {"business", "economics"},
}


# ── Stage 1: Hard eligibility ────────────────────────────────────────────────

def compute_eligibility_multiplier(student: Student, scholarship: Scholarship) -> float:
    """Return 0.0 for hard knockouts, 0.5 for grace-zone violations, 1.0 otherwise."""
    mult = 1.0

    if student.nationality not in scholarship.eligible_nationalities:
        return 0.0

    if (
        student.current_degree_level not in scholarship.eligible_degree_levels
        and student.target_degree_level not in scholarship.eligible_degree_levels
    ):
        return 0.0

    if student.age < scholarship.min_age - 1 or student.age > scholarship.max_age + 1:
        return 0.0
    if not (scholarship.min_age <= student.age <= scholarship.max_age):
        mult *= 0.5

    for req in scholarship.language_requirements:
        if not req.is_mandatory:
            continue
        student_scores = [
            lp.score for lp in student.language_proficiency
            if lp.test_type == req.test_type
        ]
        if not student_scores or max(student_scores) < req.min_score:
            return 0.0

    if scholarship.requires_financial_need:
        try:
            max_idx = INCOME_ORDER.index(scholarship.max_family_income_category)
            stu_idx = INCOME_ORDER.index(student.family_income_category)
            if stu_idx > max_idx:
                return 0.0
        except ValueError:
            pass

    if scholarship.requires_return_home_country and not student.willing_to_return_home:
        mult *= 0.5

    return mult


# ── Stage 2: Component scores ────────────────────────────────────────────────

def score_academic(student: Student, scholarship: Scholarship) -> float:
    """Composite academic score: overall, major, math, english vs scholarship minimums."""
    def margin(score: float, minimum: float) -> float:
        if score >= minimum:
            return min(0.7 + (score - minimum) / 20.0 * 0.3, 1.0)
        return max(0.7 - (minimum - score) / 20.0 * 0.7, 0.0)

    overall = margin(student.overall_report_card_average, scholarship.min_report_card_average)
    major   = margin(student.major_subject_average, scholarship.min_major_subject_average)
    math    = margin(student.math_score, scholarship.min_major_subject_average)
    english = margin(student.english_score, scholarship.min_report_card_average)
    return 0.35 * overall + 0.35 * major + 0.15 * math + 0.15 * english


def score_olympiad(student: Student, scholarship: Scholarship) -> float:
    """Olympiad level × subject relevance to scholarship's eligible fields."""
    level_score = OLYMPIAD_LEVEL_SCORE.get(student.olympiad_level, 0.0)
    if level_score == 0.0 or not student.olympiad_subjects:
        return level_score

    eligible_fields = set(scholarship.eligible_fields)
    if not eligible_fields:
        return level_score * 0.6

    relevant = sum(
        1 for subj in student.olympiad_subjects
        if OLYMPIAD_TO_FIELDS.get(subj, set()) & eligible_fields
    )
    subj_relevance = relevant / len(student.olympiad_subjects)
    return level_score * (0.5 + 0.5 * subj_relevance)


def score_leadership(student: Student) -> float:
    """Saturating function: 0→0.25, 1→0.55, 2→0.73, 3→0.84, 5+→~1.0"""
    n = max(student.leadership_experience_count, 0)
    return min(1.0, 0.25 + 0.75 * (1.0 - (0.6 ** n)))


def score_extracurricular(student: Student) -> float:
    """Composite of volunteer experience + competition wins."""
    v = max(student.volunteer_experience_count, 0)
    c = max(student.competition_wins_count, 0)
    vol  = min(1.0, 0.25 + 0.75 * (1.0 - (0.65 ** v)))
    comp = min(1.0, 0.20 + 0.80 * (1.0 - (0.55 ** c)))
    return 0.6 * vol + 0.4 * comp


def score_essay_placeholder(rng: np.random.RandomState) -> float:
    """Stub: text-similarity needs embeddings. Mild-optimistic 0.6 + noise."""
    return float(np.clip(0.6 + rng.normal(0, 0.08), 0.0, 1.0))


def score_language_bonus(student: Student, scholarship: Scholarship) -> float:
    """Non-knockout language signal (mandatory requirements already gated in Stage 1)."""
    if not scholarship.language_requirements:
        return 1.0
    scores = []
    for req in scholarship.language_requirements:
        student_scores = [
            lp.score for lp in student.language_proficiency
            if lp.test_type == req.test_type
        ]
        if student_scores:
            max_s = max(student_scores)
            scores.append(min(1.0, max_s / req.min_score) if req.min_score > 0 else 1.0)
        elif not req.is_mandatory:
            scores.append(0.5)
    return sum(scores) / len(scores) if scores else 1.0


# ── Stage 4: Fit bonuses ─────────────────────────────────────────────────────

def _region_of(country: str, countries_by_region: dict) -> Optional[str]:
    """Reverse-lookup region from country string."""
    for region, countries in countries_by_region.items():
        if country in countries:
            return region
    return None


def compute_fit_bonuses(
    student: Student,
    scholarship: Scholarship,
    countries_by_region: dict,
) -> float:
    """Stage 4: lateral preference signals. Returns weighted average 0–1."""
    track_fit = 1.0 if student.high_school_track in scholarship.eligible_high_school_tracks else 0.3

    eligible_fields = set(scholarship.eligible_fields)
    if not eligible_fields or not student.olympiad_subjects:
        field_fit = 0.5
    else:
        mapped: set = set()
        for subj in student.olympiad_subjects:
            mapped |= OLYMPIAD_TO_FIELDS.get(subj, set())
        field_fit = 1.0 if mapped & eligible_fields else (0.3 if mapped else 0.5)

    if scholarship.host_country and scholarship.host_country in student.target_countries:
        loc_fit = 1.0
    elif scholarship.host_region:
        target_regions = {_region_of(c, countries_by_region) for c in student.target_countries}
        target_regions.discard(None)
        loc_fit = 0.6 if scholarship.host_region in target_regions else 0.2
    else:
        loc_fit = 0.5

    if not scholarship.career_track_preference:
        career_fit = 0.6
    elif student.intended_career_track == scholarship.career_track_preference:
        career_fit = 1.0
    else:
        career_fit = 0.4

    try:
        pref_idx = TIER_ORDER.index(scholarship.preferred_school_tier)
        stu_idx  = TIER_ORDER.index(student.school_tier)
        tier_fit = 1.0 if stu_idx <= pref_idx else max(0.2, 1.0 - 0.15 * (stu_idx - pref_idx))
    except ValueError:
        tier_fit = 0.6

    is_full = scholarship.funding_coverage.is_full_funding
    if student.needs_full_funding:
        fund_fit = 1.0 if is_full else 0.3
    elif student.can_self_fund_living:
        fund_fit = 0.9 if is_full else 0.8
    else:
        fund_fit = 0.7 if is_full else 0.6

    return (
        0.20 * track_fit
        + 0.18 * field_fit
        + 0.18 * loc_fit
        + 0.14 * career_fit
        + 0.10 * tier_fit
        + 0.20 * fund_fit
    )


# ── Main entry point ─────────────────────────────────────────────────────────

def compute_relevance_score(
    student: Student,
    scholarship: Scholarship,
    rng: np.random.RandomState,
    countries_by_region: dict,
) -> float:
    """Continuous relevance score (0.0–1.0) via the 5-stage pipeline."""
    elig = compute_eligibility_multiplier(student, scholarship)
    if elig == 0.0:
        return float(np.clip(rng.uniform(0.0, 0.05), 0.0, 1.0))

    acad  = score_academic(student, scholarship)
    oly   = score_olympiad(student, scholarship)
    lead  = score_leadership(student)
    extra = score_extracurricular(student)
    essay = score_essay_placeholder(rng)

    sc = scholarship.selection_criteria
    core = (
        sc.academic        * acad
        + sc.olympiad      * oly
        + sc.leadership    * lead
        + sc.extracurricular * extra
        + sc.essay         * essay
    )

    fit        = compute_fit_bonuses(student, scholarship, countries_by_region)
    lang_bonus = score_language_bonus(student, scholarship)

    base = 0.55 * core + 0.35 * fit + 0.10 * lang_bonus

    favors_diversity = (
        scholarship.requires_financial_need
        or scholarship.host_country in {"japan", "south_korea", "uk", "usa", "australia"}
    )
    if student.from_underrepresented_region and favors_diversity:
        base = min(1.0, base + 0.05)

    relevance = base * elig
    relevance = float(np.clip(relevance + rng.normal(0, 0.02), 0.0, 1.0))
    return round(relevance, 4)
