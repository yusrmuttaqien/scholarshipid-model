# Scholarship.id: Two-Tower Recommendation System

Recommendation system for high school students seeking bachelor's scholarships abroad. Uses a two-tower neural network with a three-stage pipeline (hard filter → NN similarity → text similarity bonus).

---

## Data Structure Overview

### Core Entities

| Entity | Description | Count |
|--------|-------------|-------|
| `Student` | High school student profile | 20,000 |
| `Scholarship` | Scholarship opportunity | 800 |
| `Pair` | Student-scholarship interaction (for training) | ~250,000+ |
| `Feedback` | Implicit feedback from students | ~100,000+ |

---

## Student Fields

```python
@dataclass
class Student:
    # ─── Identity ─────────────────────────────────────
    student_id: str                    # Unique identifier (e.g., "STU_000001")
    nationality: Country

    # ─── Demographics ─────────────────────────────────
    age: int                           # 16-18 (high school students)
    current_degree_level: DegreeLevel  # Always High School
    target_degree_level: DegreeLevel   # Always Bachelor's

    # ─── Academic Profile ─────────────────────────────
    high_school_track: HighSchoolTrack
    school_name: str                   # School name
    overall_report_card_average: float # Overall report card average (0-100)
    math_score: float                  # Math score on latest report card (0-100)
    english_score: float               # English score on latest report card (0-100)
    major_subject_average: float       # Average score of major-specific subjects (0-100)
                                       #   Science → avg(Physics, Chemistry, Biology)
                                       #   Social Studies → avg(Economics, Geography, Sociology)
                                       #   Languages → avg(Literature, Anthropology, Foreign Language)

    # ─── Language Proficiency ─────────────────────────
    language_proficiency: List[LanguageProficiency]  # See LanguageProficiency below
                                                     # Empty list is valid for many high school students
                                                     # have not taken a formal language test yet

    # ─── Achievements ─────────────────────────────────
    olympiad_level: OlympiadLevel
    olympiad_subjects: List[OlympiadSubject]
    leadership_experience_count: int   # Leadership roles held (student council, club head, etc.)
    volunteer_experience_count: int    # Volunteer / community service activities
    competition_wins_count: int        # Non-olympiad competition awards

    # ─── Background ───────────────────────────────────
    school_tier: SchoolTier
    family_income_category: IncomeCategory
    from_underrepresented_region: bool # From underrepresented regions

    # ─── Career Intent ────────────────────────────────
    intended_career_track: CareerTrack
    willing_to_return_home: bool       # Willing to return home after study
    target_countries: List[Country]

    # ─── Text Fields (for text similarity) ────────────
    personal_statement: str            # Personal statement text
    achievements_narrative: str        # Achievement summary
    future_goals: str                  # Post-study contribution and career goals

    # ─── Funding Preferences ──────────────────────────
    needs_full_funding: bool           # Wants fully-funded scholarship
    can_self_fund_living: bool         # Can cover living expenses independently
```

---

## Scholarship Fields

```python
@dataclass
class Scholarship:
    # ─── Identity ─────────────────────────────────────
    scholarship_id: str                # Unique identifier (e.g., "SCH_000001")
    name: str                          # Scholarship name

    # ─── Eligibility Constraints (Hard Filters) ───────
    eligible_nationalities: List[Country]
    min_age: int                       # Minimum age requirement
    max_age: int                       # Maximum age requirement
    eligible_degree_levels: List[DegreeLevel]
    eligible_high_school_tracks: List[HighSchoolTrack]
    eligible_fields: List[MajorField]
    preferred_school_tier: SchoolTier
    min_report_card_average: float     # Minimum overall report card average (0-100)
    min_major_subject_average: float   # Minimum major-subject average (0-100)
    language_requirements: List[LanguageRequirement]  # See LanguageRequirement below
    requires_financial_need: bool      # Requires low-income family background
    max_family_income_category: IncomeCategory

    # ─── Location ─────────────────────────────────────
    host_country: Country
    host_region: HostRegion

    # ─── Selection Criteria (Weights) ─────────────────
    selection_criteria: SelectionCriteria  # See SelectionCriteria below

    # ─── Funding Coverage (Structured) ────────────────
    funding_coverage: FundingCoverage  # See FundingCoverage below

    # ─── Career Preference ────────────────────────────
    career_track_preference: CareerTrack

    # ─── Service Requirements ─────────────────────────
    requires_return_home_country: bool # Must return home after study

    # ─── Text Fields (for text similarity) ────────────
    mission_statement: str             # Scholarship mission text
    target_recipient_profile: str      # Ideal candidate description
```

---

## Supporting Data Structures

### LanguageProficiency

```python
@dataclass
class LanguageProficiency:
    test_type: LanguageTest
    score: float      # Test score
    valid_until: str  # ISO date string (optional)
```

### LanguageRequirement

```python
@dataclass
class LanguageRequirement:
    test_type: LanguageTest
    min_score: float    # Minimum required score
    is_mandatory: bool  # True = hard filter, False = preferred but not required
```

### FundingCoverage

```python
@dataclass
class FundingCoverage:
    covers_tuition: bool         # Covers tuition fees
    covers_living_expense: bool  # Covers living expenses
    covers_airfare: bool         # Covers airfare
    covers_insurance: bool       # Covers insurance
    monthly_stipend: float       # Monthly stipend in local currency (0 if none)

    # Derived properties:
    is_full_funding: bool        # Covers both tuition AND living
    coverage_count: int          # Number of aspects covered (0-4)
```

### SelectionCriteria

```python
@dataclass
class SelectionCriteria:
    academic: float        # Weight for academic excellence (0-1, normalized)
    leadership: float      # Weight for leadership experience
    olympiad: float        # Weight for olympiad / competition achievement
    extracurricular: float # Weight for extracurricular activities
    essay: float           # Weight for essay/motivation

    # Note: All weights sum to 1.0
```

### Pair

```python
@dataclass
class Pair:
    student_id: str         # Student identifier
    scholarship_id: str     # Scholarship identifier
    relevance_score: float  # Continuous score 0.0-1.0 (regression target)
                            #   >=0.7 → Match
                            #   0.3-0.7 → In-Between
                            #   <0.3 → Not Match
    timestamp: str          # ISO datetime for time-based splitting
```

### Feedback

```python
@dataclass
class Feedback:
    student_id: str       # Student identifier
    scholarship_id: str   # Scholarship identifier
    feedback_type: str    # "apply" | "click" | "view" | "reject"
    timestamp: str        # ISO datetime

    # Derived property:
    weight: float         # Training weight (see table below)
```

---

## Enum Values

### DegreeLevel

| Value | Description |
|-------|-------------|
| `high_school` | High school |
| `bachelors` | Bachelor's degree |

### Country

| Value | Country |
|-------|---------|
| `china` | China |
| `india` | India |
| `indonesia` | Indonesia |
| `japan` | Japan |
| `malaysia` | Malaysia |
| `philippines` | Philippines |
| `singapore` | Singapore |
| `south_korea` | South Korea |
| `thailand` | Thailand |
| `vietnam` | Vietnam |
| `france` | France |
| `germany` | Germany |
| `netherlands` | Netherlands |
| `sweden` | Sweden |
| `uk` | United Kingdom |
| `switzerland` | Switzerland |
| `canada` | Canada |
| `usa` | United States |
| `argentina` | Argentina |
| `brazil` | Brazil |
| `chile` | Chile |
| `egypt` | Egypt |
| `kenya` | Kenya |
| `morocco` | Morocco |
| `nigeria` | Nigeria |
| `south_africa` | South Africa |
| `australia` | Australia |
| `new_zealand` | New Zealand |

### HighSchoolTrack

| Value | Description |
|-------|-------------|
| `science` | Science track (Physics, Chemistry, Biology) |
| `social_studies` | Social Studies track (Economics, Geography, Sociology) |
| `languages` | Languages & Literature track |
| `religion` | Religious studies track |
| `vocational` | Vocational track |

### OlympiadLevel

| Value | Description |
|-------|-------------|
| `none` | Never competed |
| `school` | School-level (intra-school) |
| `city` | City / district level |
| `provincial` | Provincial level |
| `national` | National level |
| `international` | International (IMO, IPhO, IOI, etc.) |

### OlympiadSubject

| Value | Description |
|-------|-------------|
| `mathematics` | Mathematics olympiad |
| `physics` | Physics olympiad |
| `chemistry` | Chemistry olympiad |
| `biology` | Biology olympiad |
| `economics` | Economics olympiad |
| `geography` | Geography olympiad |
| `computer_science` | Computer science olympiad |
| `linguistics` | Linguistics olympiad |
| `astronomy` | Astronomy olympiad |
| `informatics` | Informatics olympiad |
| `history` | History olympiad |
| `english_language` | English language olympiad |
| `business_studies` | Business studies olympiad |

### IncomeCategory

| Value | Description |
|-------|-------------|
| `very_low` | Very low income |
| `low` | Low income |
| `middle` | Middle income |
| `upper_middle` | Upper middle income |
| `high` | High income |

### SchoolTier

| Value | Description |
|-------|-------------|
| `excellence` | Top-tier / boarding school |
| `public_a` | Public school, accredited A |
| `private_a` | Private school, accredited A |
| `accredited_b` | Public or private, accredited B |
| `accredited_c` | Accredited C schools |
| `unaccredited` | Not yet accredited |
| `unknown` | Accreditation status unknown |

### MajorField

| Value | Category |
|-------|----------|
| `computer_science` | CS/IT |
| `engineering` | Engineering |
| `medicine` | Medical sciences |
| `business` | Business administration |
| `economics` | Economics |
| `law` | Legal studies |
| `education` | Education |
| `arts_humanities` | Arts & humanities |
| `social_sciences` | Social sciences |
| `agriculture` | Agriculture |
| `mathematics` | Mathematics |
| `physics` | Physics |
| `chemistry` | Chemistry |
| `biology` | Biology |

### CareerTrack

| Value | Description |
|-------|-------------|
| `academic` | Academic/research career |
| `industry` | Industry/tech career |
| `government` | Government service |
| `ngo_npo` | NGO/NPO sector |
| `entrepreneurship` | Entrepreneurship |
| `public_service` | Public service |

### LanguageTest

| Value | Description |
|-------|-------------|
| `toefl` | TOEFL (English) |
| `ielts` | IELTS (English) |
| `topik` | TOPIK (Korean) |
| `jlpt` | JLPT (Japanese) |
| `delf` | DELF (French) |
| `hsk` | HSK (Chinese) |

### HostRegion

| Value | Description |
|-------|-------------|
| `asia` | Asia |
| `europe` | Europe |
| `north_america` | North America |
| `south_america` | South America |
| `africa` | Africa |
| `oceania` | Oceania |

---

## Feedback Types & Weights

| Type | Weight | Signal Strength | Description |
|------|--------|-----------------|-------------|
| `apply` | 3.0 | Strong positive | Applied to scholarship |
| `click` | 2.0 | Soft positive | Clicked on recommendation |
| `view` | 1.0 | Weak positive | Saw in recommendations |
| `reject` | -1.0 | Explicit negative | Swiped away / not interested |

---

## Training vs Inference

### Model Training (Dataset Creation)

The two-tower neural network is trained as a **regression model** that predicts continuous `relevance_score` (0.0–1.0). The training dataset contains balanced pairs across three classes:

| Class | Relevance Range | Description |
|-------|-----------------|-------------|
| **Match** | ≥ 0.7 | High alignment between student and scholarship |
| **In-Between** | 0.3 – 0.7 | Moderate alignment, borderline cases |
| **Not Match** | < 0.3 | Low alignment between student and scholarship |

**Key point**: Hard filters are NOT used during training. The model learns soft similarity from the full spectrum of pairs, including ineligible matches (high relevance_score) and eligible but poor matches (low relevance_score). This allows the NN to learn fine-grained distinctions.

**Why train without hard filters?** The relevance_score target is computed purely from attribute alignment (see `generator_two_tower.py`). Hard filter constraints like nationality or age are already reflected in the score — a student from an ineligible country will naturally receive a low relevance_score. The NN learns to reproduce this soft signal, while the hard filter acts as a strict gatekeeper only at inference time.

### Inference Pipeline (Serving Recommendations)

After training, the three-stage pipeline is applied at inference time to produce final ranked recommendations:

```
┌─────────────────────────────────────────────┐
│  Stage 1: Hard Filter (Deterministic)       │
│  ─────────────────────────────────────────   │
│  • Nationality in eligible_nationalities    │
│  • Target degree in eligible_degree_levels  │
│  • Age within [min_age, max_age]            │
│  • Report card average meets minimum        │
│  • Major subject average meets minimum      │
│  • Mandatory language requirements met      │
│  • Return-home willingness check            │
│  • Financial need check (if required)       │
│                                             │
│  Output: Eligible scholarships only         │
└──────────────────┬──────────────────────────┘
                   ↓
┌─────────────────────────────────────────────┐
│  Stage 2: Two-Tower Neural Network          │
│  ─────────────────────────────────────────   │
│  Student Tower → Embedding (d=64-128)       │
│  Scholarship Tower → Embedding (d=64-128)   │
│                                             │
│  Cosine Similarity → Sigmoid → Probability  │
│                                             │
│  Output: Soft similarity score [0, 1]       │
└──────────────────┬──────────────────────────┘
                   ↓
┌─────────────────────────────────────────────┐
│  Stage 3: Text Similarity (Bonus)           │
│  ─────────────────────────────────────────   │
│  • personal_statement ↔ mission_statement   │
│  • achievements_narrative ↔ target_recipient_profile  │
│  • future_goals ↔ target_recipient_profile            │
│                                             │
│  Output: Text bonus added to NN score       │
│  Final score clamped to [0, 1]              │
└──────────────────┬──────────────────────────┘
                   ↓
┌─────────────────────────────────────────────┐
│  Final Ranked Recommendations per Student   │
└─────────────────────────────────────────────┘
