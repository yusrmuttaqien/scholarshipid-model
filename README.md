# Scholarship.id — Two-Tower Recommendation System

Recommendation system for SMA-level high school students seeking scholarships abroad. Uses a two-tower neural network with a three-stage pipeline (hard filter → NN similarity → TF-IDF text bonus).

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
    nationality: str                  # Country of citizenship

    # ─── Demographics ─────────────────────────────────
    age: int                          # 16-18 (SMA students)
    current_degree_level: str         # Always "SMA"
    target_degree_level: str          # "S1" (80%) or "S2" (20%)

    # ─── Academic Profile ─────────────────────────────
    current_major_field: str          # MajorField enum value
    current_university: str           # School name
    current_university_tier: str      # UniversityTier enum value
    current_gpa_4scale: float         # GPA on 4.0 scale (2.0-4.0)
    current_gpa_percentage: float     # GPA on percentage scale (50-100)

    # ─── Language Proficiency ─────────────────────────
    language_proficiency: List[LanguageProficiency]  # See LanguageProficiency below

    # ─── Experience Counts ────────────────────────────
    research_experience_count: int    # Number of research projects
    leadership_experience_count: int  # Leadership roles held
    volunteer_experience_count: int   # Volunteer hours/projects
    competition_wins_count: int       # Competition awards

    # ─── Career Intent ────────────────────────────────
    intended_career_track: str        # CareerTrack enum value
    willing_to_return_home: bool      # Willing to return home after study
    target_countries: List[str]       # Preferred destination countries

    # ─── Text Fields (for TF-IDF similarity) ──────────
    personal_statement: str           # Personal statement text
    achievements_narrative: str       # Achievement summary
    research_interest: str            # Research interest area

    # ─── Funding Preferences ──────────────────────────
    needs_full_funding: bool          # Wants fully-funded scholarship
    can_self_fund_living: bool        # Can cover living expenses independently
```

---

## Scholarship Fields

```python
@dataclass
class Scholarship:
    # ─── Identity ─────────────────────────────────────
    scholarship_id: str               # Unique identifier (e.g., "SCH_000001")
    name: str                         # Scholarship name

    # ─── Eligibility Constraints (Hard Filters) ───────
    eligible_nationalities: List[str] # Countries accepted
    min_age: int                      # Minimum age requirement
    max_age: int                      # Maximum age requirement
    eligible_degree_levels: List[str] # Degree levels accepted (always includes "SMA")
    eligible_fields: List[str]        # MajorField enum values accepted
    eligible_majors_specific: List[str]  # Optional specific majors (30% chance)
    min_gpa_4scale: float             # Minimum GPA on 4.0 scale
    min_gpa_percentage: float         # Minimum GPA on percentage scale
    language_requirements: List[LanguageRequirement]  # See LanguageRequirement below

    # ─── Location ─────────────────────────────────────
    host_country: str                 # Country where scholarship is based
    host_region: str                  # HostRegion enum value

    # ─── Selection Criteria (Weights) ─────────────────
    selection_criteria: SelectionCriteria  # See SelectionCriteria below

    # ─── Funding Coverage (Structured) ────────────────
    funding_coverage: FundingCoverage # See FundingCoverage below

    # ─── Career Preference ────────────────────────────
    career_track_preference: str      # Preferred career track (optional)

    # ─── Service Requirements ─────────────────────────
    requires_return_home_country: bool  # Must return home after study

    # ─── Text Fields (for TF-IDF similarity) ──────────
    mission_statement: str            # Scholarship mission text
    target_recipient_profile: str     # Ideal candidate description
```

---

## Supporting Data Structures

### LanguageProficiency

```python
@dataclass
class LanguageProficiency:
    test_type: str    # LanguageTest enum value
    score: float      # Test score
    valid_until: str  # ISO date string (optional)
```

### LanguageRequirement

```python
@dataclass
class LanguageRequirement:
    test_type: str    # LanguageTest enum value
    min_score: float  # Minimum required score
    is_mandatory: bool  # True = hard filter, False = preferred but not required
```

### FundingCoverage

```python
@dataclass
class FundingCoverage:
    covers_tuition: bool        # Covers tuition fees
    covers_living_expense: bool # Covers living expenses
    covers_airfare: bool        # Covers airfare
    covers_insurance: bool      # Covers insurance
    total_amount: float         # Total USD equivalent

    # Derived properties:
    is_full_funding: bool       # Covers both tuition AND living
    coverage_count: int         # Number of aspects covered (0-4)
```

### SelectionCriteria

```python
@dataclass
class SelectionCriteria:
    academic: float          # Weight for academic excellence (0-1, normalized)
    leadership: float        # Weight for leadership experience
    research: float          # Weight for research orientation
    extracurricular: float   # Weight for extracurricular activities
    essay: float             # Weight for essay/motivation

    # Note: All weights sum to 1.0
```

### Pair

```python
@dataclass
class Pair:
    student_id: str          # Student identifier
    scholarship_id: str      # Scholarship identifier
    label: int               # 1 = positive (eligible), 0 = negative
    relevance_score: float   # 0.0-1.0 soft relevance for ranking loss
    timestamp: str           # ISO datetime for time-based splitting
```

### Feedback

```python
@dataclass
class Feedback:
    student_id: str          # Student identifier
    scholarship_id: str      # Scholarship identifier
    feedback_type: str       # "apply" | "click" | "view" | "reject"
    timestamp: str           # ISO datetime

    # Derived property:
    weight: float            # Training weight (see table below)
```

---

## Enum Values

### DegreeLevel

| Value | Description |
|-------|-------------|
| `SMA` | Indonesian high school |
| `S1` | Bachelor's degree |
| `S2` | Master's degree |
| `S3` | PhD/Doctorate |

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
| `gre` | GRE (Graduate entrance) |

### HostRegion

| Value | Description |
|-------|-------------|
| `asia` | Asia |
| `europe` | Europe |
| `north_america` | North America |
| `south_america` | South America |
| `africa` | Africa |
| `oceania` | Oceania |

### UniversityTier

| Value | Description |
|-------|-------------|
| `tier1` | Top-tier universities |
| `tier2` | Mid-tier universities |
| `tier3` | Lower-tier universities |
| `other` | Other/not classified |

---

## Feedback Types & Weights

| Type | Weight | Signal Strength | Description |
|------|--------|-----------------|-------------|
| `apply` | 3.0 | Strong positive | Applied to scholarship |
| `click` | 2.0 | Soft positive | Clicked on recommendation |
| `view` | 1.0 | Weak positive | Saw in recommendations |
| `reject` | -1.0 | Explicit negative | Swiped away / not interested |

---

## Dataset Files

### Approach A: Separate Pools (`generator.py`)

```
datasets/
├── train/
│   ├── students.csv          # 14,000 students
│   ├── scholarships.csv      # 560 scholarships
│   ├── pairs.csv             # ~38,000 pairs
│   ├── positive_pairs.csv    # ~2,500 positive
│   ├── negative_pairs.csv    # ~35,500 negative
│   └── feedback.csv          # ~104,000 entries
├── val/                       # Same structure (smaller)
└── test/                     # Same structure (smaller)
```

### Approach B: Single Pool (`generator_single_pool.py`)

```
datasets_single_pool/
├── students.csv               # 20,000 students (shared)
├── scholarships.csv           # 800 scholarships (shared)
├── pairs.csv                  # ALL pairs with timestamps
└── feedback.csv               # ALL feedback with timestamps
```

---

## Pipeline Architecture

```
┌─────────────────────────────────────────────┐
│  Stage 1: Hard Filter (Deterministic)       │
│  ─────────────────────────────────────────   │
│  • Nationality in eligible_nationalities    │
│  • Target degree in eligible_degree_levels  │
│  • Age within [min_age, max_age]           │
│  • GPA meets minimum (correct scale)        │
│  • Mandatory language requirements met      │
│  • Return-home willingness check            │
│                                             │
│  Output: Eligible pairs only                │
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
│  Stage 3: TF-IDF Text Similarity (Bonus)   │
│  ─────────────────────────────────────────   │
│  • personal_statement ↔ mission_statement   │
│  • achievements_narrative ↔ target_profile  │
│                                             │
│  Output: Text bonus added to NN score       │
└──────────────────┬──────────────────────────┘
                   ↓
┌─────────────────────────────────────────────┐
│  Final Ranked Recommendations per Student   │
└─────────────────────────────────────────────┘
```

---

## Generation Commands

```bash
# Approach A: Separate pools per split
python generator.py

# Approach B: Single pool with time-based splitting
python generator_single_pool.py
```