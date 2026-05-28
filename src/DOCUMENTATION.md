# src/ — Technical Documentation

Scholarship matching model: a pure-function 5-stage pipeline that scores student–scholarship pairs on a [0, 1] scale for use in a two-tower neural network.

---

## Module Index

| File | Role |
|---|---|
| `scorer.py` | Core scoring pipeline — 5 stages, pure functions |
| `io.py` | Serializes domain objects to CSV for ML training |
| `data/scholarships.py` | ~18 hardcoded scholarship profiles |
| `data/students.py` | Seed data (countries, schools, templates) |
| `schemas/enums.py` | Enums + multilingual normalization maps |
| `schemas/scholarship.py` | Scholarship, FundingCoverage, SelectionCriteria |
| `schemas/student.py` | Student profile dataclass |
| `schemas/pair.py` | Pairing entity (student_id + scholarship_id + score) |
| `schemas/feedback.py` | Feedback loop for match corrections |

---

## scorer.py — 5-Stage Pipeline

All functions are pure. No class state. Accepts `rng: np.random.RandomState` and `countries_by_region: dict` explicitly so callers control randomness and geography lookups.

### Stage 1: Eligibility Gate

| Condition | Effect |
|---|---|
| Nationality mismatch | → 0.0 (hard knockout) |
| Degree level mismatch | → 0.0 |
| Age outside range ±1 | → 0.5 (soft penalty if within ±1, hard knockout otherwise) |
| Mandatory language score below minimum | → 0.0 |
| Financial need violated | → 0.0 |
| Return-home requirement not met | → 0.5 |

### Stage 2: Component Scores (each returns [0, 1])

- **score_academic** — Margin-aware composite of report card, major subject, math, and English scores vs. scholarship minimums. Exceeding threshold rewards up to 1.0; falling short penalizes down to 0.0.
- **score_olympiad** — Olympiad level score (none=0.30 → international=1.00) × subject-to-field cross-relevance. Maps olympiad subjects (math, CS, economics, etc.) to eligible academic fields via a predefined ontology.
- **score_leadership** — Saturating curve: 0→0.25, 1→0.55, 3+→~0.96. Diminishing returns prevent gaming.
- **score_extracurricular** — Composite of volunteer experience (saturation) + competition wins (saturation). Weights: 60% vol / 40% comp.
- **score_essay_placeholder** — Stub returning ~0.6 + Gaussian noise. To be replaced by text-similarity from the two-tower embeddings.
- **score_language_bonus** — Non-knockout language signal. Computes how student test scores exceed minimums for non-mandatory requirements (mandatory ones already gated in Stage 1).

### Stage 3: Weighted Core Score

`core = Σ(selection_criteria_weight_i × score_i)` using scholarship-specific weights (e.g., MEXT: academic=0.45, olympiad=0.20; UWC: leadership=0.25, extracurricular=0.20).

### Stage 4: Fit Bonuses

| Signal | Description |
|---|---|
| Track fit | High school track (science/social_studies/languages) in eligible tracks |
| Field fit | Olympiad subjects mapped to fields intersect with eligible fields |
| Location fit | Host country/region in student's target countries |
| Career fit | Intended career track matches scholarship preference |
| Tier fit | Student school tier ordinal ≤ preferred tier (penalizes upward mismatch) |
| Fund fit | Funding need × funding provision alignment |

**Output**: weighted average of all signals.

### Stage 5: Final Score

```
base = 0.55 × core + 0.35 × fit + 0.10 × language_bonus
if (underrepresented_region AND diversity_preferring_scholarship): base += 0.05
score = clamp(base × eligibility_multiplier + noise(σ=0.02), 0, 1)
```

Ineligible pairs get uniform random in [0, 0.05] rather than exact zero (preserves training signal).

---

## data/scholarships.py — Scholarship Profiles

Each function returns `list[Scholarship]`. All fields hardcoded — no API calls.

| Profile | Host | Key Traits |
|---|---|---|
| MEXT Undergraduate | Japan | STEM focus, olympiad-heavy |
| GKS (Korean) | South Korea | Fully funded, broad fields |
| ASEAN Scholarship | Singapore | Elite tier, age 16–18 |
| Stipendium Hungaricum | Hungary | Multi-field, European |
| Türkiye Bursları | Turkey* | Broad, cultural exchange |
| CSC (Chinese Gov) | China | STEM-heavy, large stipend |
| BIM (S1 Luar Negeri) | Multiple* | Indonesian nationals only, mandatory return |
| UWC | Netherlands* | Needs-based, NGO track |
| AFS Intercultural | USA* | High school exchange, needs-based |
| NUS/NTU ASEAN | Singapore | Elite STEM, separate entries per uni |
| Australia Awards | Australia | Development-focused, return-home required |
| Russian Gov | Russia* | STEM, preparatory language year |

\* Host country uses placeholder mapping where the actual country is not in `Country` enum.

---

## data/students.py — Seed Data

- **COUNTRIES_BY_REGION** — 6 regions, ~30 countries total; used by scorer for location-fit computation
- **SCHOOL_NAMES** — Per-country school name pools
- **PERSONAL_STATEMENT_TEMPLATES / FUTURE_GOALS_TEMPLATES** — With `{field}`, `{interest}`, `{country}` placeholders
- **RESEARCH_INTERESTS** — 12 research topics for synthetic generation
- **ACHIEVEMENT_TEMPLATES** — Olympiad, competition, volunteer patterns

---

## schemas/enums.py — Domain Types

| Enum | Values |
|---|---|
| DegreeLevel | high_school, bachelors |
| Country | ~30 countries (Asia/Europe/Americas/Africa/Oceania) |
| HighSchoolTrack | science, social_studies, languages, religion, vocational |
| OlympiadLevel | none → school → city → provincial → national → international |
| OlympiadSubject | 13 subjects (mathematics through business_studies) |
| IncomeCategory | very_low → low → middle → upper_middle → high |
| SchoolTier | excellence → public_a → private_a → accredited_b/c → unaccredited → unknown |
| MajorField | 13 fields (cs, engineering, medicine, economics, etc.) |
| CareerTrack | academic, industry, government, ngo_npo, entrepreneurship, public_service |
| LanguageTest | toefl, ielts, topik, jlpt, delf, hsk |
| HostRegion | 6 regions |

**Normalization**: `normalize_country()` and `normalize_field()` map raw strings (including Indonesian/Malay variants like "jepang"→japan, "kedokteran"→medicine) to enum values.

---

## schemas/scholarship.py — Domain Model

- **LanguageRequirement** — test_type, min_score, is_mandatory
- **FundingCoverage** — 4 boolean flags + monthly_stipend; properties: `is_full_funding`, `coverage_count`
- **SelectionCriteria** — academic, leadership, olympiad, extracurricular, essay (weights sum to 1.0)
- **Scholarship** — aggregates all constraints + selection_criteria + funding + career_preference + text fields (`mission_statement`, `target_recipient_profile`) for two-tower text similarity

---

## schemas/student.py — Student Profile

Identity → Demographics → Academic (GPA, math, English, major avg) → Language proficiency list → Olympiad (level + subjects) → Leadership/volunteer/competition counts → School tier + income → Career intent + target countries → Text fields (`personal_statement`, `achievements_narrative`, `future_goals`) → Funding preferences.

---

## Design Principles

1. **Pure functions** — zero mutable state, explicit dependencies
2. **Deterministic RNG** — same inputs + seed = identical output
3. **Hard/soft constraint split** — nationality=hard knockout vs. age±1=0.5 penalty
4. **Saturation curves** — exponential decay prevents score runaway on count-based metrics
5. **Externalized geography** — `countries_by_region` passed as parameter, not hardcoded
6. **Two-tower text fields** — `mission_statement`, `personal_statement`, etc. feed the embedding model's similarity branch