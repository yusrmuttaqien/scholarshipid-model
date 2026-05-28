# Scholarship Recommendation System — user questions
Recommendation system for high school students seeking bachelor's scholarships abroad.

## The architecture
Model trained with `students` and `scholarships` dataset. Using `interaction` dataset to iterate on the model, finetuning and retraining for better accuracy and reflect real-world situation.

For model type, we still deciding to either use:
1. `Two tower model`, or
2. `Traditional one model`

For either model type, the `interaction` dataset probably have the same core definition as follows:

| Event | Meaning |
|-------|---------|
| view | Student viewed the scholarship page (weak signal) |
| click | Student clicked on the scholarship (moderate signal) |
| apply | Student applied to the scholarship (strong signal) |
| reject | Student rejected the recommendation (negative signal) |

How does this affect/bias said scholarship against certain student profile is yet to be discussed and most be adjusted based on the next specs below and what is the model type is used.

## What needs to be kept in mind
These are things that need to be accounted when choosing either model type:
1. Must use Tensorflow/Keras with Funcitonal API or subclassing
2. Must implement atleast one custom component. This can be:
    1. Custom layer
    2. Custom loss function
    3. Custom callback
    
    Would be better if we implement that is beneficial for the model, not just sake of having a custom component
3. Model should able to take a student profile, then match it against the scholarships. From this, model should produce:
    1. Ranked array of relevance score, ranging from 0 to 1 for no match to perfect match or decimals in between. Can limit how much scholarships the model should return.

    Additionally, if it possible, and better if so. The model should be able to output also the `compability breakdown`. Each of it also shaped as a score ranging from 0 to 1 for low to high. Followings are the metrics that we consider (and open to feedback):
    1. Academic fit
    2. Leadership
    3. Language
4. Have a feedback loop system, where it can take the `interaction` dataset we gather from user action, then somehow bias the model/datasets to better align the prediction based on the interaction.
5. Monitor the training with TensorBoard
6. Produce a model with metric:
    1. Accuracy ≥ 85%
    2. MAE ≤ 0.10 on validation set

Optionally, below can be implemented if can be done in clean, organic, and simplistic way
1. Implement custom training and evaluation loop using tf.GradientTape
2. Use Generative AI (something like Gemini Flash API, prefer free APIs) to provide textual recommendation for each scored scholarships of what the student needs to do to become better fit.


## Questions for both model type
List of questions I have for both model type:
1. How is the `training` and `inference` code difficulty and complexity?
2. How is the `inference` code portability? Like how much resource sharing is needed between `training` and `inferencing`. Is it only need the final keras model? or it also need another resource that is being generated/used by the `training` code?
3. How to handling cold data? For example:
    1. When there is new `student`. I supppose it's fine? The model can build a representation of the `student` profile based on the trained profiles and make a decision which `scholarships` is best match (CMIIW)
    2. When there is new `scholarship`. I'm confused as how to integrate this into the model and surface it to then gain `interaction` dataset that contain this new `scholarship`
4. The datasets shape. How would it affected by the model type choice? I know with `traditional one model`, the data usually is a single tabular data, and `two tower model` is separated into two. Is it true too in this case? And how about the interaction data? How I consolidate it for re/training? This question will be continued in the next section with how we sync with the web backend developer.
5. Is pair dataset required? I was thinking if the model can at first learn from the datasets feature itself instead needing to have a pair dataset, but still able to generate the match range (0-1). Is it possible?
6. On my past implementation, we have this called Hard Filter that's outside and before the model inference. Is it still needed? Can't we fully rely on the model assesment?

## What needs to be prepared and considered for the live service
1. Serve an API using FastAPI to be called by the web backend team. I wonder:
    1. What endpoints we need to provide for them?
2. Sync between web backend database with how it will be utilized as datasets for the model retraining. The consideration is:
    1. How should the `interaction` data be saved? And later be fetched alongside the `students` and `scholarships` data for retraining?

## What we have right now
Here the list what resources i have right now, i'm open in build everything from the ground up, while using these as the reference point. Especially because i'm avoiding the use of relevance score generation:

### The fields schema

#### Student
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

    # ─── Text Fields (processed within Student Tower) ──────
    personal_statement: str            # Personal statement text
    achievements_narrative: str        # Achievement summary
    future_goals: str                  # Post-study contribution and career goals

    # ─── Funding Preferences ──────────────────────────
    needs_full_funding: bool           # Wants fully-funded scholarship
    can_self_fund_living: bool         # Can cover living expenses independently
```

#### Scholarship
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
    career_track_preference: Optional[CareerTrack]  # None = no preference

    # ─── Service Requirements ─────────────────────────
    requires_return_home_country: bool # Must return home after study

    # ─── Text Fields (processed within Scholarship Tower) ──
    mission_statement: str             # Scholarship mission text
    target_recipient_profile: str      # Ideal candidate description
```

#### Supporting structures

##### LanguageProficiency

```python
@dataclass
class LanguageProficiency:
    test_type: LanguageTest
    score: float      # Test score
    valid_until: Optional[str] = None  # ISO date string
```

##### LanguageRequirement

```python
@dataclass
class LanguageRequirement:
    test_type: LanguageTest
    min_score: float    # Minimum required score
    is_mandatory: bool = True  # True = hard filter, False = preferred but not required
```

##### FundingCoverage

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

##### SelectionCriteria

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

#### Enum values

##### DegreeLevel

| Value | Description |
|-------|-------------|
| `high_school` | High school |
| `bachelors` | Bachelor's degree |

##### Country

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

##### HighSchoolTrack

| Value | Description |
|-------|-------------|
| `science` | Science track (Physics, Chemistry, Biology) |
| `social_studies` | Social Studies track (Economics, Geography, Sociology) |
| `languages` | Languages & Literature track |
| `religion` | Religious studies track |
| `vocational` | Vocational track |

##### OlympiadLevel

| Value | Description |
|-------|-------------|
| `none` | Never competed |
| `school` | School-level (intra-school) |
| `city` | City / district level |
| `provincial` | Provincial level |
| `national` | National level |
| `international` | International (IMO, IPhO, IOI, etc.) |

##### OlympiadSubject

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

##### IncomeCategory

| Value | Description |
|-------|-------------|
| `very_low` | Very low income |
| `low` | Low income |
| `middle` | Middle income |
| `upper_middle` | Upper middle income |
| `high` | High income |

##### SchoolTier

| Value | Description |
|-------|-------------|
| `excellence` | Top-tier / boarding school |
| `public_a` | Public school, accredited A |
| `private_a` | Private school, accredited A |
| `accredited_b` | Public or private, accredited B |
| `accredited_c` | Accredited C schools |
| `unaccredited` | Not yet accredited |
| `unknown` | Accreditation status unknown |

##### MajorField

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

##### CareerTrack

| Value | Description |
|-------|-------------|
| `academic` | Academic/research career |
| `industry` | Industry/tech career |
| `government` | Government service |
| `ngo_npo` | NGO/NPO sector |
| `entrepreneurship` | Entrepreneurship |
| `public_service` | Public service |

##### LanguageTest

| Value | Description |
|-------|-------------|
| `toefl` | TOEFL (English) |
| `ielts` | IELTS (English) |
| `topik` | TOPIK (Korean) |
| `jlpt` | JLPT (Japanese) |
| `delf` | DELF (French) |
| `hsk` | HSK (Chinese) |

##### HostRegion

| Value | Description |
|-------|-------------|
| `asia` | Asia |
| `europe` | Europe |
| `north_america` | North America |
| `south_america` | South America |
| `africa` | Africa |
| `oceania` | Oceania |


### The generator base modules `src/`
So i have this set up for generator i've done before. It i think matches with the defined fields schema above. But you are welcome to compare and correct me. REMEMBER, use this as point of reference. You can absolutely reuse this IF it's aligned with whatever the plan you will form from my questions above and what i confirm decide to use after your response.

#### Module Index

| File | Role |
|---|---|
| `data/scholarships.py` | ~18 hardcoded scholarship profiles |
| `data/students.py` | Seed data (countries, schools, templates) |
| `schemas/enums.py` | Enums + multilingual normalization maps |
| `schemas/scholarship.py` | Scholarship, FundingCoverage, SelectionCriteria |
| `schemas/student.py` | Student profile dataclass |

#### data/scholarships.py — Scholarship Profiles

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

#### data/students.py — Seed Data

- **COUNTRIES_BY_REGION** — 6 regions, ~30 countries total; used by scorer for location-fit computation
- **SCHOOL_NAMES** — Per-country school name pools
- **PERSONAL_STATEMENT_TEMPLATES / FUTURE_GOALS_TEMPLATES** — With `{field}`, `{interest}`, `{country}` placeholders
- **RESEARCH_INTERESTS** — 12 research topics for synthetic generation
- **ACHIEVEMENT_TEMPLATES** — Olympiad, competition, volunteer patterns

#### schemas/enums.py — Domain Types

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

#### schemas/scholarship.py — Domain Model

- **LanguageRequirement** — test_type, min_score, is_mandatory
- **FundingCoverage** — 4 boolean flags + monthly_stipend; properties: `is_full_funding`, `coverage_count`
- **SelectionCriteria** — academic, leadership, olympiad, extracurricular, essay (weights sum to 1.0)
- **Scholarship** — aggregates all constraints + selection_criteria + funding + career_preference + text fields (`mission_statement`, `target_recipient_profile`) for two-tower text similarity

#### schemas/student.py — Student Profile

Identity → Demographics → Academic (GPA, math, English, major avg) → Language proficiency list → Olympiad (level + subjects) → Leadership/volunteer/competition counts → School tier + income → Career intent + target countries → Text fields (`personal_statement`, `achievements_narrative`, `future_goals`) → Funding preferences.