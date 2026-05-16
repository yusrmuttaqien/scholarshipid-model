# Schema & Pipeline Migration Notes

**From**: Two-Tower v0 (original notebook)  
**To**: Scholarship Lab Two-Tower v0.1 (schema_spec.md)  
**Date**: 2026-05-10

---

## 1. Schema Comparison

### 1.1 Feature Mapping Table

| Category | Old Field | New Field | Change Type | Notes |
|----------|-----------|-----------|-------------|-------|
| **Identity** | `student_id` | `student_id` | Unchanged | — |
| | `name` | *(removed)* | Removed | Not needed for matching logic |
| **Nationality** | *(none)* | `nationality` (student) ↔ `eligible_nationalities` (scholarship) | **Added** | Critical hard filter — was completely missing |
| **Age** | `age` | `age` ↔ `min_age`, `max_age` | **Enhanced** | Old: age was a raw feature fed into NN. New: hard filter with explicit min/max bounds |
| **Degree Level** | *(none)* | `current_degree_level`, `target_degree_level` ↔ `eligible_degree_levels` | **Added** | Hard filter — prevents recommending S3 scholarships to SMA students |
| **Field of Study** | `field_of_study` (string) | `current_major_field` (enum) ↔ `eligible_fields` (list[enum]) | **Enhanced** | Old: raw string with LabelEncoder. New: controlled vocabulary enum, multi-value support |
| | | `eligible_majors_specific` (optional) | **Added** | For scholarships that target specific majors within a field |
| **University** | *(none)* | `current_university`, `current_university_tier` | **Added** | Contextual signal for soft scoring |
| **GPA** | `gpa` (single value) | `current_gpa_4scale` + `current_gpa_percentage` ↔ `min_gpa_4scale` + `min_gpa_percentage` | **Enhanced** | Old: single GPA, ambiguous scale. New: dual-scale support (4.0 and percentage), each with explicit minimums |
| **SMP Score** | `smp_score` | *(removed)* | Removed | Absorbed into general GPA framework; SMP students use `current_gpa_percentage` |
| **TOEFL** | `toefl` (single score) | `language_proficiency[]` ↔ `language_requirements[]` | **Enhanced** | Old: single TOEFL score. New: multi-language, multi-test support (TOEFL, IELTS, Duolingo, TOPIK, HSK) with mandatory/optional flags |
| **Recommendation Letters** | `rec_letters` (count) | `min_recommendation_letters` | Unchanged | Kept as integer threshold |
| **Personal Statement** | `has_ps` (boolean) | `personal_statement` | **Enhanced** | Old: binary yes/no. New: actual text for TF-IDF similarity scoring |
| **Funding** | *(none)* | `needs_full_funding`, `can_self_fund_living` ↔ `funding_coverage{}` (structured) | **Added** | Student funding needs matched against scholarship coverage (tuition, living, airfare, insurance) |
| **Experience** | *(none)* | `research_experience_count`, `leadership_experience_count`, `volunteer_experience_count`, `competition_wins_count` | **Added** | Rich experience profile matched against scholarship selection criteria weights |
| **Career Intent** | *(none)* | `intended_career_track` ↔ `career_track_preference` | **Added** | Soft scoring: student career goals aligned with scholarship target profile |
| **Return Home** | *(none)* | `willing_to_return_home` ↔ `requires_return_home_country` | **Added** | Hard filter — prevents recommending scholarships requiring home return to students unwilling to do so |
| **Target Countries** | *(none)* | `target_countries` ↔ `host_country`, `host_region` | **Added** | Student country preferences matched against scholarship host locations |
| **Selection Criteria** | *(none)* | `selection_criteria{}` (weights) | **Added** | Scholarship specifies what it values (academic, leadership, research, extracurricular, essay) — enables weighted soft scoring |
| **Text Fields** | *(none)* | TF-IDF text pairs: personal_statement ↔ mission_statement, achievements_narrative ↔ target_recipient_profile | **Added** | Text similarity as bonus scoring layer |

### 1.2 Summary of Changes

| Metric | Old Schema | New Schema | Change |
|--------|-----------|-----------|--------|
| Student features | 7 | ~20 | +185% |
| Scholarship features | 6 | ~20 | +233% |
| Hard filter dimensions | 0 | 7 | **New capability** |
| Soft scoring dimensions | 7 (all NN) | ~15 (NN + TF-IDF) | +114% |
| Language support | TOEFL only | 7+ languages, multiple tests | **New capability** |
| Text similarity | None | 2 text pairs | **New capability** |
| Feedback loop | None | Implicit feedback (apply/click/view/reject) | **New capability** |

### 1.3 Dataclasses Reference

All data structures are defined in `generator.py` using Python `@dataclass`:

**Student**: ~20 fields including nationality, GPA (dual scale), language proficiency, experience counts, career track, target countries, text fields (personal_statement, achievements_narrative, research_interest), funding preferences.

**Scholarship**: ~20 fields including eligibility constraints (nationalities, age range, degree levels, fields), GPA requirements, language requirements (with mandatory/optional flag), selection criteria weights, structured funding coverage, career track preference, text fields (mission_statement, target_recipient_profile).

**Supporting structures**:
- `LanguageProficiency`: test_type, score, valid_until
- `LanguageRequirement`: test_type, min_score, **is_mandatory**
- `FundingCoverage`: covers_tuition, covers_living_expense, covers_airfare, covers_insurance, total_amount
- `SelectionCriteria`: academic, leadership, research, extracurricular, essay (weights sum to 1.0)
- `Pair`: student_id, scholarship_id, label, relevance_score, **timestamp** (for time-based splitting)
- `Feedback`: student_id, scholarship_id, feedback_type, timestamp, weight (derived)

---

## 2. Pipeline Architecture Improvements

### 2.1 Old Pipeline (Single-Stage)

```
Raw Features → Preprocessing → Two-Tower NN → Cosine Similarity → Rankings
```

**Problems:**
- Neural network had to learn hard eligibility rules (e.g., "GPA must be above threshold") as soft signals
- No nationality or degree-level filters — could recommend S3 scholarships to SMA students
- Single TOEFL score — couldn't handle IELTS, TOPIK, etc.
- `has_ps` was binary — no text quality assessment
- `bonded` was a raw feature — the NN learned it as a preference signal rather than a hard constraint
- No separation between "eligible" and "good fit"

### 2.2 New Pipeline (Three-Stage)

```
┌─────────────────────────────────────────────┐
│  Stage 1: Hard Filter (Deterministic)       │
│  ─────────────────────────────────────────   │
│  • Nationality in eligible_nationalities    │
│  • Target degree in eligible_degree_levels  │
│  • Age within [min_age, max_age]           │
│  • GPA meets minimum (correct scale)        │
│  • Mandatory language requirements met      │
│  • Bond willingness check                   │
│  • Return-home willingness check            │
│                                             │
│  Output: Eligible (student, scholarship)    │
│         pairs only                          │
└──────────────────┬──────────────────────────┘
                   ↓
┌─────────────────────────────────────────────┐
│  Stage 2: Two-Tower Neural Network          │
│  ─────────────────────────────────────────   │
│  Student Tower Input:                        │
│    • GPA (dual scale), language scores      │
│    • Experience counts (research, leadership │
│      volunteer, competition)                │
│    • Career track, target field             │
│    • Funding needs                          │
│                                             │
│  Scholarship Tower Input:                   │
│    • Selection criteria weights             │
│    • Funding coverage                       │
│    • Career track preference                │
│                                             │
│  Output: Soft similarity score [-1, 1]      │
└──────────────────┬──────────────────────────┘
                   ↓
┌─────────────────────────────────────────────┐
│  Stage 3: TF-IDF Text Similarity (Bonus)   │
│  ─────────────────────────────────────────   │
│  • personal_statement ↔ mission_statement   │
│  • achievements_narrative ↔ target_profile  │
│  • research_interest ↔ mission_statement    │
│                                             │
│  Output: Text bonus score added to NN score │
└──────────────────┬──────────────────────────┘
                   ↓
┌─────────────────────────────────────────────┐
│  Final Ranked Recommendations per Student   │
└─────────────────────────────────────────────┘
```

### 2.3 Why Three-Stage is Better

| Aspect | Old (Single NN) | New (Three-Stage) |
|--------|----------------|-------------------|
| **Correctness** | NN might recommend ineligible scholarships (low score but still ranked) | Hard filter guarantees only eligible scholarships reach ranking |
| **Interpretability** | "Why was this recommended?" — NN is a black box | Clear pipeline: eligible → good fit → text match bonus |
| **Debugging** | If wrong recommendation, unclear which feature caused it | Each stage can be inspected independently |
| **Training Data** | NN must learn everything from limited data (30 students × 5 scholarships) | Hard filter removes impossible pairs; NN trains only on eligible pairs with richer signals |
| **Scalability** | Adding new eligibility rules requires retraining | New hard filters are pure logic changes, no retraining needed |
| **Text Features** | Not supported | TF-IDF bonus layer adds qualitative matching without NN complexity |

---

## 3. Data Preprocessing Changes

### 3.1 What Changes in Preprocessing

| Step | Old Approach | New Approach |
|------|-------------|--------------|
| **GPA Scaling** | Single StandardScaler on raw GPA | Dual-scale: normalize 4-scale and percentage separately, use whichever scholarship expects |
| **Language Scores** | Single TOEFL column | Multi-column: extract scores per language+test_type, mask missing ones |
| **Categorical Encoding** | LabelEncoder on free-text strings | Enum-to-integer mapping on controlled vocabularies (no LabelEncoder drift risk) |
| **Missing Values** | Fill with median/0 | Explicit masking: `null` means "not applicable" (e.g., S3 student has no SMP score), not "missing data" |
| **Text Fields** | Not present | TF-IDF vectorization on raw text fields |
| **Funding** | Not present | Structured FundingCoverage with boolean flags + total amount |

### 3.2 New Preprocessing Pipeline Pseudocode

```python
# Student preprocessing
def preprocess_student(student):
    features = {}
    
    # GPA: use whichever scale is available
    if student.current_gpa_4scale is not None:
        features['gpa_4scale'] = normalize_0_to_4(student.current_gpa_4scale)
    elif student.current_gpa_percentage is not None:
        features['gpa_pct'] = normalize_0_to_100(student.current_gpa_percentage)
    
    # Language proficiency: flatten list to fixed-width features
    for lang in ['English', 'Korean', 'Chinese', 'Japanese']:
        score = get_language_score(student, lang)
        features[f'lang_{lang}'] = score if score is not None else 0.0
        features[f'lang_{lang}_has'] = 1.0 if score is not None else 0.0
    
    # Experience counts
    features['research_exp'] = student.research_experience_count
    features['leadership'] = student.leadership_experience_count
    features['volunteer'] = student.volunteer_experience_count
    features['competition'] = student.competition_wins_count
    
    # Boolean flags
    features['needs_full_funding'] = int(student.needs_full_funding)
    features['can_self_fund_living'] = int(student.can_self_fund_living)
    features['willing_to_return'] = int(student.willing_to_return_home)
    
    # Enum features (one-hot or ordinal encode)
    features['career_track'] = encode_enum(student.intended_career_track)
    features['major_field'] = encode_enum(student.current_major_field)
    
    # Text features (for TF-IDF, not for NN)
    features['texts'] = [student.personal_statement,
                          student.achievements_narrative,
                          student.research_interest]
    
    return features

# Scholarship preprocessing (symmetric structure)
def preprocess_scholarship(scholarship):
    features = {}
    
    # Selection criteria weights (already 0-1, sum to 1.0)
    features['weight_academic'] = scholarship.selection_criteria.academic
    features['weight_leadership'] = scholarship.selection_criteria.leadership
    features['weight_research'] = scholarship.selection_criteria.research
    features['weight_extracurricular'] = scholarship.selection_criteria.extracurricular
    features['weight_essay'] = scholarship.selection_criteria.essay
    
    # Funding coverage (structured)
    features['covers_tuition'] = int(scholarship.funding_coverage.covers_tuition)
    features['covers_living'] = int(scholarship.funding_coverage.covers_living_expense)
    features['covers_airfare'] = int(scholarship.funding_coverage.covers_airfare)
    features['covers_insurance'] = int(scholarship.funding_coverage.covers_insurance)
    features['funding_total_amount'] = normalize(scholarship.funding_coverage.total_amount)
    features['is_full_funding'] = int(scholarship.funding_coverage.is_full_funding)
    
    # Career track preference
    features['career_track'] = encode_enum(scholarship.career_track_preference)
    
    # Text features (for TF-IDF, not for NN)
    features['texts'] = [scholarship.mission_statement,
                          scholarship.target_recipient_profile]
    
    return features
```

---

## 4. Hard Filter Logic

### 4.1 Eligibility Rules

The hard filter runs **before** the neural network and removes infeasible (student, scholarship) pairs:

```python
def is_eligible(student, scholarship):
    """
    Returns True if student is eligible for scholarship.
    All checks are deterministic — no learned parameters.
    """
    
    # 1. Nationality check
    if student.nationality not in scholarship.eligible_nationalities:
        return False, "nationality"
    
    # 2. Degree level check
    if student.target_degree_level not in scholarship.eligible_degree_levels:
        return False, "degree_level"
    
    # 3. Age range check
    if scholarship.min_age and student.age < scholarship.min_age:
        return False, "min_age"
    if scholarship.max_age and student.age > scholarship.max_age:
        return False, "max_age"
    
    # 4. GPA check (use matching scale)
    if scholarship.min_gpa_4scale is not None:
        if student.current_gpa_4scale is None or student.current_gpa_4scale < scholarship.min_gpa_4scale:
            return False, "gpa_4scale"
    elif scholarship.min_gpa_percentage is not None:
        if student.current_gpa_percentage is None or student.current_gpa_percentage < scholarship.min_gpa_percentage:
            return False, "gpa_percentage"
    
    # 5. Language requirements (mandatory ones only)
    for lang_req in scholarship.language_requirements:
        if not lang_req.is_mandatory:
            continue  # Preferred but not required — skip hard filter
        student_score = get_student_language_score(student, lang_req.test_type)
        if student_score is None or student_score < lang_req.min_score:
            return False, f"language_{lang_req.test_type}"
    
    # 6. Return home country
    if scholarship.requires_return_home_country and not student.willing_to_return_home:
        return False, "return_home"
    
    return True, "eligible"
```

### 4.2 Hard Filter Output

Each filtered pair gets a rejection reason for auditability:

```
Student S001 × Scholarship A: REJECTED (reason: nationality)
Student S001 × Scholarship B: REJECTED (reason: gpa_4scale)
Student S001 × Scholarship C: ELIGIBLE  → passed to NN
Student S002 × Scholarship A: REJECTED (reason: max_age)
Student S002 × Scholarship C: ELIGIBLE  → passed to NN
```

---

## 5. Migration Checklist

- [x] Create new CSV/JSON data files matching the v0.1 schema (generator.py + generator_single_pool.py updated)
- [x] Define dataclasses with proper enums (Student, Scholarship, Pair, Feedback)
- [x] Implement hard filter module (`PositivePairGenerator.check_hard_filter`)
- [ ] Implement two-tower model with TensorFlow Functional API
- [ ] Update model input features to match new schema
- [ ] Implement TF-IDF text similarity layer
- [ ] Integrate three-stage pipeline (hard filter → NN → TF-IDF bonus)
- [ ] Implement custom loss function (Batch All-Negative Softmax Loss)
- [ ] Implement custom callback (FeedbackRetrainCallback for online learning)
- [ ] Build data pipeline to load CSVs into TensorFlow Dataset
- [ ] Deploy with feedback loop integration
- [ ] Re-train and validate model on new dataset
- [ ] Evaluate with Recall@K, NDCG@K, MRR metrics

## 6. Model Output Architecture (Brief Requirement #5)

The two-tower model returns a **probability score** for each (student, scholarship) pair:

```
Student Features → Student Tower → Embedding (d=64-128)
                                     ↓
Cosine Similarity → Sigmoid Activation → Probability [0, 1]
                                     ↑
Scholarship Features → Scholarship Tower → Embedding (d=64-128)
```

**Output interpretation**:
- Near 1.0 = high probability of good match
- Near 0.0 = low probability of match
- Scores are ranked per student to produce ordered recommendations

**Why cosine similarity + sigmoid?**
- Cosine similarity produces [-1, 1] range
- Sigmoid squashes to [0, 1], giving interpretable probability
- During training: batch all-negative sampling for efficient contrastive learning
- During inference: cosine sim over all eligible scholarships → sigmoid → rank by probability

## 7. Feedback Loop Architecture (Brief Requirement #2)

The system supports implicit feedback from students on recommended scholarships:

| Feedback Type | Weight | Description |
|---------------|--------|-------------|
| `apply` | 3.0 | Strong positive (applied to scholarship) |
| `click` | 2.0 | Soft positive (clicked on recommendation) |
| `view` | 1.0 | Weak positive (saw in recommendations) |
| `reject` | -1.0 | Explicit negative (swiped away / not interested) |

**Feedback loop pipeline**:
```
Student Interaction → Log Feedback → Convert to Training Pairs → Fine-tune Model → Deploy
```

**Implementation**:
- `FeedbackGenerator` creates synthetic feedback for training
- Feedback records saved to one flat `feedback.csv`, split by timestamp in code
- Weights determine signal strength during online retraining
- Custom callback (`FeedbackRetrainCallback`) triggers model updates when new feedback accumulates

---

## 10. Key Design Decisions & Rationale

### Why Hard Filter Before NN?
Eligibility is a **logical constraint**, not a learned preference. A scholarship that requires Indonesian nationality should never appear for an American student — even with a low score. Neural networks are bad at hard constraints; they produce probabilistic outputs where impossible options just get low scores but can still surface.

### Why TF-IDF as a Separate Layer?
Text similarity is conceptually different from numerical feature matching. TF-IDF cosine similarity is simple, interpretable, and doesn't require training data. Keeping it separate means:
- You can tune the text bonus weight independently
- It works even with small datasets (no overfitting risk)
- The NN focuses on what it's good at: learning non-linear relationships between numerical/categorical features

### Why Dual GPA Scales?
Indonesian students may have percentage-based GPAs (0-100) while international scholarships expect 4.0 scale. Supporting both prevents data conversion errors and lets each scholarship specify which scale it expects.

### Why Enum Controlled Vocabularies?
The old schema used raw strings for fields of study, which caused LabelEncoder drift (new categories in test data that weren't in training data). Enums prevent this by enforcing a fixed vocabulary with an "other/none" catch-all.

### Why `is_mandatory` on LanguageRequirements?
Not all language requirements are equally strict. Some scholarships say "TOEFL ≥ 80 required" while others say "IELTS preferred but not required." The `is_mandatory` flag ensures:
- Mandatory requirements are enforced by hard filter (Stage 1)
- Optional/preferred requirements can influence soft scoring without blocking eligibility
- More realistic representation of real scholarship policies

### Why Structured FundingCoverage?
A single float for funding amount loses important information. The structured `FundingCoverage` dataclass captures:
- Which aspects are covered (tuition, living, airfare, insurance)
- Whether it's full funding (covers both tuition AND living)
- Total monetary value for comparison
- Enables matching student `needs_full_funding` against scholarship coverage

### Why Implicit Feedback Instead of Explicit Ratings?
Explicit ratings (1-5 stars) are unrealistic for scholarship applications. Implicit feedback is more natural:
- `apply` = strongest positive signal (student applied to scholarship)
- `click` = soft positive (student clicked on recommendation)
- `view` = weak positive (student saw in recommendations)
- `reject` = explicit negative (student swiped away)

### Why SMA-Level Students?
Per project brief requirement #1, the system targets high school students at the SMA (Sekolah Menengah Atas) educational level — equivalent to grades 10-12. "SMA" refers to the **Indonesian education system**, not nationality. Students can be from any country.

All generated students have:
- `current_degree_level = "SMA"`
- Age range 16-18
- Mixed nationalities (Indonesia, Malaysia, Thailand, Philippines, Vietnam, Singapore, Japan, South Korea, China, India, and others)
- Target degree level S1 (Bachelor's) or S2 (Master's) for direct-to-graduate scholarships
- All scholarships include SMA in `eligible_degree_levels`

---

## 9. Dataset Generation Approaches

Two generators are available, both using identical dataclasses from `generator.py`:

### Approach A: Separate Pools Per Split (`generator.py`)

```
datasets/
├── train/       # 14,000 students + 560 scholarships
├── val/          # 3,000 students + 120 scholarships
└── test/         # 3,000 students + 120 scholarships
```

**Use case:** Traditional ML where each split has completely separate entities.

### Approach B: Single Pool + Time-Based Splitting (`generator_single_pool.py`)

```
datasets_single_pool/
├── students.csv          # 20,000 students (shared across all splits)
├── scholarships.csv      # 800 scholarships (shared across all splits)
├── pairs.csv             # ALL pairs with timestamps
└── feedback.csv          # ALL feedback with timestamps
```

**Time-based splitting (done in code):**
```python
pairs = pd.read_csv("pairs.csv")
pairs = pairs.sort_values("timestamp")

train_pairs = pairs.iloc[:int(len(pairs) * 0.7)]
val_pairs = pairs.iloc[int(len(pairs) * 0.7):int(len(pairs) * 0.85)]
test_pairs = pairs.iloc[int(len(pairs) * 0.85):]
```

**Why Approach B is preferred for two-tower models:**
- Model learns embeddings for ALL students during training
- Val/test evaluate on unseen interactions from known students
- Matches production reality: known student, predict over all scholarships
- Same (student, scholarship) pair belongs to EXACTLY ONE split
- Same student can appear across splits with DIFFERENT scholarships (correct behavior)

### Key Difference Summary

| Aspect | Approach A | Approach B |
|--------|-----------|------------|
| Students | Different per split | Shared across splits |
| Scholarships | Different per split | Shared across splits |
| Pairs | Separate files per split | One file, split by timestamp |
| Production match | Low (students unseen in val/test) | High (known students, unseen interactions) |