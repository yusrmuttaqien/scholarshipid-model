# Two-Tower Recommendation System — Model Building Guide

Step-by-step specification for building the two-tower neural network using TensorFlow/Keras Functional API.

---

## Part A: Architecture Specification

### 1. Overview

The system uses a **two-tower neural network** architecture to predict a continuous relevance score (0.0–1.0) between a student and a scholarship. Each tower independently encodes its input into a fixed-size embedding vector, and the final similarity is computed via cosine similarity followed by sigmoid activation.

**Why two-tower?**
- Student tower produces one embedding per student, independent of scholarships
- Scholarship tower produces one embedding per scholarship, independent of students
- At inference time, any student can be scored against any number of scholarships in a single batch (broadcast inference)
- Enables efficient ranking: compute all scores, sort descending, return top-K

### 2. Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `EMBEDDING_DIM` | 64 | Sweet spot for 20K students / 800 scholarships |
| Student/Scholarship tower hidden units | 128 → 64 | Progressive dimensionality reduction |
| `LEARNING_RATE` | 0.001 | Adam optimizer default, with warmup + decay via ReduceLROnPlateau |
| `BATCH_SIZE` | 2048 | Large batch for CPU efficiency on M1 |
| `EPOCHS` | 50 (with EarlyStopping) | Patience=5, min_delta=0.001 |
| Loss | WeightedMSE (custom) | Balanced gradient across 3 relevance classes |
| Metrics | RMSE, MAE | Regression target evaluation |

### 3. Student Tower

The student tower encodes a student profile into a **64-dimensional embedding**. Input features are grouped into four categories:

#### Categorical Features (Embedding Layers)
Each categorical feature is passed through a `StringLookup` layer followed by an `Embedding` layer. The embedding dimension follows the `sqrt(cardinality)` rule of thumb, capped at 16.

| Feature | Cardinality | Embedding Dim |
|---------|-------------|---------------|
| nationality | 27 countries | 8 |
| high_school_track | 5 tracks | 4 |
| school_tier | 7 tiers | 4 |
| family_income_category | 5 categories | 4 |
| intended_career_track | 6 tracks | 4 |
| olympiad_level | 6 levels | 4 |

#### Numerical Features (Normalized)
All numerical features are standardized (zero mean, unit variance) using training set statistics.

| Feature | Range |
|---------|-------|
| age | 16–18 |
| overall_report_card_average | 0–100 |
| math_score | 0–100 |
| english_score | 0–100 |
| major_subject_average | 0–100 |
| leadership_experience_count | 0+ (Poisson) |
| volunteer_experience_count | 0+ (Poisson) |
| competition_wins_count | 0+ (Poisson) |

#### Boolean Features (Binary Float)
Each boolean is cast to float (0.0 / 1.0).

| Feature | Description |
|---------|-------------|
| willing_to_return_home | Willing to return home after study |
| from_underrepresented_region | From underrepresented regions |
| needs_full_funding | Wants fully-funded scholarship |
| can_self_fund_living | Can cover living expenses independently |

#### Language Proficiency Features (Fixed-Size Vector)
The JSON list of language tests is parsed into a fixed-size vector. For each of the 6 supported tests (TOEFL, IELTS, TOPIK, JLPT, DELF, HSK), two features are extracted: **maximum score** (if taken) and **has_test** (binary flag). This produces a 12-dimensional vector.

#### Tower Architecture
```
All Features → Concatenate → Dense(128, relu) → BatchNorm → Dense(64, relu) → BatchNorm → Student Embedding
```

### 4. Scholarship Tower

The scholarship tower encodes a scholarship profile into a **64-dimensional embedding**.

#### Categorical Features (Embedding Layers)

| Feature | Cardinality | Embedding Dim |
|---------|-------------|---------------|
| host_region | 6 regions | 4 |
| preferred_school_tier | 7 tiers | 4 |
| career_track_preference | 7 options (6 tracks + None) | 4 |
| max_family_income_category | 5 categories | 4 |

#### Numerical Features (Normalized)

| Feature | Range |
|---------|-------|
| min_age | 15–17 |
| max_age | 18–23 |
| min_report_card_average | 0–100 |
| min_major_subject_average | 0–100 |
| funding_monthly_stipend | 0–2000+ |
| funding_coverage_count | 0–4 |

#### Boolean Features (Binary Float)

| Feature | Description |
|---------|-------------|
| requires_financial_need | Requires low-income background |
| requires_return_home_country | Must return home after study |
| funding_covers_tuition | Covers tuition fees |
| funding_covers_living | Covers living expenses |
| funding_covers_airfare | Covers airfare |
| funding_covers_insurance | Covers insurance |
| funding_is_full_funding | Covers both tuition AND living |

#### List Fields (Binary Encoding)
List fields such as `eligible_nationalities`, `eligible_fields`, and `eligible_high_school_tracks` are encoded as fixed-size binary vectors. Each possible value gets a slot in the vector, set to 1.0 if present or 0.0 if absent. The combined encoding produces a **32-dimensional vector**.

#### Tower Architecture
```
All Features → Concatenate → Dense(128, relu) → BatchNorm → Dense(64, relu) → BatchNorm → Scholarship Embedding
```

### 5. Connection Layer: Cosine Similarity

A custom layer computes cosine similarity between the student and scholarship embeddings, then applies sigmoid to map the result to [0, 1].

**Process:**
1. L2-normalize both embedding vectors
2. Compute element-wise product and sum → scalar cosine similarity
3. Apply sigmoid activation → final relevance score in [0, 1]

**Why sigmoid?** Cosine similarity naturally ranges from [-1, 1]. Sigmoid maps this to [0, 1], aligning with the target relevance_score range.

### 6. Loss Function: WeightedMSE

Custom loss function that applies class-aware weighting to prevent the model from ignoring minority classes.

| Class | Relevance Range | Weight | Rationale |
|-------|-----------------|--------|-----------|
| Match | ≥ 0.7 | 1.0 | Baseline weight |
| In-Between | 0.3 – 0.7 | 1.5 | Harder to predict, needs stronger gradient |
| Not Match | < 0.3 | 1.0 | Baseline weight |

### 7. Custom Callback: ClassDistribution

Custom callback that tracks per-class RMSE and MAE during training. After each epoch, it evaluates predictions on the validation set and prints metrics broken down by relevance class. This provides visibility into which classes the model struggles with.

---

## Part B: Implementation Guide

### 1. File Structure

```
project/
├── data.py          # Data loading pipeline (tf.data.Dataset)
├── model.py         # Model definition (towers, layers, assembly)
├── train.py         # Training script (model.fit with callbacks)
├── inference.py     # Inference pipeline (3-stage: filter → NN → text bonus)
└── feedback.py      # Feedback capture and retraining helper
```

### 2. Data Pipeline (`data.py`)

**Purpose:** Load CSVs and produce `tf.data.Dataset` objects for efficient training.

**Steps:**
1. Read `students.csv`, `scholarships.csv`, `pairs.csv` using pandas
2. Parse JSON columns (language_proficiency, olympiad_subjects, eligible_nationalities, etc.)
3. Apply integer encoding to categorical features using `StringLookup` layers
4. Standardize numerical features using training set mean/std
5. Build feature dictionaries for student and scholarship inputs
6. Create `tf.data.Dataset` objects with batching and shuffling

**Train/Val/Test Split:** Time-based split using the `timestamp` column. Sort pairs by timestamp, then assign 70% train, 15% validation, 15% test.

### 3. Model Definition (`model.py`)

**Purpose:** Define the two-tower architecture and custom components.

**Exports:**
- `CosineSimilarity` — Custom Keras layer for cosine similarity + sigmoid
- `WeightedMSE` — Custom loss function with class-aware weighting
- `ClassDistributionCallback` — Custom callback for per-class metrics
- `build_student_tower()` — Returns student tower model and input specs
- `build_scholarship_tower()` — Returns scholarship tower model and input specs
- `build_model()` — Assembles full two-tower model, compiles with Adam optimizer

### 4. Training Procedure (`train.py`)

**Purpose:** Train the model with proper callbacks and monitoring.

**Callbacks:**
- `EarlyStopping(monitor="val_loss", patience=5, min_delta=0.001, restore_best_weights=True)`
- `ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6)`
- `ClassDistributionCallback()` — Per-class metrics tracking

**Expected Training Output:**
```
Epoch 1/50: loss=0.12, val_loss=0.11, mse=0.10
  Epoch 1 per-class metrics:
    match: RMSE=0.15, MAE=0.12
    in_between: RMSE=0.18, MAE=0.14
    not_match: RMSE=0.10, MAE=0.08
...
Epoch 12/50: loss=0.04, val_loss=0.05, mse=0.04  ← EarlyStopping triggers
Restoring best model weights.
```

**Performance Targets (from POINTS_TO_REMEMBER.md):**
- Accuracy ≥ 85% (binary threshold at 0.5)
- MAE ≤ 0.10 on validation set

### 5. Evaluation Criteria

After training, evaluate on the held-out test set:

| Metric | Target |
|--------|--------|
| Overall RMSE | ≤ 0.15 |
| Overall MAE | ≤ 0.10 |
| Per-class MAE (Match) | ≤ 0.12 |
| Per-class MAE (In-Between) | ≤ 0.15 |
| Per-class MAE (Not Match) | ≤ 0.10 |

**Calibration Check:** Plot predicted vs actual scores. Points should cluster near the diagonal line y=x. Systematic deviation in one direction indicates poor calibration.

### 6. Inference Pipeline (`inference.py`)

The inference pipeline applies three stages to produce final ranked recommendations for a given student:

#### Stage 1: Hard Filter (Deterministic)
Filter scholarships using deterministic eligibility checks. A scholarship passes only if ALL conditions are met:

| Check | Condition |
|-------|-----------|
| Nationality | Student nationality is in `eligible_nationalities` |
| Age | Student age is within `[min_age, max_age]` |
| Degree Level | Student target degree is in `eligible_degree_levels` |
| Report Card | Student overall average ≥ `min_report_card_average` |
| Major Subject | Student major average ≥ `min_major_subject_average` |
| Language | Student has required test scores for all mandatory language requirements |
| Return Home | If scholarship requires return, student must be willing |
| Financial Need | If scholarship requires financial need, student income ≤ `max_family_income_category` |

**Output:** List of eligible scholarships only.

#### Stage 2: Two-Tower Neural Network
For each eligible scholarship, compute a soft similarity score using the trained two-tower model.

**Process:**
1. Encode student features → student embedding (d=64)
2. Encode all eligible scholarship features → scholarship embeddings (batch × d=64)
3. Compute cosine similarity + sigmoid → relevance scores
4. Sort by predicted score descending

**Output:** Ranked list of eligible scholarships with NN scores.

#### Stage 3: Text Similarity Bonus
Apply TF-IDF text similarity as a bonus to the NN score.

**Pairs Compared:**
| Student Field | Scholarship Field |
|---------------|-------------------|
| personal_statement | mission_statement |
| future_goals | target_recipient_profile |

**Process:**
1. For each pair, compute TF-IDF cosine similarity
2. Combine into a bonus score (0.0–0.1 range)
3. Add bonus to NN score, clamp to [0, 1]

**Final Output:** Top-K recommendations ranked by final score.

### 7. Feedback Loop (`feedback.py`)

**Purpose:** Capture student feedback and prepare data for model retraining.

**Feedback Types and Weights:**

| Type | Weight | Signal |
|------|--------|--------|
| apply | 3.0 | Strong positive — student applied |
| click | 2.0 | Soft positive — student clicked |
| view | 1.0 | Weak positive — student saw in recommendations |
| reject | -1.0 | Explicit negative — student swiped away |

**Retraining Process:**
1. Collect feedback records since last training
2. Generate new pairs from feedback data
3. Assign relevance scores based on feedback weight and current NN prediction
4. Merge with existing training data
5. Retrain model (or fine-tune) with updated dataset

---

## Appendix: Complete Feature Reference

### Student Input Features

| Category | Feature Count | Details |
|----------|---------------|---------|
| Categorical (Embedding) | 6 features | nationality, high_school_track, school_tier, family_income_category, intended_career_track, olympiad_level |
| Numerical (Normalized) | 8 features | age, overall_report_card_average, math_score, english_score, major_subject_average, leadership_experience_count, volunteer_experience_count, competition_wins_count |
| Boolean (Float) | 4 features | willing_to_return_home, from_underrepresented_region, needs_full_funding, can_self_fund_living |
| Language (Fixed Vector) | 12 features | 6 tests × (score + has_test flag) |

### Scholarship Input Features

| Category | Feature Count | Details |
|----------|---------------|---------|
| Categorical (Embedding) | 4 features | host_region, preferred_school_tier, career_track_preference, max_family_income_category |
| Numerical (Normalized) | 6 features | min_age, max_age, min_report_card_average, min_major_subject_average, funding_monthly_stipend, funding_coverage_count |
| Boolean (Float) | 7 features | requires_financial_need, requires_return_home_country, 5 funding coverage flags |
| List Fields (Binary) | 32 features | eligible_nationalities, eligible_fields, eligible_high_school_tracks encoded as binary vectors |

### Dataset Statistics

| Entity | Count |
|--------|-------|
| Students | 20,000 |
| Scholarships | 800 |
| Training Pairs | ~525,000 (70% of 750K) |
| Validation Pairs | ~112,500 (15%) |
| Test Pairs | ~112,500 (15%) |

### Class Distribution (Balanced)

| Class | Relevance Range | Approximate Count |
|-------|-----------------|--------------------|
| Match | ≥ 0.7 | ~250,000 |
| In-Between | 0.3 – 0.7 | ~250,000 |
| Not Match | < 0.3 | ~250,000 |