# Scholarship Recommendation System — v1

A two-tower recommendation system for matching students with scholarships. Trained on historical student-scholarship pairs, the model learns which features drive relevance and predicts a score in [0, 1] for any student-scholarship pair.

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌──────────────┐
│   Training   │     │   Model      │     │ Inference    │
│              │     │ (best_model  │     │              │
│ CSV datasets │────▶│ .keras)      │────▶│ student.json │
│ + schemas    │     │              │     │ → recs       │
└─────────────┘     └──────────────┘     └──────────────┘
```

The system uses a **two-tower neural network**: one tower encodes the student, another encodes the scholarship. Their embeddings are compared via cosine similarity + sigmoid to produce a relevance score. A two-stage pipeline applies hard filters first, then ranks remaining candidates with the model.

## Generator (`generator.py`)

### What it does

Generates synthetic training data for the two-tower recommendation system: 20K students, 800 scholarships, balanced relevance-scored pairs, and implicit feedback.

The generator combines:
- Student/scholarship generation logic (country-specific schools, realistic distributions)
- A 5-stage relevance scorer (hard eligibility + component scores with weights)

### Output Structure

```
v1/datasets/
├── students.csv          # 20,000 synthetic student profiles
├── scholarships.csv      # 800 synthetic scholarship programs
├── pairs.csv             # Balanced pairs: relevance_score (0.0-1.0) + timestamp
└── feedback.csv          # Implicit user feedback (apply/click/view/reject)
```

### Relevance Scoring Algorithm

`compute_relevance_score(student, scholarship)` returns a continuous score in [0, 1] using weighted components:

| Component | Weight | Logic |
|-----------|--------|-------|
| Nationality match | 20% | Exact match (1.0) or no match (0.0) |
| Age compatibility | 10% | Within `[min_age, max_age]` range |
| High school track | 15% | Match (1.0) or partial (0.1) |
| Report card margin | 10% | Margin above threshold scaled to [0, 1] |
| Major subject margin | 10% | Same as report card |
| Language requirement | 10% | Ratio of achieved score vs required |
| Return home | 5% | Matches if student willing AND scholarship requires |
| Financial need | 5% | Income-based scoring (very_low/low = strong, middle = partial) |

### Pair Generation Strategy

`generate_balanced_pairs()` creates three categories:

- **Match** (relevance ≥ 0.7): Strong attribute alignment — used as positive examples
- **In-Between** (0.3 ≤ relevance < 0.7): Partial alignment — provides nuanced learning signal
- **Not Match** (relevance < 0.3): No alignment — negative examples

Default targets: 250K match, 250K in-between, 250K not-match (configurable via `ratio_inbetween`, `ratio_notmatch`).

### Usage

```bash
# Generate datasets from scratch
python v1/generator.py

# Regenerate with different parameters (edit NUM_STUDENTS, etc. at top of main())
python v1/generator.py
```

### How It Works

```
TwoTowerDatasetGenerator
├── generate_student()    → Student object with realistic attributes
│   ├── nationality from 28 countries across 6 regions
│   ├── age: 16-18, scores: 50-100 scale
│   ├── language tests (TOEFL/IELTS/etc.) probabilistically assigned
│   └── olympiads, leadership, volunteer counts from Poisson distributions
│
├── generate_scholarship() → Scholarship object with realistic constraints
│   ├── host country + region from predefined pools
│   ├── eligible_nationalities: 2-8 countries
│   ├── age range, degree levels, tracks, fields randomly sampled
│   └── language requirements, financial need, funding coverage randomized
│
├── compute_relevance_score() → float in [0, 1]
│   └── weighted sum of 8 alignment components (see table above)
│
├── generate_balanced_pairs() → List[Pair]
│   ├── computes relevance for all student-scholarship combinations
│   ├── bins into match/in-between/not-match
│   └── samples to target counts
│
└── generate_feedback() → List[Feedback]
    └── implicit signals: apply, click, view, reject (weighted probabilities)
```

### Key Design Decisions

- **Seed-based reproducibility**: `seed=42` in constructor ensures identical outputs every run
- **Time-based splitting**: Pairs are assigned timestamps; training splits by time (70/15/15) rather than random — this prevents data leakage between train/test sets
- **Balanced pairs**: The 3-category split ensures the model sees all three types of examples in roughly equal proportions
- **Realistic distributions**: Student attributes follow realistic distributions (e.g., most students don't have language test scores yet, income distribution skews middle)

## Directory Structure

```
v1/
├── train.py                    # Training script
│
├── inference.py                # Inference engine (library + CLI)
│                               # - InferenceEngine class
│                               # - apply_hard_filter()
│                               # - Feature encoding functions
│                               # - CLI with --input (all paths explicit)
│
├── generator.py                # Synthetic data generation
│                               # - TwoTowerDatasetGenerator class
│                               # - Relevance scoring algorithm
│                               # - Balanced pair generation
│
├── models/                     # Saved artifacts from training
│   ├── best_model.keras        # Trained model weights
│   ├── schema.json             # Feature schema (column names, types, dims)
│   └── mappings.pkl            # Label encodings for categorical fields
│
├── datasets/                   # Raw data (generated by generator.py)
│   ├── students.csv            # ~20K student profiles
│   ├── scholarships.csv        # 800 scholarship programs
│   ├── pairs.csv               # Labeled student-scholarship relevance scores
│   └── feedback.csv            # Implicit user feedback
│
└── example_student.json        # Sample student profile for inference
```

## Training (`train.py`)

### What it does

1. **Loads schema** from `schema.json` (column names, types, list vector structure)
2. **Loads** `students.csv`, `scholarships.csv`, and `pairs.csv` from `datasets/` (or custom paths via CLI flags)
3. **Preprocesses features** using pandas: label-encodes categoricals, normalizes numericals, builds list vectors (countries, tracks, fields) — all driven by the schema
4. **Splits data** by timestamp (70% train / 15% val / 15% test)
5. **Builds two-tower model**: student tower (64-dim embedding) + scholarship tower (64-dim embedding) → CosineSimilarity layer → sigmoid output
6. **Trains** with MSE loss, early stopping on validation MAE, ReduceLROnPlateau
7. **Saves artifacts**: `best_model.keras`, `schema.json` (updated with actual input dims), `mappings.pkl`

### Usage

```bash
# Use defaults (30 epochs, batch size 256, reads from v1/datasets/ and outputs to v1/models/)
python train.py

# Custom hyperparameters
python train.py --epochs 50 --batch-size 128

# Full external file control (e.g., feedback-loop retraining)
python train.py \
  --students-file v1/datasets/students.csv \
  --scholarships-file v1/datasets/scholarships.csv \
  --pairs-file v1/datasets/pairs_feedback.csv \
  --schema-file v1/models/schema.json \
  --output-dir v1/models/
```

> **External file paths**: All dataset files and the schema can be specified via CLI flags. When omitted, they default to `v1/datasets/` for inputs and `v1/models/` for outputs. This makes it easy to retrain with adjusted datasets (e.g., from feedback) without modifying code.

### Model Architecture

```
Student Tower:
  Input (30 dims) → Dense(128, relu) → BN → Dense(64, relu) → BN → Output(64)

Scholarship Tower:
  Input (63 dims) → Dense(128, relu) → BN → Dense(64, relu) → BN → Output(64)

CosineSimilarity Layer:
  cosine_similarity = L2_norm(student_emb) · L2_norm(scholarship_emb)
  score = sigmoid(cosine_similarity)   # maps to [0, 1]
```

### Feature Engineering

**Student features (30 dims):**
| Type | Columns | Count |
|------|---------|-------|
| Categorical | nationality, high_school_track, school_tier, family_income_category, intended_career_track, olympiad_level | 6 |
| Numerical | age, overall_report_card_average, math_score, english_score, major_subject_average, leadership_experience_count, volunteer_experience_count, competition_wins_count | 8 |
| Boolean | willing_to_return_home, from_underrepresented_region, needs_full_funding, can_self_fund_living | 4 |
| List (language) | toefl/max, toefl/presence, ielts/max, ..., hsk/presence | 12 |

**Scholarship features (63 dims):**
| Type | Columns | Count |
|------|---------|-------|
| Categorical | host_region, preferred_school_tier, career_track_preference, max_family_income_category | 4 |
| Numerical | min_age, max_age, min_report_card_average, min_major_subject_average, funding_monthly_stipend, funding_coverage_count | 6 |
| Boolean | requires_financial_need, requires_return_home_country, funding_covers_tuition, ..., funding_is_full_funding | 7 |
| List vector | eligible_nationalities (27) + eligible_high_school_tracks (5) + eligible_fields (14) | 46 |

### Custom Layer: CosineSimilarity

The `CosineSimilarity` layer is decorated with `@register_keras_serializable(package="ScholarshipID")` so it gets serialized into the `.keras` file. When loading in inference, you must pass it via `custom_objects=...`.

## Inference (`inference.py`)

### Two-Stage Pipeline

| Stage | Method | Purpose |
|-------|--------|---------|
| **Hard Filter** | Deterministic rules | Remove clearly ineligible scholarships (wrong nationality, age out of range, etc.) |
| **Model Scoring** | Two-tower neural network | Rank remaining candidates by learned relevance score |

### Hard Filters (Stage 1)

Six checks run in order. A scholarship is kept only if ALL pass:

1. **Nationality**: Student's nationality must be in `eligible_nationalities`
2. **Age**: Student age must be within `[min_age, max_age]`
3. **Degree Level**: Student's target degree level must be in `eligible_degree_levels`
4. **High School Track**: Student's track (science, social_studies, etc.) must be eligible
5. **Language Requirements**: Student must meet mandatory language scores (TOEFL/IELTS/etc.)
6. **Academic Thresholds**: Report card and major subject averages must exceed minimums

> Note: `willing_to_return_home` and strict financial need are intentionally NOT hard filters. These are captured by the model's learned embeddings, so filtering them out would leave too few candidates.

### Feature Encoding (Schema-Based)

The inference engine loads `schema.json` which defines all column names, types, and list vector dimensions. No hardcoded constants needed — inference adapts to whatever schema the model was trained with.

**Student feature construction:**
```python
student_features = [
    categorical(nationality),  # one-hot encoded via training mappings
    categorical(high_school_track),
    ...                          # all student columns
    language_vector(12)         # parsed from JSON proficiency data
]
# Total: 30 dimensions
```

**Scholarship feature construction:**
```python
scholarship_features = [
    categorical(host_region),
    numerical(min_age, max_age, ...),
    boolean(requires_financial_need, ...),
    list_vector(46)             # countries (27) + tracks (5) + fields (14)
]
# Total: 63 dimensions
```

### Library API

```python
from v1.inference import InferenceEngine, Recommendation, InferenceResult
from pathlib import Path

# Initialize engine — all paths explicit
engine = InferenceEngine(
    model_dir=Path("v1/models"),
    schema_file="v1/models/schema.json",
)

# Run recommendation — student is a dict from your JSON input
result = engine.recommend(
    student=student_dict,           # Dict with student fields (from JSON input)
    scholarships=[sch1, sch2],     # List of Scholarship dicts
    top_k=5,                       # Number of recommendations
    use_hard_filter=True,          # Toggle Stage 1 hard filter
)

# Access results
for rec in result.recommendations:
    print(f"#{rec.rank} {rec.scholarship_name}: {rec.relevance_score:.4f}")
```

### CLI Usage

All paths must be provided explicitly — no auto-loading:

```bash
# Single student profile (JSON) + external datasets
python v1/inference.py \
  --input example_student.json \
  --models-dir v1/models \
  --schema-file v1/models/schema.json \
  --students-file v1/datasets/students.csv \
  --scholarships-file v1/datasets/scholarships.csv \
  --top-k 5

# Disable hard filter (let model handle all scoring)
python v1/inference.py \
  --input v1/example_student.json \
  --models-dir v1/models \
  --schema-file v1/models/schema.json \
  --students-file v1/datasets/students.csv \
  --scholarships-file v1/datasets/scholarships.csv \
  --no-hard-filter

# With custom model directory (e.g., feedback-loop retraining)
python v1/inference.py \
  --input student.json \
  --models-dir /path/to/experiment_A/ \
  --schema-file /path/to/experiment_A/schema.json \
  --students-file /path/to/students.csv \
  --scholarships-file /path/to/scholarships.csv \
  --top-k 3
```

### Output Format

```
Recommendations for STU_000050
  School: Gymnasium Berlin
----------------------------------------------------------------------
  #1  Australia Education Scholarship      Score: 0.5817
  #2  China Social Sciences Scholarship    Score: 0.5510
  #3  Brazil Mathematics Scholarship       Score: 0.5491
```

### JSON Input Format

See `example_student.json` for a complete example. Required fields are marked with `<required>` in the Student schema; all others have defaults.

**Language proficiency format:**
```json
"language_proficiency": [
    {"test_type": "ielts", "score": 7.5},
    {"test_type": "toefl", "score": 102}
]
```

Supported test types: `toefl`, `ielts`, `topik`, `jlpt`, `delf`, `hsk`. These are defined in `schema.json` and can be extended without changing code.

## Portability

The inference pipeline is designed to work with just three files copied anywhere:
- `best_model.keras` — trained weights
- `schema.json` — feature structure definition
- `mappings.pkl` — categorical label encodings

No training code, no CSV datasets, and no hardcoded constants needed. The schema makes the system self-describing.

## Schema (`schema.json`)

The schema file is the single source of truth for feature structure:

```json
{
  "student": {
    "categorical": ["nationality", "high_school_track", ...],
    "numerical": ["age", "overall_report_card_average", ...],
    "boolean": ["willing_to_return_home", ...],
    "language_dim": 12,
    "language_tests": ["toefl", "ielts", ...],
    "input_dim": 30
  },
  "scholarship": {
    "categorical": ["host_region", ...],
    "numerical": ["min_age", ...],
    "boolean": ["requires_financial_need", ...],
    "list_vector_dim": 46,
    "list_country_dim": 27,
    "list_track_dim": 5,
    "list_field_dim": 14,
    "all_countries": [...],
    "all_tracks": [...],
    "all_fields": [...],
    "input_dim": 63
  }
}
```

## Files Reference

| File | Purpose | Key Exports |
|------|---------|-------------|
| `generator.py` | Synthetic data generation | `TwoTowerDatasetGenerator`, `main()` |
| `train.py` | Training pipeline (external file paths via CLI flags) | `train()`, `prepare_features()`, `build_two_tower_model()` |
| `inference.py` | Inference engine | `InferenceEngine`, `apply_hard_filter()`, `_parse_language_proficiency()` |
| `mappings.pkl` | Label encodings from training | `{student: {...}, scholarship: {...}}` |
| `schema.json` | Feature structure definition | See schema section above |
| `best_model.keras` | Trained model weights | Keras Functional API model |
| `example_student.json` | Sample student profile for inference | — |

## Usage Examples

### Generate Synthetic Data

```bash
cd v1
python generator.py
# → generates students.csv, scholarships.csv, pairs.csv, feedback.csv in datasets/
```

### Programmatic Usage

```python
from v1.inference import InferenceEngine
import pandas as pd
from pathlib import Path

engine = InferenceEngine(
    model_dir=Path("v1/models"),
    schema_file="v1/models/schema.json",
)

# Load data from external files
students_df = pd.read_csv("v1/datasets/students.csv")
scholarships_df = pd.read_csv("v1/datasets/scholarships.csv")

# Pick a student and all scholarships
student = students_df.iloc[0]
result = engine.recommend(
    student=student,                    # Student row as dict
    scholarships=scholarships_df.to_dict(orient="records"),
    top_k=5,
)
```

### Reproducing Training

```bash
cd v1
python train.py --epochs 50 --batch-size 256
# → saves best_model.keras, schema.json, mappings.pkl to models/
```

### Monitoring with TensorBoard

TensorBoard logs are saved to `v1/models/logs/` during training.

```bash
tensorboard --logdir v1/models/logs/ --port 6006
```

Then open http://localhost:6006 in your browser. Key tabs to monitor:

- **Scalars**: loss, mae (train/val), learning rate schedule
- **Graph**: model architecture visualization
- **Distributions/Histograms**: weight and activation distributions over time
