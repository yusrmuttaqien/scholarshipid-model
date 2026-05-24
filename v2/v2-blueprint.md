# Scholarship Recommendation System — v2 Blueprint

## Architecture Overview

Two-tower neural network trained **directly from user interaction signals**. No synthetic pairs, no hard filters.

```
┌──────────────-------─┐     ┌──────────────┐     ┌──────────────┐     ┌─────────────---------------┐
│  interactions.csv    │     │  training    │     │  best_model  │     │ Inference                  │
│  (+ students.csv)    │────▶│  pipeline    │────▶│  .keras      │────▶│ student.json               │
│  (+ scholarships.csv)│     │              │     │              │     │ → scholarships score [0-1] │
└──────────────---─----┘     └──────────────┘     └──────────────┘     └─────────────---------------┘
```

## Data Model

### Core Tables (CSV files)

**`students.csv`** — Student profiles (unchanged from v1)
- Fields: student_id, nationality, age, school_track, report_card_average, etc.
- Generated synthetically by `generator.py`, or uploaded from real data

**`scholarships.csv`** — Scholarship catalog (unchanged from v1)
- Fields: scholarship_id, host_country, eligible_nationalities, min_age, max_age, language_requirements, etc.
- Static reference data that gets updated manually

**`interactions.csv`** — User interaction events (replaces feedback.csv + pairs.csv)
```csv
student_id      ,scholarship_id ,event_type ,weight ,timestamp
STU_001         ,SCH_456        ,view       ,0.3    ,2025-01-15T10:30:00Z
STU_001         ,SCH_456        ,click      ,1.0    ,2025-01-15T10:31:00Z
STU_001         ,SCH_456        ,apply      ,2.0    ,2025-01-16T14:00:00Z
STU_001         ,SCH_789        ,view       ,0.3    ,2025-01-17T09:00:00Z
```

### Event Types & Weights

| Event | Weight | Meaning |
|-------|--------|---------|
| view | 0.3 | Student viewed the scholarship page (weak signal) |
| click | 1.0 | Student clicked on the scholarship (moderate signal) |
| apply | 2.0 | Student applied to the scholarship (strong signal) |
| reject | -1.0 | Student rejected the recommendation (negative signal) |

### Derived Training Data

Training pairs are **computed at training time** from `interactions.csv`:

```python
# Aggregate interactions per (student, scholarship):
#   positive = sum of view/click/apply weights
#   negative = abs(sum of reject weights)
#   score = sigmoid(positive - 0.5 * negative)  # maps to [0, 1]
#
# Negative samples: random (student, scholarship) pairs with NO interactions → label = 0.0

pairs_for_training = []
for (stu_id, sch_id), agg in interactions.groupby(["student_id", "scholarship_id"]):
    positive = agg[agg.event_type != "reject"]["weight"].sum()
    negative = abs(agg[agg.event_type == "reject"]["weight"].sum())
    score = sigmoid(positive - 0.5 * negative)
    pairs_for_training.append((stu_id, sch_id, score))

# Add random negatives (no interactions → no interest)
negative_pairs = sample_random_pairs(n_students * n_scholarships, k=100000)
for stu_id, sch_id in negative_pairs:
    pairs_for_training.append((stu_id, sch_id, 0.0))
```

## Component Breakdown

### `generator.py` — Synthetic Data Generation

- Generates initial `students.csv` and `scholarships.csv` for bootstrapping
- Provides `compute_relevance_score(student, scholarship)` function for seeding new pairs `EXPLAIN MORE`
- **No longer generates pairs.csv** — that's replaced by real interaction data

### `pair_manager.py` — Pair Operations (New)

Handles pair creation and lookup during production: `EXPLAIN MORE`

```python
def ensure_pair_exists(student_id, scholarship_id):
    """Check if a pair exists in interactions; seed with compute_relevance_score if not."""
    existing = interactions[(interactions.student_id == student_id) & 
                            (interactions.scholarship_id == scholarship_id)]
    
    if existing.empty:
        # Seed with synthetic score from the scoring function
        student = load_student(student_id)
        scholarship = load_scholarship(scholarship_id)
        seed_score = compute_relevance_score(student, scholarship)
        
        # Insert as a "view" event with low weight (seeded, not user-driven)
        interactions.append({
            "student_id": student_id,
            "scholarship_id": scholarship_id,
            "event_type": "seeded_view",  # special type, low weight
            "weight": seed_score * 0.3,   # scaled to event weights
            "timestamp": now(),
        })

def record_interaction(student_id, scholarship_id, event_type):
    """Append user interaction."""
    interactions.append({
        "student_id": student_id,
        "scholarship_id": scholarship_id,
        "event_type": event_type,  # view/click/apply/reject
        "weight": EVENT_WEIGHTS[event_type],
        "timestamp": now(),
    })

def aggregate_for_training(interactions_df):
    """Compute training pairs from interactions."""
    # See aggregation logic above
    ...
```

### `train.py` — Training Pipeline

1. Load `students.csv`, `scholarships.csv`, `interactions.csv`
2. Aggregate interactions into pair scores (see "Derived Training Data" above)
3. Sample negative examples (random pairs with no interactions) `EXPLAIN MORE`
4. Preprocess features using schema.json `EXPLAIN MORE`
5. Train two-tower model on `(student_features, scholarship_features) → score`

### `inference.py` — Inference Engine (Simplified)

**No hard filters.** The model itself learns eligibility patterns from training data.

```python
engine = InferenceEngine(model_dir="models", schema_file="models/schema.json")

# Single student → top-k scholarships
result = engine.recommend(student_dict, scholarships_list, top_k=5)

# Returns: list of (scholarship_id, score) sorted by score descending
```

### `inference_loop.py` — Production Interaction Handler (New)

Called by the web service after user actions:

```python
# After user clicks a scholarship:
from v2.inference_loop import record_interaction
record_interaction(student_id="STU_001", scholarship_id="SCH_456", event_type="click")

# Also ensures pair exists (seeds with compute_relevance_score if cold start)
ensure_pair_exists("STU_001", "SCH_456")
```

### `retrain.py` — Scheduled Retraining (New)

Cron job that runs periodically:
1. Read latest interactions
2. Aggregate into training pairs
3. Run `train.py` with new data
4. Save updated model to `models/`

```bash
python retrain.py --schedule daily
```

## Production Flow

### 1. Inference (Web Service)

```
Client Request → Web Service → InferenceEngine.recommend()
                                                    ↓
                                         Returns ranked list:
                                         - Known scholarships with model scores
                                         - New scholarships (unranked, no score yet)
```

**New scholarship handling:**
- Model doesn't have embeddings for SCH_999 → can't score it
- Show as unranked list alongside scored results
- No hard filter needed — just "unknown" items appear without a score

### 2. User Interaction

```
User Action (click/apply/reject)
    ↓
inference_loop.record_interaction()
    ↓
Append to interactions.csv
```

### 3. Cold Start for New Scholarship

When user interacts with SCH_999 for the first time:
```
record_interaction("STU_001", "SCH_999", "click")
    ↓
ensure_pair_exists() checks if (STU_001, SCH_999) exists
    ↓
If not: seed with compute_relevance_score(STU_001, SCH_999) * 0.3
    ↓
Now pair has a baseline score + user interaction signal
```

### 4. Retraining (Cron)

```
retrain.py runs every N hours/days
    ↓
Read interactions.csv → aggregate into pair scores
    ↓
Sample negatives (random unobserved pairs)
    ↓
train.py trains on new data → saves best_model.keras
    ↓
New model now has embeddings for SCH_999!
```

## Key Design Decisions

### 1. No Hard Filters
The model learns eligibility patterns from training data. If nationality matching matters, the model will learn it because pairs with mismatched nationalities have lower interaction scores. This makes inference.py fully modular — no external schema knowledge needed at runtime.

**Trade-off:** Some truly ineligible pairs get scored (model may give a non-zero score to wrong-nationality pair). But this is rare in practice and the model learns to push them down over time.

### 2. Synthetic Seeding via `compute_relevance_score()`
For cold-start pairs (new scholarship × existing student, or vice versa), use the deterministic scoring function as a **weak seed signal**. This gives the model something to learn from immediately rather than waiting for real interactions.

The seed weight is scaled down (`seed_score * 0.3`) so real user signals dominate once they arrive.

### 3. Random Negative Sampling
Instead of relying on synthetic "not match" pairs, sample random (student, scholarship) pairs that have NO interactions in `interactions.csv`. These are assumed to be uninteresting → label = 0.0.

This is more honest than synthetic negatives because it reflects actual user behavior: if no one interacted with a pair, it's likely low relevance.

### 4. Event Types > Binary Labels
Using weighted events (view/click/apply/reject) instead of binary labels captures richer signals:
- A student who viewed but never clicked → weak interest
- A student who applied and then rejected → strong negative signal
- Multiple views over time → growing interest

### 5. Schema Still Used for Feature Engineering
The schema.json is still needed at **training time** to understand feature dimensions, column names, and list vector structure. But at **inference time**, the model only needs the packed feature vectors — no schema knowledge required.

## Files Reference (v2)

| File | Purpose |
|------|---------|
| `generator.py` | Generate initial students.csv + scholarships.csv; provides `compute_relevance_score()` |
| `pair_manager.py` | Pair creation, lookup, seeding (new) |
| `inference_loop.py` | Production interaction handler (new) |
| `train.py` | Training pipeline — reads interactions, aggregates pairs, trains model |
| `retrain.py` | Scheduled retraining via cron (new) |
| `inference.py` | Inference engine — no hard filters, fully modular (simplified from v1) |

## CLI Flag Reference

All scripts follow the strict **no-default-path** policy: every file path must be provided explicitly via flags. Only parameters (epochs, batch-size, alpha, etc.) may have defaults. The only exception is `generator.py`, which auto-loads data from `src/` in the project root.

### `generator.py`
```bash
python generator.py \
  [--n-students N] [--n-scholarships M] [--output-dir DIR]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--n-students` | 1000 | Number of synthetic students to generate |
| `--n-scholarships` | 800 | Number of synthetic scholarships to generate |
| `--output-dir` | (auto: `src/datasets/`) | Where to write CSVs. Auto-resolves relative to project root if omitted |

**Note:** Generator is the only script with auto-loading defaults. It reads seed data from `src/` at project root and writes outputs there. All other scripts require explicit paths.

### `train.py`
```bash
python train.py \
  --students-file /path/to/students.csv \
  --scholarships-file /path/to/scholarships.csv \
  --interactions-file /path/to/interactions.csv \
  --schema-file /path/to/schema.json \
  --output-dir /path/to/output/ \
  [--epochs N] [--batch-size N] [--negative-ratio R] [--seed S]
```

| Flag | Required | Default | Description |
|------|----------|---------|-------------|
| `--students-file` | **yes** | — | Path to students CSV |
| `--scholarships-file` | **yes** | — | Path to scholarships CSV |
| `--interactions-file` | **yes** | — | Path to interactions CSV (replaces pairs.csv) |
| `--schema-file` | **yes** | — | Path to schema JSON |
| `--output-dir` | **yes** | — | Directory for model, schema, mappings |
| `--epochs` | no | 30 | Training epochs |
| `--batch-size` | no | 256 | Batch size |
| `--negative-ratio` | no | 0.15 | Ratio of random negatives to positive pairs (e.g., 0.15 = 15% negatives) |
| `--seed` | no | 42 | Random seed for reproducibility |

### `inference.py`
```bash
python inference.py \
  --input /path/to/student.json \
  --models-dir /path/to/model/ \
  --scholarships-file /path/to/scholarships.csv \
  [--top-k N] [--no-hard-filter]
```

| Flag | Required | Default | Description |
|------|----------|---------|-------------|
| `--input` | **yes** | — | JSON file with student profile |
| `--models-dir` | **yes** | — | Directory containing `best_model.keras` and `schema.json` |
| `--scholarships-file` | **yes** | — | Path to scholarships CSV (to filter unknown IDs) |
| `--top-k` | no | 5 | Number of recommendations to return |

**Note:** Hard filters removed. No `--students-file`, `--schema-file` flags needed at inference time. Unknown scholarship IDs appear as unranked entries in results.

### `inference_loop.py`
```bash
python inference_loop.py \
  --interactions-file /path/to/interactions.csv \
  --students-file /path/to/students.csv \
  --scholarships-file /path/to/scholarships.csv \
  --student-id ID \
  --scholarship-id ID \
  --event-type TYPE
```

| Flag | Required | Default | Description |
|------|----------|---------|-------------|
| `--interactions-file` | **yes** | — | Path to interactions CSV (written atomically) |
| `--students-file` | **yes** | — | Path to students CSV (for seeding) |
| `--scholarships-file` | **yes** | — | Path to scholarships CSV (for seeding) |
| `--student-id` | **yes** | — | Student ID |
| `--scholarship-id` | **yes** | — | Scholarship ID |
| `--event-type` | **yes** | — | Event type: view, click, apply, reject, seeded_view |

### `retrain.py`
```bash
python retrain.py \
  --students-file /path/to/students.csv \
  --scholarships-file /path/to/scholarships.csv \
  --interactions-file /path/to/interactions.csv \
  --schema-file /path/to/schema.json \
  --output-dir /path/to/output/ \
  [--epochs N] [--batch-size N] [--negative-ratio R] [--seed S]
```

Same flags as `train.py` (retrain.py is a thin wrapper that calls train.py internally).

### Example: Full Retraining Pipeline

```bash
# Step 1: Generate synthetic data (one-time bootstrap)
python generator.py --n-students 5000 --n-scholarships 2000

# Step 2: Collect real interactions in production via inference_loop.py

# Step 3: Retrain with real interaction data
python retrain.py \
  --students-file src/datasets/students.csv \
  --scholarships-file src/datasets/scholarships.csv \
  --interactions-file /path/to/current/interactions.csv \
  --schema-file src/models/schema.json \
  --output-dir models/v3 \
  --epochs 50 \
  --negative-ratio 0.1

# Step 4: Run inference with new model
python inference.py \
  --input student.json \
  --models-dir models/v3 \
  --scholarships-file src/datasets/scholarships.csv \
  --top-k 5
```
