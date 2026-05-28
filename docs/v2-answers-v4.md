# Scholarship Recommendation System — Final Spec
> For high school students seeking bachelor's scholarships abroad. <br>
> This spec is written to be given directly to a coding model. Read it top-to-bottom before writing any code — every section builds on the previous one. <br>
> Any python code run must be ran inside a env using `.venv/bin/activate`.

---

## 1. Architecture Decision

**Model type: Two Tower (Dual Encoder)**

### Why Two Tower
| Concern | Two Tower behavior |
|---|---|
| New scholarship cold-start | Included in next retrain payload → pseudo-label pairs generated from eligibility → gets an embedding and is surfaced immediately. Sharpens as interactions accumulate. |
| New student cold-start | Student tower produces embedding from features alone. Works immediately. |
| Inference speed | Pre-compute all scholarship embeddings once. Query = 1 student forward pass + dot product. |
| Interaction data fit | (student, scholarship, weight) pairs map directly to contrastive training. |
| Compatibility breakdown | Sub-scores can be extracted per feature group inside each tower. |

### High-level flow

```
[Student Profile]
       │
  ┌────▼────────┐
  │ Hard Filter │  ← Eligibility gate (nationality, age, degree, etc.)
  └────┬────────┘
       │ Eligible scholarships only
  ┌────▼────────┐       ┌──────────────────────┐
  │Student Tower│       │ Scholarship Embeddings│
  │  (Keras)   │       │ Index (.npy + id map) │
  └────┬────────┘       └──────────┬───────────┘
       │ student_embedding          │ pre-computed, sliced by eligible indices
       └──────────┬─────────────────┘
                  │
         cosine_similarity(student_emb, eligible_scholarship_embs)
                  │
         ┌────────▼─────────┐
         │  Ranked Results  │ relevance scores [0, 1]
         │  + Breakdown     │ academic, leadership, language
         └──────────────────┘
```

---

## 2. Global Rules

### Reproducibility — Single Seed
All randomness across the entire pipeline uses one seed defined in `training/config.py`:

```python
# training/config.py
SEED = 42
```

**Every component that has randomness imports and uses this value. The number 42 must never appear anywhere else in the codebase.**

| Component | How seed is applied |
|---|---|
| `training/train.py` | `random.seed(SEED)`, `np.random.seed(SEED)`, `tf.random.set_seed(SEED)` |
| `data/generator/students.py` | `rng = np.random.default_rng(SEED)` — use this rng for all synthetic generation |
| `data/pair_builder.py` | `rng = np.random.default_rng(SEED)` — used when sampling negatives |
| `training/train.py` (dataset) | `dataset.shuffle(buffer_size=10_000, seed=SEED)` |
| `training/evaluate.py` | `rng = np.random.default_rng(SEED)` — if sampling during eval |

---

## 3. Repository Structure

```
scholarship-recommender/
│
├── schemas/                    # Domain models (reuse from existing generator)
│   ├── enums.py                # All enums: Country, DegreeLevel, etc.
│   ├── student.py              # Student dataclass
│   └── scholarship.py          # Scholarship, FundingCoverage, SelectionCriteria, etc.
│
├── data/                       # Data generation and loading
│   ├── generator/
│   │   ├── scholarships.py     # Hardcoded scholarship profiles (~18)
│   │   └── students.py         # Seed data and synthetic generation (uses SEED)
│   ├── preprocessor.py         # Feature extraction: dataclass → numeric/text tensors
│   └── pair_builder.py         # Reads exports CSVs → builds training pairs
│
├── model/
│   ├── towers.py               # StudentTower and ScholarshipTower (Keras subclassing)
│   ├── recommender.py          # Full TwoTowerRecommender model (ties towers together)
│   ├── components/
│   │   ├── loss.py             # Custom: WeightedContrastiveLoss
│   │   ├── layers.py           # Custom: CompatibilityScorer layer
│   │   └── callbacks.py        # Custom: NDCGCallback
│   └── text_encoder.py         # TF-Hub Universal Sentence Encoder wrapper
│
├── training/
│   ├── train.py                # Main training script
│   ├── config.py               # Single source of truth: SEED, hyperparameters, all paths
│   └── evaluate.py             # NDCG@K, Hit Rate@K, MAE evaluation
│
├── inference/
│   ├── embeddings.py           # Build and save scholarship embedding index
│   ├── hard_filter.py          # Pre-model eligibility gate (pure Python, no tensors)
│   └── predictor.py            # Load model + index + scholarships → produce ranked results
│
├── service/
│   ├── main.py                 # FastAPI app, router registration, Predictor init on startup
│   ├── routes/
│   │   ├── recommend.py        # POST /v1/recommend, POST /v1/recommend/explain
│   │   ├── interactions.py     # POST /v1/interactions
│   │   ├── scholarships.py     # GET /v1/scholarships
│   │   └── retrain.py          # POST /v1/retrain — receives CSVs, triggers retraining
│   └── schemas_api.py          # Pydantic request/response models (separate from domain schemas)
│
├── feedback/
│   └── retrain_trigger.py      # Orchestrates: pair_builder → train → embeddings → promote
│
└── artifacts/                  # Generated at runtime, never committed to git
    ├── active.json             # Source of truth for which model/embeddings version is live
    ├── exports/                # Staging area for CSVs received from web backend
    │   ├── students.csv
    │   ├── scholarships.csv
    │   └── interactions.csv
    ├── model/
    │   ├── model_v1/
    │   └── model_v2/           # Keras SavedModel format
    ├── embeddings/
    │   ├── v1/
    │   │   ├── scholarship_embeddings.npy
    │   │   └── scholarship_id_map.json
    │   └── v2/
    │       ├── scholarship_embeddings.npy
    │       └── scholarship_id_map.json
    └── logs/                   # TensorBoard logs
```

---

## 4. Configuration (`training/config.py`)

All constants live here. Nothing is hardcoded elsewhere.

```python
import os
from pathlib import Path

# ─── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42

# ─── Model ────────────────────────────────────────────────────────────────────
EMBEDDING_DIM       = 128
BATCH_SIZE          = 256
LEARNING_RATE       = 1e-3
MAX_EPOCHS          = 50
EARLY_STOP_PATIENCE = 5
TOP_K               = 10        # for NDCG@K evaluation
NDCG_THRESHOLD      = 0.80      # minimum NDCG@10 required to promote a new model version
MAX_SCHOLARSHIP_AGE = 30        # normalization denominator for scholarship min/max age

# ─── Text encoder ─────────────────────────────────────────────────────────────
USE_MODEL_URL = "https://tfhub.dev/google/universal-sentence-encoder/4"

# ─── Interaction event weights ─────────────────────────────────────────────────
EVENT_WEIGHTS = {
    "view":   0.1,
    "click":  0.3,
    "apply":  1.0,
    "reject": -0.5,
}

# ─── Paths ────────────────────────────────────────────────────────────────────
ARTIFACTS_DIR  = Path("artifacts/")
ACTIVE_FILE    = ARTIFACTS_DIR / "active.json"
EXPORTS_DIR    = ARTIFACTS_DIR / "exports"
MODEL_DIR      = ARTIFACTS_DIR / "model"
EMBEDDINGS_DIR = ARTIFACTS_DIR / "embeddings"
LOGS_DIR       = ARTIFACTS_DIR / "logs"
```

---

## 5. Data Architecture

### 5.1 CSV Schemas (from web backend)

The web backend sends three CSVs to `POST /v1/retrain`. Their columns must match the domain schemas exactly.

**students.csv** — one row per student, columns match `Student` dataclass fields.
Multi-value fields (lists) are stored as pipe-separated strings:
```
student_id, nationality, age, high_school_track, overall_report_card_average, ...
language_proficiency (pipe-separated: "ielts:7.5|toefl:100"),
olympiad_subjects    (pipe-separated: "mathematics|physics"),
target_countries     (pipe-separated: "japan|south_korea")
```

**scholarships.csv** — one row per scholarship, columns match `Scholarship` dataclass fields.
Same pipe-separated convention for list fields.

**interactions.csv** — one row per event:
```
interaction_id, student_id, scholarship_id, event_type, timestamp, session_id
```

### 5.2 Feature Extraction (`data/preprocessor.py`)

Converts dataclass instances into tensor dicts. This is the bridge between domain schemas and the model. Called during training (from pair_builder) and during inference (for the student only).

**Student features → four groups:**

```python
# Group A: Numeric — normalized to [0, 1]
student_numeric = [
    age / 18.0,
    overall_report_card_average / 100.0,
    math_score / 100.0,
    english_score / 100.0,
    major_subject_average / 100.0,
    min(leadership_experience_count, 10) / 10.0,
    min(volunteer_experience_count, 10) / 10.0,
    min(competition_wins_count, 10) / 10.0,
    float(needs_full_funding),
    float(can_self_fund_living),
    float(from_underrepresented_region),
    float(willing_to_return_home),
]
# shape: (12,)

# Group B: Categorical — integer-encoded or multi-hot
student_categorical = {
    "nationality":      Country → integer index,          # shape: (1,)
    "hs_track":         HighSchoolTrack → integer index,  # shape: (1,)
    "olympiad_level":   OlympiadLevel → ordinal 0–5,      # shape: (1,)
    "olympiad_subjects":multi-hot (len = num OlympiadSubject values),  # shape: (13,)
    "target_countries": multi-hot (len = num Country values),          # shape: (30,)
    "career_track":     CareerTrack → integer index,      # shape: (1,)
    "school_tier":      SchoolTier → ordinal integer,     # shape: (1,)
    "income_category":  IncomeCategory → ordinal integer, # shape: (1,)
}

# Group C: Language — fixed-length vector, one slot per LanguageTest enum value
# Value = student's score for that test, or 0.0 if not taken
student_language = [score_toefl, score_ielts, score_topik, score_jlpt, score_delf, score_hsk]
# shape: (6,)

# Group D: Text — each field encoded by TF-Hub USE → 512-dim, then concatenated
student_text = concat([
    USE(personal_statement),      # (512,)
    USE(achievements_narrative),  # (512,)
    USE(future_goals),            # (512,)
])
# shape: (1536,)
```

**Scholarship features → same four groups, adapted:**

```python
# Group A: Numeric
scholarship_numeric = [
    min_age / 30.0,             # scholarships can require min age up to ~25–30
    max_age / 30.0,             # scholarships can allow max age up to ~30
    min_report_card_average / 100.0,
    min_major_subject_average / 100.0,
    float(requires_financial_need),
    float(requires_return_home_country),
    funding_coverage.coverage_count / 4.0,
    funding_coverage.monthly_stipend / 5000.0,
    selection_criteria.academic,
    selection_criteria.leadership,
    selection_criteria.olympiad,
    selection_criteria.extracurricular,
    selection_criteria.essay,
]
# shape: (13,)

# Group B: Categorical
scholarship_categorical = {
    "host_country":              Country → integer index,           # shape: (1,)
    "host_region":               HostRegion → integer index,        # shape: (1,)
    "eligible_nationalities":    multi-hot (len = num Country),     # shape: (30,)
    "eligible_hs_tracks":        multi-hot (len = num HighSchoolTrack), # shape: (5,)
    "eligible_fields":           multi-hot (len = num MajorField),  # shape: (14,)
    "preferred_school_tier":     SchoolTier → ordinal integer,      # shape: (1,)
    "max_income_category":       IncomeCategory → ordinal integer,  # shape: (1,)
    "career_track_preference":   CareerTrack → integer (0 for None),# shape: (1,)
}

# Group C: Language — min_score per LanguageTest, or 0.0 if not required
scholarship_language = [min_toefl, min_ielts, min_topik, min_jlpt, min_delf, min_hsk]
# shape: (6,)

# Group D: Text
scholarship_text = concat([
    USE(mission_statement),        # (512,)
    USE(target_recipient_profile), # (512,)
])
# shape: (1024,)
```

### 5.3 Pair Construction (`data/pair_builder.py`)

Reads from `artifacts/exports/*.csv`. Produces training pairs.

```python
from training.config import SEED, EVENT_WEIGHTS, EXPORTS_DIR
import numpy as np

rng = np.random.default_rng(SEED)
```

**Per-scholarship fallback logic — this is the core rule:**

For every scholarship in `scholarships.csv`, the pair builder checks whether it has
any rows in `interactions.csv`. The signal source is chosen per scholarship, not globally:

```
scholarships.csv → for each scholarship:

  Has at least one row in interactions.csv?
  │
  ├── YES → build interaction-based pairs for it
  │           Group its interaction rows by (student_id, scholarship_id)
  │           For each group:
  │             aggregated = sum(EVENT_WEIGHTS[event] for each event)
  │             clipped    = clip(aggregated, -1.0, 1.0)
  │             label      = (clipped + 1) / 2     # [-1, 1] → [0, 1]
  │             weight     = 1.0                    # real signal, full weight
  │
  └── NO  → fall back to pseudo-label pairs for it
              Run hard_filter(student, [this_scholarship]) for each student
              If eligible:
                create positive pair (label=0.6, weight=0.5)
                  label is 0.6 not 1.0 — synthetic, not confirmed by behavior
              Sample 3–5 students who are ineligible using rng:
                create negative pairs (label=0.0, weight=0.5)
```

This means a new scholarship with zero interaction history is never silently skipped —
it always gets pseudo-label pairs generated from eligibility logic, enters training,
and receives an embedding. Once interactions accumulate, the next retrain replaces
its pseudo-label pairs with real interaction-based ones automatically.

**Final pair structure:**
```python
{
    "student_features":     dict of tensors,  # from preprocessor
    "scholarship_features": dict of tensors,  # from preprocessor
    "label":  float,   # 0.0 to 1.0
    "weight": float,   # 0.5 for pseudo, 1.0 for interaction-based
}
```

### 5.4 New Scholarship Lifecycle

When the web backend adds a new scholarship to their database, no special action is
needed from the ML team. The scholarship flows through the pipeline naturally:

```
Web backend adds new scholarship to their DB
        │
        ▼
Next retrain payload includes it in scholarships.csv
(zero rows for it in interactions.csv)
        │
        ▼
pair_builder.py: no interactions found → pseudo-label pairs generated
        │
        ▼
train.py: model trains on pseudo-label pairs for this scholarship
        │
        ▼
embeddings.py: all scholarships in scholarships.csv get encoded,
including the new one → enters the embedding index
        │
        ▼
Service now surfaces the new scholarship to eligible students
        │
        ▼
Students interact with it (view, click, apply, reject)
        │
        ▼
Next retrain payload includes its interaction rows
→ pseudo-label pairs replaced by interaction-based pairs
→ embedding sharpens to reflect real fit signal
```

**Important caveat**: pseudo-labels are based on eligibility only — if a student passes
the hard filter, they get a positive pair. Eligibility ≠ good fit. A student might be
technically eligible but a poor match on softer criteria (essay weight, leadership focus,
career alignment). This means a new scholarship's embedding starts somewhat noisy —
discoverable but not yet well-ranked relative to similar scholarships. This is expected
and corrects itself as real interactions accumulate.

### 5.5 Retrain Trigger Agreement (with web backend team)

The web backend team controls when `POST /v1/retrain` is called. Agree on these two rules:

1. **On a regular cadence** — e.g. weekly, to incorporate accumulated interaction data.
2. **When new scholarships are added** — trigger a retrain payload promptly after a batch
   of new scholarships is added to the database. Without this, new scholarships sit in
   the DB but never receive an embedding and are never surfaced to students.

The ML service does not poll the web backend database. It only processes data when
a retrain payload is explicitly sent. This agreement must be documented between teams.

---

## 6. Model Architecture (`model/`)

### 6.1 Student Tower (`model/towers.py`)

```python
class StudentTower(tf.keras.Model):
    """
    Encodes a student profile into a fixed-size embedding vector.
    Architecture: parallel branches per feature group → concatenate → shared MLP.
    Output is L2-normalized so cosine similarity = dot product.
    """
    def __init__(self, embedding_dim=EMBEDDING_DIM):
        # Numeric branch:      Dense(64) → LayerNorm → ReLU
        # Categorical branch:  Embedding layers per field → flatten → Dense(64) → ReLU
        # Language branch:     Dense(32) → ReLU
        # Text branch:         USE encoder (frozen) → Dense(64) → ReLU
        # Merge: Concatenate all → Dense(256) → ReLU → Dense(embedding_dim) → L2 normalize

    def call(self, inputs):
        # Returns: student_embedding  shape (batch, embedding_dim)
```

### 6.2 Scholarship Tower (`model/towers.py`)

Same pattern as StudentTower, adapted for scholarship feature shapes. The two towers are **separate models — they do not share weights.** They share the same `embedding_dim` so outputs live in a comparable vector space.

### 6.3 Full Recommender (`model/recommender.py`)

```python
class TwoTowerRecommender(tf.keras.Model):
    """
    Wraps both towers. During training, takes (student, scholarship) pairs.
    During inference, towers are called independently.
    """
    def __init__(self):
        self.student_tower     = StudentTower(embedding_dim=EMBEDDING_DIM)
        self.scholarship_tower = ScholarshipTower(embedding_dim=EMBEDDING_DIM)
        self.compatibility_scorer = CompatibilityScorer()

    def call(self, inputs):
        student_emb     = self.student_tower(inputs["student"])
        scholarship_emb = self.scholarship_tower(inputs["scholarship"])

        # Overall relevance score
        overall_score = tf.reduce_sum(student_emb * scholarship_emb, axis=-1)
        overall_score = (overall_score + 1) / 2     # cosine [-1, 1] → [0, 1]

        # Compatibility breakdown
        breakdown = self.compatibility_scorer(inputs["student"], inputs["scholarship"])

        return {"score": overall_score, "breakdown": breakdown}
```

---

## 7. Custom Components (`model/components/`)

All three are **genuinely useful**, not just present for the sake of it.

### 7.1 `WeightedContrastiveLoss` (`components/loss.py`)

**Why**: Standard BCE treats all pairs equally. An `apply` event is much stronger signal than a `view`. This loss applies each pair's `weight` directly in the loss computation.

```python
class WeightedContrastiveLoss(tf.keras.losses.Loss):
    """
    Binary cross-entropy where each pair carries a weight reflecting
    the strength of its interaction signal (apply > click > view).
    Reject pairs push embeddings apart via negative label.
    """
    def call(self, y_true, y_pred, sample_weight=None):
        bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        if sample_weight is not None:
            bce = bce * sample_weight
        return tf.reduce_mean(bce)
```

### 7.2 `CompatibilityScorer` (`components/layers.py`)

**Why**: Produces the per-dimension breakdown (academic, leadership, language) required by the spec. Operates on raw feature slices — not embeddings — so scores are interpretable.

```python
class CompatibilityScorer(tf.keras.layers.Layer):
    """
    Produces a 3-dimensional compatibility breakdown:
      [0] Academic fit   — student GPA/scores vs. scholarship minimums
      [1] Leadership fit — student leadership count vs. scholarship leadership weight
      [2] Language fit   — student language scores vs. scholarship requirements

    Each output is in [0, 1]. Computed from raw features, not embeddings,
    so they are human-interpretable.
    """
    def call(self, student_features, scholarship_features):
        academic   = self._compute_academic_fit(student_features, scholarship_features)
        leadership = self._compute_leadership_fit(student_features, scholarship_features)
        language   = self._compute_language_fit(student_features, scholarship_features)
        return tf.stack([academic, leadership, language], axis=-1)
        # Returns shape (batch, 3)
```

### 7.3 `NDCGCallback` (`components/callbacks.py`)

**Why**: Accuracy and MAE don't capture ranking quality. NDCG@K checks whether good scholarships appear near the top. This is the primary early-stopping signal.

```python
class NDCGCallback(tf.keras.callbacks.Callback):
    """
    After each epoch, computes NDCG@10 on the validation set.
    Logs to TensorBoard. Used as the monitor for EarlyStopping.
    """
    def on_epoch_end(self, epoch, logs=None):
        ndcg = self._compute_ndcg_at_k(k=TOP_K)
        logs["val_ndcg@10"] = ndcg
        tf.summary.scalar("val_ndcg@10", ndcg, step=epoch)
```

---

## 8. Training Pipeline (`training/train.py`)

```python
import random, numpy as np, tensorflow as tf
from training.config import SEED, ...

# Apply seed to all sources of randomness — must be the first thing in the script
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
```

```
Step 1:  Load students and scholarships (generator/ for initial run, exports/ for retraining)
Step 2:  preprocessor.py → produce feature tensor dicts for all entities
Step 3:  pair_builder.py → build training pairs (pseudo + interaction-based)
Step 4:  tf.data.Dataset with shuffle(seed=SEED), batch(BATCH_SIZE), prefetch
Step 5:  Instantiate TwoTowerRecommender
Step 6:  model.compile(
           optimizer = Adam(lr=LEARNING_RATE),
           loss      = WeightedContrastiveLoss(),
           metrics   = [MAE],
         )
Step 7:  model.fit(
           train_dataset,
           validation_data = val_dataset,
           epochs          = MAX_EPOCHS,
           callbacks = [
             NDCGCallback(val_dataset),
             tf.keras.callbacks.TensorBoard(log_dir=LOGS_DIR),
             tf.keras.callbacks.EarlyStopping(
               monitor="val_ndcg@10", patience=EARLY_STOP_PATIENCE, mode="max"
             ),
             tf.keras.callbacks.ModelCheckpoint(save_best_only=True),
           ],
         )
Step 8:  Save model to MODEL_DIR / "model_v{N}/"
         train.py stops here — embedding generation, evaluation, and promotion
         are orchestrated by feedback/retrain_trigger.py (or called directly
         for the initial bootstrap run).
```

---

## 9. Artifact Versioning

### `artifacts/active.json` — Single source of truth

This file defines which model and embeddings version the live service uses.
It is **only updated after a new version passes evaluation.**

**Bootstrap (first deployment)**: Before the service can start, an initial training
run must be completed to produce `model_v1/` and `embeddings/v1/`. At the end of
that run, `active.json` must be written with `"model_version": "v1"` and
`"embeddings_version": "v1"`. The service must not be started until this file exists.
Add a startup guard in `service/main.py`:

```python
if not ACTIVE_FILE.exists():
    raise RuntimeError("active.json not found — run initial training before starting the service.")
```

```json
{
  "model_version": "v2",
  "embeddings_version": "v2"
}
```

### Versioned folder structure

```
artifacts/
  model/
    model_v1/     ← kept for rollback
    model_v2/     ← current live version
  embeddings/
    v1/
      scholarship_embeddings.npy
      scholarship_id_map.json
    v2/
      scholarship_embeddings.npy
      scholarship_id_map.json
```

### Promoting a new version (`feedback/retrain_trigger.py`)

```python
def promote_version(new_version: str):
    active = {
        "model_version":      new_version,
        "embeddings_version": new_version,
    }
    json.dump(active, open(ACTIVE_FILE, "w"), indent=2)
```

### Rolling back

Write the previous version string back to `active.json` and restart the service.
Old version folders are never deleted until manually cleaned up.

---

## 10. Inference Pipeline (`inference/`)

### 10.1 Hard Filter (`inference/hard_filter.py`)

Pure Python — no tensors, no model. Operates on Scholarship dataclass objects.
Returns a subset of scholarships the student is eligible for.

```python
def hard_filter(student: Student, scholarships: list[Scholarship]) -> list[Scholarship]:
    """
    Checks (in order):
      1. Nationality eligibility
      2. Age range (min_age ≤ student.age ≤ max_age)
      3. Target degree level
      4. High school track eligibility
      5. Minimum overall GPA
      6. Minimum major-subject average
      7. Mandatory language requirements
      8. Financial need requirement
      9. Income ceiling (max_family_income_category)
    Returns only scholarships that pass all checks.
    """
```

**Toggling**: Hard filter is on by default but can be disabled per-request (e.g. for development/evaluation):

```python
def recommend(self, student: Student, top_k: int = 10, use_hard_filter: bool = True):
    if use_hard_filter:
        eligible = hard_filter(student, list(self.scholarships.values()))
    else:
        eligible = list(self.scholarships.values())   # all scholarships
    # rest of pipeline unchanged
```

When disabled, the model scores all scholarships regardless of eligibility.
Use this during evaluation to check whether the model is learning eligibility signals on its own.
**Always leave it on in production.**

### 10.2 Embedding Index (`inference/embeddings.py`)

Called after every training run and when new scholarships are added.

```python
def build_scholarship_embeddings(model, scholarships, preprocessor):
    """
    Runs all scholarships through the scholarship tower.
    Saves:
      EMBEDDINGS_DIR / "v{N}" / "scholarship_embeddings.npy"  → shape (N, EMBEDDING_DIM)
      EMBEDDINGS_DIR / "v{N}" / "scholarship_id_map.json"     → {index: scholarship_id}
    Called by: retrain_trigger.py after every training run.
    """

def append_scholarship_embeddings(model, new_scholarships, preprocessor, version):
    """
    For new scholarships only — encodes only the new scholarships and appends to
    the existing .npy file. Updates scholarship_id_map.json.

    NOTE: This is a lightweight alternative to a full retrain. It is NOT called
    during the normal retrain flow (retrain_trigger.py always calls build_scholarship_embeddings).
    Use this only if the web backend team needs to surface new scholarships immediately
    without waiting for the next scheduled retrain. If used, the web backend must call
    a dedicated endpoint (e.g. POST /v1/scholarships/embed) that is outside the main
    retrain flow. Do not implement this endpoint unless the web backend team requests it —
    the default lifecycle (Section 5.4) routes everything through a full retrain.
    """
```

### 10.3 Predictor (`inference/predictor.py`)

Holds both the model and scholarship objects in memory. Loaded once on service startup.

```python
class Predictor:
    def __init__(self, artifacts_dir: Path, scholarship_objects: list[Scholarship]):
        active = json.load(open(artifacts_dir / "active.json"))

        model_path      = artifacts_dir / "model" / f"model_{active['model_version']}"
        embeddings_path = artifacts_dir / "embeddings" / active["embeddings_version"]

        self.model      = tf.keras.models.load_model(model_path)
        self.embeddings = np.load(embeddings_path / "scholarship_embeddings.npy")
        self.id_map     = json.load(open(embeddings_path / "scholarship_id_map.json"))
        self.index_map  = {sid: idx for idx, sid in self.id_map.items()}  # reverse lookup
        self.scholarships = {s.scholarship_id: s for s in scholarship_objects}
        self.version    = active["model_version"]
```

---

## 11. Inference: Inputs and Outputs

### API Input

```json
{
  "student": { ...full student profile fields... },
  "top_k": 10,
  "use_hard_filter": true
}
```

### Stage-by-stage flow

```
Stage 1 — Hard filter
  Input:  Student object + all Scholarship objects (in memory, e.g. 500)
  Output: eligible_scholarships list[Scholarship]   (e.g. 47)
          eligible_indices      list[int]            (e.g. [2, 17, 44, ...])

Stage 2 — Preprocessor (student only)
  Input:  Student object
  Output: dict of tensors:
    "numeric":      shape (1, 12)
    "nationality":  shape (1,)
    "hs_track":     shape (1,)
    "olympiad_lvl": shape (1,)
    "olympiad_sub": shape (1, 13)
    "target_ctrs":  shape (1, 30)
    "career_track": shape (1,)
    "school_tier":  shape (1,)
    "income_cat":   shape (1,)
    "language":     shape (1, 6)
    "text":         shape (1, 1536)

Stage 3 — Student tower forward pass
  Input:  tensor dict from Stage 2
  Output: student_embedding   shape (1, 128)   L2-normalized

Stage 4 — Embedding lookup + cosine similarity
  Input:  student_embedding   shape (1, 128)
          eligible_embs       shape (47, 128)  ← self.embeddings[eligible_indices]
  Output: scores              shape (47,)      values in [0, 1]

Stage 5 — CompatibilityScorer
  Input:  raw student feature slices + raw scholarship feature slices (not embeddings)
  Output: breakdown           shape (47, 3)
            [:, 0] = academic fit
            [:, 1] = leadership fit
            [:, 2] = language fit

Stage 6 — Sort + package
  Input:  scores (47,), breakdown (47, 3), eligible_scholarships list (47)
  Output: top_k ranked result objects
```

Note: **Only the student profile passes through the neural network at inference time.**
Scholarship embeddings are pre-computed — the scholarship tower is never called at query time.

### API Output

```json
{
  "results": [
    {
      "scholarship_id": "SCH_000003",
      "name": "MEXT Undergraduate",
      "score": 0.91,
      "breakdown": {
        "academic":    0.88,
        "leadership":  0.75,
        "language":    0.95
      }
    },
    {
      "scholarship_id": "SCH_000017",
      "name": "GKS Korean Government",
      "score": 0.84,
      "breakdown": {
        "academic":    0.91,
        "leadership":  0.60,
        "language":    0.80
      }
    }
  ],
  "total_eligible": 47,
  "model_version": "v2"
}
```

---

## 12. FastAPI Service (`service/`)

### Endpoints

```
POST   /v1/recommend
  Request:  { student: StudentSchema, top_k: int = 10, use_hard_filter: bool = true }
  Response: { results: [...], total_eligible: int, model_version: str }

POST   /v1/recommend/explain
  Request:  { student: StudentSchema, scholarship_id: str }
  Response: { score: float, breakdown: {...}, explanation_text: str (optional Gemini) }

POST   /v1/interactions
  Request:  { student_id: str, scholarship_id: str, event_type: str }
  Response: { recorded: true }

POST   /v1/retrain
  Request:  multipart/form-data
              - students.csv
              - scholarships.csv
              - interactions.csv
  Response: { status: "started", version: "v3" }

GET    /v1/scholarships
  Response: list of active scholarships (for UI display)

GET    /health
  Response: { status: "ok", model_version: "v2" }
```

### Retrain route (`service/routes/retrain.py`)

```python
@router.post("/v1/retrain")
async def trigger_retrain(
    background_tasks:  BackgroundTasks,
    students_csv:      UploadFile = File(...),
    scholarships_csv:  UploadFile = File(...),
    interactions_csv:  UploadFile = File(...),
):
    # Save uploaded files to staging area
    save_upload(students_csv,     EXPORTS_DIR / "students.csv")
    save_upload(scholarships_csv, EXPORTS_DIR / "scholarships.csv")
    save_upload(interactions_csv, EXPORTS_DIR / "interactions.csv")

    # Run retrain pipeline in background so API returns immediately
    new_version = retrain_trigger.get_next_version()
    background_tasks.add_task(retrain_trigger.run, EXPORTS_DIR)

    return { "status": "started", "version": new_version }
```

### API schemas (`service/schemas_api.py`)

Define Pydantic models here that are JSON-serializable versions of the domain dataclasses.
Keep these **separate** from domain schemas in `schemas/` — they serve different purposes
(API contract vs. internal model). Multi-value fields are plain lists here, not pipe-separated strings.

---

## 13. Feedback Loop (`feedback/retrain_trigger.py`)

```python
def run(exports_dir: Path):
    # Step 1: Build training pairs from exported CSVs
    pairs = pair_builder.build_pairs(exports_dir)

    # Step 2: Determine next version number
    new_version = get_next_version()   # e.g. "v3"

    # Step 3: Train new model
    train.run(pairs, model_version=new_version)
    # Saves to MODEL_DIR / "model_v3/"

    # Step 4: Build scholarship embedding index for new model
    embeddings.build(model_version=new_version)
    # Saves to EMBEDDINGS_DIR / "v3/"

    # Step 5: Evaluate
    ndcg = evaluate.compute_ndcg(model_version=new_version)

    # Step 6: Promote only if evaluation passes
    if ndcg >= NDCG_THRESHOLD:
        promote_version(new_version)
        # active.json now points to v3
        # service restarts / reloads predictor
    else:
        log(f"v{new_version} did not pass evaluation (NDCG={ndcg:.3f}). Keeping current version.")
        # active.json unchanged, old model stays live
```

**When retraining is triggered** — the ML service does not poll the web backend.
`POST /v1/retrain` is called explicitly by the web backend team. Two agreed triggers:

1. **Regular cadence** — e.g. weekly, to incorporate accumulated interaction data.
2. **When new scholarships are added** — trigger promptly after a batch of new scholarships
   is added to the web backend DB. Without this, new scholarships never receive an embedding
   and are never surfaced to students regardless of how long they sit in the database.

See Section 5.4 for the full new scholarship lifecycle and Section 5.5 for the trigger agreement details.

---

## 14. File Inventory

### Source files (you write these)

| File | Contains | Used by |
|---|---|---|
| `schemas/enums.py` | All enums | Imported everywhere |
| `schemas/student.py` | Student dataclass | preprocessor, hard_filter, pair_builder, API |
| `schemas/scholarship.py` | Scholarship, FundingCoverage, SelectionCriteria | preprocessor, hard_filter, pair_builder, embeddings, API |
| `data/generator/scholarships.py` | ~18 hardcoded Scholarship objects | train.py (initial), predictor.py (object store) |
| `data/generator/students.py` | Synthetic Student generation (uses SEED) | train.py (initial) |
| `data/preprocessor.py` | Dataclass → tensor dict conversion | pair_builder, train, embeddings, predictor |
| `data/pair_builder.py` | Reads exports CSVs → training pairs (uses SEED for negative sampling) | train.py |
| `model/text_encoder.py` | TF-Hub USE wrapper, frozen weights | towers.py |
| `model/towers.py` | StudentTower, ScholarshipTower | recommender.py |
| `model/recommender.py` | TwoTowerRecommender | train.py, predictor.py |
| `model/components/loss.py` | WeightedContrastiveLoss | train.py |
| `model/components/layers.py` | CompatibilityScorer | recommender.py |
| `model/components/callbacks.py` | NDCGCallback | train.py |
| `training/config.py` | SEED, all hyperparameters, all paths | Everything |
| `training/train.py` | Main training script (applies SEED first) | retrain_trigger.py |
| `training/evaluate.py` | NDCG@K, Hit Rate@K, MAE | NDCGCallback, train.py, retrain_trigger.py |
| `inference/hard_filter.py` | Eligibility gate — pure Python | predictor.py |
| `inference/embeddings.py` | Build/append scholarship embedding index | train.py, retrain_trigger.py |
| `inference/predictor.py` | Loads model + index + scholarships → recommend() + explain() | service/routes/recommend.py |
| `service/schemas_api.py` | Pydantic request/response models | All route files |
| `service/routes/recommend.py` | POST /v1/recommend, POST /v1/recommend/explain | main.py |
| `service/routes/interactions.py` | POST /v1/interactions | main.py |
| `service/routes/scholarships.py` | GET /v1/scholarships — returns in-memory scholarship list | main.py |
| `service/routes/retrain.py` | POST /v1/retrain — saves CSVs, triggers background retrain | main.py |
| `service/main.py` | FastAPI app init, router registration, Predictor init on startup | Entry point |
| `feedback/retrain_trigger.py` | Orchestrates: pair_builder → train → embeddings → evaluate → promote | retrain route |

### Artifact files (generated at runtime, never committed to git)

| File | Produced by | Used by |
|---|---|---|
| `artifacts/active.json` | retrain_trigger.py (promote_version) | predictor.py on startup |
| `artifacts/exports/students.csv` | POST /v1/retrain handler | pair_builder.py |
| `artifacts/exports/scholarships.csv` | POST /v1/retrain handler | pair_builder.py |
| `artifacts/exports/interactions.csv` | POST /v1/retrain handler | pair_builder.py |
| `artifacts/model/model_vN/` | train.py | predictor.py |
| `artifacts/embeddings/vN/scholarship_embeddings.npy` | embeddings.py | predictor.py — sliced every recommendation |
| `artifacts/embeddings/vN/scholarship_id_map.json` | embeddings.py | predictor.py — index ↔ scholarship_id mapping |
| `artifacts/logs/` | TensorBoard callback in train.py | tensorboard --logdir artifacts/logs/ |

---

## 15. Evaluation Targets

| Metric | Target | How measured |
|---|---|---|
| **NDCG@10** | ≥ 0.80 | NDCGCallback on held-out validation set — primary metric |
| **Hit Rate@5** | ≥ 0.70 | Does at least one relevant scholarship appear in top 5? |
| **MAE** (compatibility scores) | ≤ 0.10 | On labeled pairs where ground-truth score is known |
| Accuracy | ≥ 85% | Binary: did the top recommendation match a known positive? |

NDCG@10 is the primary metric and the gate for version promotion.
Accuracy and MAE are secondary — tracked to satisfy the original assignment requirement.

---

## 16. Optional Additions

### `tf.GradientTape` custom training loop
Replace `model.fit(...)` in `training/train.py` with a manual loop if the assignment requires it.
This adds explicit control over each training step but does not change model behavior.

```python
@tf.function
def train_step(batch, model, optimizer, loss_fn):
    with tf.GradientTape() as tape:
        predictions = model(batch, training=True)
        loss = loss_fn(
            batch["label"],
            predictions["score"],
            sample_weight=batch["weight"]
        )
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss
```

### Gemini Flash textual explanation (`/v1/recommend/explain`)
Build only after core scoring is working. For a student-scholarship pair, construct a prompt
from their profiles and compatibility scores, call Gemini Flash free API, return
2–3 actionable bullet points on how the student can improve their fit.

---

## 17. Implementation Order

Follow this order strictly. Each phase produces something testable before the next begins.

```
Phase 1 — Foundation
  [1]  schemas/enums.py, schemas/student.py, schemas/scholarship.py    (reuse existing)
  [2]  data/generator/scholarships.py, data/generator/students.py      (reuse existing)
  [3]  training/config.py                                               (SEED + all constants)
  [4]  data/preprocessor.py                                             (core bridge)
  [5]  inference/hard_filter.py                                         (testable standalone)

Phase 2 — Model
  [6]  model/text_encoder.py                                            (TF-Hub USE wrapper)
  [7]  model/towers.py                                                  (StudentTower, ScholarshipTower)
  [8]  model/components/layers.py                                       (CompatibilityScorer)
  [9]  model/components/loss.py                                         (WeightedContrastiveLoss)
  [10] model/recommender.py                                             (TwoTowerRecommender)

Phase 3 — Training
  [11] data/pair_builder.py                                             (reads exports CSVs)
  [12] model/components/callbacks.py                                    (NDCGCallback)
  [13] training/train.py                                                (seed applied first)
  [14] training/evaluate.py

Phase 4 — Inference
  [15] inference/embeddings.py                                          (build scholarship index)
  [16] inference/predictor.py

Phase 5 — Service
  [17] service/schemas_api.py
  [18] service/routes/recommend.py
  [19] service/routes/interactions.py
  [20] service/routes/retrain.py
  [21] service/main.py

Phase 6 — Feedback Loop
  [22] feedback/retrain_trigger.py

Phase 7 — Optional
  [23] tf.GradientTape training loop
  [24] Gemini Flash integration
```

---

## 18. Key Decisions Summary

| Decision | Choice | Reason |
|---|---|---|
| Model type | Two Tower | Cold-start, inference speed, natural interaction-pair mapping |
| Text encoding | TF-Hub USE (frozen) | Pre-trained, free, Keras-compatible, 512-dim sentence vectors |
| Training signal | Pseudo-labels → interaction-based | Bootstrap without data; improve over time |
| Hard filter | Keep, run before model, toggleable | Eligibility is compliance logic, not ML. Toggle off for evaluation only. |
| Custom loss | WeightedContrastiveLoss | Event types differ in signal strength — loss must reflect this |
| Custom layer | CompatibilityScorer | Interpretable breakdown scores; directly useful output |
| Custom callback | NDCGCallback | Accuracy/MAE don't capture ranking quality |
| Primary eval metric | NDCG@10 | Standard retrieval/ranking metric; gates version promotion |
| Pair dataset | Bootstrapped from eligibility, not manual | No annotation needed; interactions refine over time |
| Scholarship index | .npy + json map | Simple, portable, no external dependency |
| Artifact versioning | active.json + versioned folders | One source of truth; old versions kept for rollback |
| Retraining data | CSV payload via POST /v1/retrain | Web backend sends exports; ML service reads from artifacts/exports/ |
| Reproducibility | Single SEED=42 in config.py | One number, imported everywhere, never hardcoded elsewhere |