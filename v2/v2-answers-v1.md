# Scholarship Recommendation System — Final Spec
> For high school students seeking bachelor's scholarships abroad.
> This spec is written to be given directly to a coding model. Read it top-to-bottom before writing any code — every section builds on the previous one.

---

## 1. Architecture Decision

**Model type: Two Tower (Dual Encoder)**

### Why Two Tower
| Concern | Two Tower behavior |
|---|---|
| New scholarship cold-start | Encode new scholarship through its tower → append to embedding index. No retraining. |
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
       │ student_embedding          │ pre-computed
       └──────────┬─────────────────┘
                  │
         cosine_similarity(student_emb, scholarship_embs)
                  │
         ┌────────▼─────────┐
         │  Ranked Results  │ relevance scores [0, 1]
         │  + Breakdown     │ academic, leadership, language
         └──────────────────┘
```

---

## 2. Repository Structure

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
│   │   └── students.py         # Seed data and synthetic generation
│   ├── preprocessor.py         # Feature extraction: dataclass → numeric/text tensors
│   └── pair_builder.py         # Build training pairs from interactions + pseudo-labels
│
├── model/
│   ├── towers.py               # StudentTower and ScholarshipTower (Keras subclassing)
│   ├── recommender.py          # Full TwoTowerRecommender model (ties towers together)
│   ├── components/             # Custom Keras components
│   │   ├── loss.py             # Custom: WeightedContrastiveLoss
│   │   ├── layers.py           # Custom: CompatibilityScorer layer
│   │   └── callbacks.py        # Custom: NDCGCallback
│   └── text_encoder.py         # TF-Hub Universal Sentence Encoder wrapper
│
├── training/
│   ├── train.py                # Main training script
│   ├── config.py               # Hyperparameters, paths, constants
│   └── evaluate.py             # NDCG@K, Hit Rate@K, MAE evaluation
│
├── inference/
│   ├── embeddings.py           # Build and save scholarship embedding index
│   ├── hard_filter.py          # Pre-model eligibility gate
│   └── predictor.py            # Load model + index → produce ranked results
│
├── service/
│   ├── main.py                 # FastAPI app
│   ├── routes/
│   │   ├── recommend.py        # POST /v1/recommend, POST /v1/recommend/explain
│   │   └── interactions.py     # POST /v1/interactions
│   └── schemas_api.py          # Pydantic request/response models (separate from domain schemas)
│
├── feedback/
│   └── retrain_trigger.py      # Aggregates interaction table → triggers retraining pipeline
│
├── artifacts/                  # Generated at runtime, not committed to git
│   ├── model/                  # Saved Keras model (versioned: model_v1/, model_v2/, ...)
│   ├── embeddings/             # scholarship_embeddings.npy + scholarship_id_map.json
│   └── logs/                   # TensorBoard logs
│
└── notebooks/                  # Exploration only, not production code
    └── exploration.ipynb
```

---

## 3. Data Architecture

### 3.1 Feature Extraction (`data/preprocessor.py`)

The preprocessor converts dataclass instances into numeric tensors. This is the bridge between your domain schemas and the model.

**Student features → three groups:**

```python
# Group A: Numeric (normalized to [0, 1])
student_numeric = [
    age / 18.0,
    overall_report_card_average / 100.0,
    math_score / 100.0,
    english_score / 100.0,
    major_subject_average / 100.0,
    leadership_experience_count / 10.0,         # clip at 10
    volunteer_experience_count / 10.0,
    competition_wins_count / 10.0,
    float(needs_full_funding),
    float(can_self_fund_living),
    float(from_underrepresented_region),
    float(willing_to_return_home),
]

# Group B: Categorical (one-hot or integer-encoded)
student_categorical = {
    "nationality": Country enum → integer index,
    "high_school_track": HighSchoolTrack → integer index,
    "olympiad_level": OlympiadLevel → ordinal integer (0–5),
    "olympiad_subjects": multi-hot vector (length = num OlympiadSubject values),
    "target_countries": multi-hot vector (length = num Country values),
    "career_track": CareerTrack → integer index,
    "school_tier": SchoolTier → ordinal integer,
    "income_category": IncomeCategory → ordinal integer,
}

# Group C: Text (encoded by TF-Hub USE, produces 512-dim vector each)
student_texts = {
    "personal_statement": str,
    "achievements_narrative": str,
    "future_goals": str,
}

# Group D: Language proficiency (for each LanguageTest enum value, store the score or 0)
# This produces a fixed-length vector of size = len(LanguageTest)
student_language = [score_for_toefl, score_for_ielts, score_for_topik, ...]
```

**Scholarship features → same group pattern:**

```python
# Group A: Numeric
scholarship_numeric = [
    min_age / 18.0,
    max_age / 18.0,
    min_report_card_average / 100.0,
    min_major_subject_average / 100.0,
    float(requires_financial_need),
    float(requires_return_home_country),
    funding_coverage.coverage_count / 4.0,
    funding_coverage.monthly_stipend / 5000.0,  # normalize by max expected stipend
    selection_criteria.academic,
    selection_criteria.leadership,
    selection_criteria.olympiad,
    selection_criteria.extracurricular,
    selection_criteria.essay,
]

# Group B: Categorical
scholarship_categorical = {
    "host_country": Country → integer index,
    "host_region": HostRegion → integer index,
    "eligible_nationalities": multi-hot vector,
    "eligible_high_school_tracks": multi-hot vector,
    "eligible_fields": multi-hot vector,
    "preferred_school_tier": SchoolTier → ordinal integer,
    "max_family_income_category": IncomeCategory → ordinal integer,
    "career_track_preference": CareerTrack → integer (or 0 for None),
}

# Group C: Text
scholarship_texts = {
    "mission_statement": str,
    "target_recipient_profile": str,
}

# Group D: Language requirements (fixed-length: min_score for each LanguageTest, or 0)
scholarship_language = [min_score_toefl, min_score_ielts, ...]
```

### 3.2 Interaction Dataset Schema

Stored in a database table (or CSV for prototyping):

```
interactions table:
  interaction_id   UUID / auto-increment
  student_id       str
  scholarship_id   str
  event_type       enum: view | click | apply | reject
  timestamp        datetime
  session_id       str (optional, for grouping events per session)
```

**Event weights (tunable, stored in `training/config.py`):**
```python
EVENT_WEIGHTS = {
    "view":   0.1,
    "click":  0.3,
    "apply":  1.0,
    "reject": -0.5,
}
```

### 3.3 Training Pair Construction (`data/pair_builder.py`)

Two sources of pairs:

**Source 1 — Pseudo-labels (used before you have real interactions):**
```
For each student:
  Run hard filter against all scholarships
  → eligible_scholarships = scholarships student qualifies for
  → label these as positive pairs (weight = 0.6, not 1.0 — they're synthetic)
  → sample 3–5 random ineligible scholarships as negative pairs (weight = 0.0)
```

**Source 2 — Interaction-based labels (used once interactions exist):**
```
Group interactions by (student_id, scholarship_id)
Compute aggregated_weight = sum(EVENT_WEIGHTS[event] for each event)
Clip to [-1, 1] range
Normalize to [0, 1] via: (aggregated_weight + 1) / 2
→ This is the pair's training label
```

**Final training pair structure:**
```python
TrainingPair = {
    "student_features": {...},      # preprocessed student
    "scholarship_features": {...},  # preprocessed scholarship
    "label": float,                 # 0.0 to 1.0
    "weight": float,                # sample weight (higher for apply/reject pairs)
}
```

---

## 4. Model Architecture (`model/`)

### 4.1 Student Tower (`model/towers.py`)

```python
class StudentTower(tf.keras.Model):
    """
    Encodes a student profile into a fixed-size embedding vector.
    Architecture: parallel branches per feature group → concatenate → shared MLP.
    """
    def __init__(self, embedding_dim=128, ...):
        # Numeric branch: Dense(64) → LayerNorm → ReLU
        # Categorical branch: Embedding layers per field → flatten → Dense(64)
        # Language branch: Dense(32) → ReLU
        # Text branch: USE encoder (frozen) → Dense(64) → ReLU
        # Merge: Concatenate all → Dense(256) → ReLU → Dense(embedding_dim) → L2 normalize
        
    def call(self, inputs):
        # Returns: student_embedding shape (batch, embedding_dim)
```

### 4.2 Scholarship Tower (`model/towers.py`)

Same pattern as StudentTower, adapted for scholarship features. The two towers are **separate models** — they do not share weights. They share the same `embedding_dim` so their outputs are comparable in the same vector space.

### 4.3 Full Recommender (`model/recommender.py`)

```python
class TwoTowerRecommender(tf.keras.Model):
    """
    Wraps both towers. During training, takes (student, scholarship) pairs.
    During inference, the towers are called independently.
    """
    def __init__(self):
        self.student_tower = StudentTower(embedding_dim=128)
        self.scholarship_tower = ScholarshipTower(embedding_dim=128)
        self.compatibility_scorer = CompatibilityScorer()  # custom layer

    def call(self, inputs):
        student_emb = self.student_tower(inputs["student"])
        scholarship_emb = self.scholarship_tower(inputs["scholarship"])
        
        # Overall score
        overall_score = tf.reduce_sum(student_emb * scholarship_emb, axis=-1)
        overall_score = (overall_score + 1) / 2  # cosine [-1,1] → [0,1]
        
        # Compatibility breakdown (custom layer)
        breakdown = self.compatibility_scorer(inputs["student"], inputs["scholarship"])
        
        return {"score": overall_score, "breakdown": breakdown}
```

---

## 5. Custom Components (`model/components/`)

All three custom components are **genuinely useful**, not just present for the sake of it.

### 5.1 Custom Loss: `WeightedContrastiveLoss` (`components/loss.py`)

**Why**: Standard binary cross-entropy treats all pairs equally. But an `apply` event is much stronger signal than a `view`. This loss applies the pair's `sample_weight` directly in the loss computation.

```python
class WeightedContrastiveLoss(tf.keras.losses.Loss):
    """
    Binary cross-entropy where each pair has a weight reflecting
    the strength of the interaction signal (apply > click > view).
    Negative pairs (reject) contribute to pushing embeddings apart.
    """
    def call(self, y_true, y_pred, sample_weight=None):
        bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        if sample_weight is not None:
            bce = bce * sample_weight
        return tf.reduce_mean(bce)
```

### 5.2 Custom Layer: `CompatibilityScorer` (`components/layers.py`)

**Why**: Provides the per-dimension breakdown (academic, leadership, language) that your spec requires. This is a lightweight layer that takes selective student and scholarship feature slices and computes a 3-dim breakdown score.

```python
class CompatibilityScorer(tf.keras.layers.Layer):
    """
    Produces a 3-dimensional compatibility breakdown:
      [0]: Academic fit   — student GPA/scores vs. scholarship minimums
      [1]: Leadership fit — student leadership count vs. scholarship leadership weight
      [2]: Language fit   — student language scores vs. scholarship requirements

    Each output is [0, 1]. These are computed directly from features,
    not from embeddings, so they are interpretable.
    """
    def call(self, student_features, scholarship_features):
        academic_score  = self._compute_academic_fit(student_features, scholarship_features)
        leadership_score = self._compute_leadership_fit(student_features, scholarship_features)
        language_score  = self._compute_language_fit(student_features, scholarship_features)
        return tf.stack([academic_score, leadership_score, language_score], axis=-1)
```

### 5.3 Custom Callback: `NDCGCallback` (`components/callbacks.py`)

**Why**: The built-in Keras metrics (accuracy, MAE) don't measure ranking quality. NDCG@K checks whether the model puts good scholarships near the top of the ranked list. This callback computes NDCG@10 on the validation set after each epoch and logs it to TensorBoard.

```python
class NDCGCallback(tf.keras.callbacks.Callback):
    """
    After each epoch, evaluates NDCG@10 on the validation set.
    Logs to TensorBoard. Used as the primary early-stopping signal.
    """
    def on_epoch_end(self, epoch, logs=None):
        ndcg = self._compute_ndcg_at_k(k=10)
        logs["val_ndcg@10"] = ndcg
        tf.summary.scalar("val_ndcg@10", ndcg, step=epoch)
```

---

## 6. Training Pipeline (`training/train.py`)

```
Step 1: Load student and scholarship data (from generator or DB)
Step 2: Run preprocessor → produce feature tensors for all students and scholarships
Step 3: Build training pairs (pseudo-labels first; interaction-based when available)
Step 4: Create tf.data.Dataset with shuffling, batching, prefetching
Step 5: Instantiate TwoTowerRecommender
Step 6: Compile with:
          optimizer = Adam(lr=1e-3)
          loss      = WeightedContrastiveLoss()
          metrics   = [MAE]   ← track alongside NDCG from callback
Step 7: model.fit(
          train_dataset,
          validation_data=val_dataset,
          epochs=50,
          callbacks=[
            NDCGCallback(val_dataset),
            tf.keras.callbacks.TensorBoard(log_dir="artifacts/logs/"),
            tf.keras.callbacks.EarlyStopping(monitor="val_ndcg@10", patience=5, mode="max"),
            tf.keras.callbacks.ModelCheckpoint(save_best_only=True),
          ]
        )
Step 8: Save model to artifacts/model/model_v{N}/
Step 9: Run embeddings.py → generate scholarship_embeddings.npy + scholarship_id_map.json
```

### Training config (`training/config.py`)
```python
EMBEDDING_DIM      = 128
BATCH_SIZE         = 256
LEARNING_RATE      = 1e-3
MAX_EPOCHS         = 50
EARLY_STOP_PATIENCE = 5
TOP_K              = 10        # for NDCG@K evaluation
USE_MODEL_URL      = "https://tfhub.dev/google/universal-sentence-encoder/4"

EVENT_WEIGHTS = {
    "view":   0.1,
    "click":  0.3,
    "apply":  1.0,
    "reject": -0.5,
}

ARTIFACTS_DIR       = "artifacts/"
MODEL_DIR           = "artifacts/model/"
EMBEDDINGS_DIR      = "artifacts/embeddings/"
TENSORBOARD_LOG_DIR = "artifacts/logs/"
```

---

## 7. Inference Pipeline (`inference/`)

### 7.1 Hard Filter (`inference/hard_filter.py`)

Run **before** the model. Returns only scholarships the student is eligible for.

```python
def hard_filter(student: Student, scholarships: list[Scholarship]) -> list[Scholarship]:
    """
    Eliminates scholarships the student cannot apply to.
    Checks (in order):
      1. Nationality eligibility
      2. Age range
      3. Degree level eligibility
      4. High school track eligibility
      5. Minimum GPA (overall and major-subject)
      6. Language requirements (mandatory ones only)
      7. Financial need (if scholarship requires it)
      8. Income ceiling (max_family_income_category)
    Returns the subset of scholarships that pass all checks.
    """
```

### 7.2 Embedding Index (`inference/embeddings.py`)

Called after every training run (and when new scholarships are added):

```python
def build_scholarship_embeddings(model, scholarships, preprocessor):
    """
    Runs all scholarships through the scholarship tower.
    Saves:
      artifacts/embeddings/scholarship_embeddings.npy   → shape (N, embedding_dim)
      artifacts/embeddings/scholarship_id_map.json      → {index: scholarship_id}
    """
```

For new scholarships only (no retraining needed):
```python
def append_scholarship_embeddings(model, new_scholarships, preprocessor):
    """
    Encodes only the new scholarships and appends to existing .npy file.
    Updates id_map.json.
    """
```

### 7.3 Predictor (`inference/predictor.py`)

```python
class Predictor:
    def __init__(self, model_path, embeddings_dir, all_scholarships):
        # Load model, embedding matrix, id map, scholarship objects

    def recommend(self, student: Student, top_k: int = 10) -> list[RecommendationResult]:
        """
        1. Run hard_filter → eligible_scholarships + their embedding indices
        2. Encode student via student_tower
        3. Compute cosine similarity: student_emb · eligible_scholarship_embs
        4. Sort descending → top_k
        5. For each, compute compatibility breakdown via CompatibilityScorer
        6. Return ranked list with overall score + breakdown
        """

    def explain(self, student: Student, scholarship: Scholarship) -> CompatibilityBreakdown:
        """
        Returns detailed breakdown for a single student-scholarship pair.
        """
```

---

## 8. Feedback Loop (`feedback/retrain_trigger.py`)

```
Interaction events accumulate in DB
        │
        ▼
[Retrain trigger] — manual call or scheduled (e.g., weekly)
        │
        ▼
pair_builder.py:
  Fetch all interactions from DB
  Aggregate by (student_id, scholarship_id) → weighted label
  Merge with existing pseudo-label pairs (use interaction labels where available,
    fall back to pseudo-labels for student-scholarship pairs with no interactions)
        │
        ▼
train.py: retrain model on updated pairs
        │
        ▼
embeddings.py: rebuild full scholarship embedding index
        │
        ▼
New model_v{N}/ and embeddings saved
Service restarts (or hot-reloads) with new artifacts
```

**Important**: Keep `model_v{N-1}/` until the new model passes evaluation checks. This gives you a rollback path.

---

## 9. FastAPI Service (`service/`)

### Endpoints

```
POST   /v1/recommend
  Request:  { student: StudentSchema, top_k: int = 10 }
  Response: { results: [{ scholarship_id, name, score, breakdown: { academic, leadership, language } }] }

POST   /v1/recommend/explain
  Request:  { student: StudentSchema, scholarship_id: str }
  Response: { score, breakdown, explanation_text (optional: Gemini Flash) }

POST   /v1/interactions
  Request:  { student_id: str, scholarship_id: str, event_type: str }
  Response: { recorded: true }

GET    /v1/scholarships
  Response: list of active scholarships (for UI display)

GET    /health
  Response: { status: "ok", model_version: "v2" }
```

### API schemas (`service/schemas_api.py`)
Define Pydantic models here that mirror the domain dataclasses but are JSON-serializable. Keep these **separate** from the domain dataclasses in `schemas/` — they serve different purposes (API contract vs. internal model).

### Optional: Gemini Flash integration (`/v1/recommend/explain`)

If implemented, add after the core scoring is working:
```python
# For a student-scholarship pair, construct a prompt:
prompt = f"""
You are a scholarship advisor. Given this student profile and scholarship requirements,
provide 2–3 actionable suggestions for what this student can do to improve their fit.

Student: {student_summary}
Scholarship: {scholarship_summary}
Current compatibility scores: academic={breakdown.academic:.2f}, 
  leadership={breakdown.leadership:.2f}, language={breakdown.language:.2f}

Respond in 2–3 concise bullet points.
"""
# Call Gemini Flash API (free tier), return text alongside scores
```

---

## 10. Optional: `tf.GradientTape` Custom Training Loop

If implemented, replace `model.fit(...)` in `training/train.py` with:

```python
@tf.function
def train_step(batch, model, optimizer, loss_fn):
    with tf.GradientTape() as tape:
        predictions = model(batch, training=True)
        loss = loss_fn(batch["label"], predictions["score"], 
                       sample_weight=batch["weight"])
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

# Manual epoch/batch loop
for epoch in range(config.MAX_EPOCHS):
    for batch in train_dataset:
        loss = train_step(batch, model, optimizer, loss_fn)
    # Run NDCGCallback logic manually here
    # Log to TensorBoard manually with tf.summary
```

Only implement this if the assignment requires it — it adds boilerplate without changing the model behavior. The upside is explicit control over the training step (useful if you need custom gradient clipping or multi-loss combinations).

---

## 11. Evaluation Targets (revised)

| Metric | Target | How measured |
|---|---|---|
| **NDCG@10** | ≥ 0.80 | NDCGCallback on held-out validation set |
| **Hit Rate@5** | ≥ 0.70 | Does at least one relevant scholarship appear in top 5? |
| **MAE** (compatibility scores) | ≤ 0.10 | On labeled pairs where ground-truth score is known |
| Accuracy | ≥ 85% | Binary: did the top recommendation match a known positive? |

Note: NDCG@10 is the primary metric. Accuracy and MAE are secondary, tracked to satisfy the original assignment requirement.

---

## 12. Implementation Order

Follow this order strictly. Each step produces something usable before the next begins.

```
Phase 1 — Foundation
  [1] schemas/enums.py, schemas/student.py, schemas/scholarship.py   (reuse existing)
  [2] data/generator/scholarships.py, data/generator/students.py     (reuse existing)
  [3] data/preprocessor.py                                           (new — core bridge)
  [4] inference/hard_filter.py                                       (new — testable standalone)

Phase 2 — Model
  [5] model/text_encoder.py                                          (TF-Hub USE wrapper)
  [6] model/towers.py                                                (StudentTower, ScholarshipTower)
  [7] model/components/layers.py                                     (CompatibilityScorer)
  [8] model/components/loss.py                                       (WeightedContrastiveLoss)
  [9] model/recommender.py                                           (TwoTowerRecommender)

Phase 3 — Training
  [10] data/pair_builder.py                                          (pseudo-labels first)
  [11] training/config.py
  [12] model/components/callbacks.py                                 (NDCGCallback)
  [13] training/train.py                                             (main training script)
  [14] training/evaluate.py

Phase 4 — Inference
  [15] inference/embeddings.py                                       (build scholarship index)
  [16] inference/predictor.py

Phase 5 — Service
  [17] service/schemas_api.py
  [18] service/routes/recommend.py
  [19] service/routes/interactions.py
  [20] service/main.py

Phase 6 — Feedback Loop
  [21] feedback/retrain_trigger.py

Phase 7 — Optional
  [22] tf.GradientTape training loop (replaces model.fit in train.py)
  [23] Gemini Flash integration in /v1/recommend/explain
```

---

## 13. Key Decisions Summary (for reference)

| Decision | Choice | Reason |
|---|---|---|
| Model type | Two Tower | Cold-start handling, inference speed, natural interaction-pair mapping |
| Text encoding | TF-Hub USE (frozen) | Pre-trained, free, works with Keras, produces 512-dim sentence vectors |
| Training signal | Pseudo-labels → interaction-based | Bootstrap without interaction data; improve over time |
| Hard filters | Keep, run before model | Eligibility is a compliance concern, not an ML concern |
| Custom loss | WeightedContrastiveLoss | Event types have different signal strengths — loss must reflect this |
| Custom layer | CompatibilityScorer | Produces interpretable breakdown scores; directly useful output |
| Custom callback | NDCGCallback | Accuracy/MAE don't capture ranking quality; NDCG does |
| Primary eval metric | NDCG@10 | Standard retrieval/ranking metric |
| Pair dataset | Bootstrapped, not manual | Eligibility logic generates initial pairs; interactions refine them |
| Scholarship index | .npy + json map | Simple, portable, no external dependency |
| Model versioning | model_v{N}/ directories | Enables rollback before switching live service to new model |