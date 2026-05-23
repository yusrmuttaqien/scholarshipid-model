# Points to Remember: Scholarship.id Recommendation System

## Target Audience & Domain
- High school students (age ~16–18) targeting Bachelor's programs abroad.
- Balanced pairs across three classes: Match (≥0.7), In-Between (0.3–0.7), Not Match (<0.3).
  ✅ **[v1]** `generator.py` with balanced pair generation

## Architecture
- Two-tower architecture using TensorFlow/Keras Functional API.
- Student Tower and Scholarship Tower each produce a fixed-size embedding vector.
  ✅ **[v1]** `build_student_tower()` / `build_scholarship_tower()` in `train.py`

## Custom Components
- Implement at least one custom component: custom layer, loss function, or callback.
  ✅ **[v1]** `CosineSimilarity` custom layer in `train.py` (lines 90–102)

## Model Output & Inference
- Continuous relevance scores in [0, 1], not binary classification. Higher = better match.
  ✅ **[v1]** CosineSimilarity outputs sigmoid-normalized cosine similarity
- Supports broadcast inference: one student against N scholarships → array of N scores.
  ✅ **[v1]** `InferenceEngine.recommend()` with batched input + hard filter filtering
- Sort scores descending → top-K recommendations.
  ✅ **[v1]** Ranked in `inference.py` with `#1..#K` output

## Feedback Loop
- System supports feedback loop (apply, click, view, reject).
  ⚠️ **[v1]** `generate_feedback()` creates `feedback.csv`, but not yet integrated into training loop.

## Training & Monitoring (Optional)
- Training and evaluation loop from scratch using `tf.GradientTape`.
  ❌ — Not done. Using high-level `model.fit()`.
- Integrate with TensorBoard for monitoring and visualization.
  ✅ **[v1]** `TensorBoard` callback logging to `<v1>/models/logs/`
- Commit TensorBoard logs to the repository.
  ⚠️ — Logs generated, but not committed to repo.

## Performance Targets
- Model accuracy ≥ 85% (binary classification threshold at 0.5).
  📊 **[v1]** ~82.99% — Close to target.
- MAE ≤ 0.10 on validation set.
  ⚠️ [v1] Not met — In-between MAE=0.103, Not match MAE=0.092

## Versioning Notes

When adding new implementations:
1. Create a new version directory (e.g., `v2/`) with its own `README.md` and `POINTS_TO_REMEMBER.md`.
2. Add a row to this table for each feature — mark which version(s) implement it.
3. Compare versions side-by-side using the status markers:
   - ✅ **[v1]** Done in v1
   - ⚠️ **[v1] [v2]** Partial — improved in v2 with X change
   - ❌ — Not done yet
4. Keep the latest version's `README.md` as the canonical reference; older versions are for historical comparison.
