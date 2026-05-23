# Points to Remember: Scholarship.id Recommendation System

## Target Audience & Domain
- The scholarship is for high schoolers (let's say age around 16–18).
- Students target Bachelor's programs abroad.
- Dataset has balanced pairs across three classes: Match (≥0.7), In-Between (0.3–0.7), Not Match (<0.3).
  → [v1] Implemented in `generator.py` with balanced pair generation across all three classes

## Architecture
- Two-tower architecture using TensorFlow/Keras Functional API.
- Student Tower and Scholarship Tower each produce a fixed-size embedding vector.
  → [v1] `build_student_tower()` and `build_scholarship_tower()` in `train.py`

## Custom Components
- Implement at least one custom component: custom layer, custom loss function, or custom callback.
  → [v1] `CosineSimilarity` custom layer in `train.py` (lines 90-102)

## Model Output & Inference
- Model returns continuous relevance scores in range [0, 1], not binary classification. Higher score = better match.
  → [v1] CosineSimilarity layer outputs sigmoid-normalized cosine similarity in [0, 1]
- Supports broadcast inference: one student profile against N scholarships returns an array of N scores.
  → [v1] Model supports this via input shape flexibility (not yet a dedicated helper function)
- Sort scores descending → top-K recommendations.
  → [v1] Not yet implemented

## Feedback Loop
- System supports feedback loop: students can provide feedback on recommendations (apply, click, view, reject).
  → [v1] `generate_feedback()` in `generator.py` creates feedback.csv with feedback types
- Feedback is recorded with weights and used to improve / retrain the model.
  → [v1] Not yet integrated into training loop

## Training & Monitoring (Optional)
- Implement training and evaluation loop from scratch using `tf.GradientTape`.
  → [v1] Not implemented (using high-level `model.fit()` instead)
- Integrate with TensorBoard for monitoring and visualization.
  → [v1] `TensorBoard` callback in `train.py` logging to `OUTPUT_DIR/logs/`
- Commit TensorBoard logs to the repository.
  → [v1] Logs generated to `<vN>/models/logs/`

## Performance Targets
- Model accuracy ≥ 85% (binary classification threshold at 0.5).
  → [v1] Achieved ~82.99% (close to target)
- MAE ≤ 0.10 on validation set.
  → [v1] Not yet met (in-between MAE=0.103, not match MAE=0.092)