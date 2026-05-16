# Points to Remember: Scholarship.id Recommendation System

## Target Audience & Domain
- The scholarship is for high schoolers (let's say age around 16–18).
- Students target Bachelor's programs abroad.
- Dataset has balanced pairs across three classes: Match (≥0.7), In-Between (0.3–0.7), Not Match (<0.3).

## Architecture
- Two-tower architecture using TensorFlow/Keras Functional API.
- Student Tower and Scholarship Tower each produce a fixed-size embedding vector.

## Custom Components
- Implement at least one custom component: custom layer, custom loss function, or custom callback.

## Model Output & Inference
- Model returns continuous relevance scores in range [0, 1], not binary classification. Higher score = better match.- Supports broadcast inference: one student profile against N scholarships returns an array of N scores.
- Sort scores descending → top-K recommendations.

## Feedback Loop
- System supports feedback loop: students can provide feedback on recommendations (apply, click, view, reject).
- Feedback is recorded with weights and used to improve / retrain the model.

## Training & Monitoring (Optional)
- Implement training and evaluation loop from scratch using `tf.GradientTape`.
- Integrate with TensorBoard for monitoring and visualization.
- Commit TensorBoard logs to the repository.

## Performance Targets
- Model accuracy ≥ 85% (binary classification threshold at 0.5).
- MAE ≤ 0.10 on validation set.