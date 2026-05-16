# Points to Remember — Scholarship.id Recommendation System

## Target Audience & Domain
1. The scholarship is for high schoolers (SMA students, ages 16–18).
2. Students target Bachelor's programs abroad.
3. Dataset has balanced pairs across three classes: Match (≥0.7), In-Between (0.3–0.7), Not Match (<0.3).

## Architecture
4. Two-tower architecture using TensorFlow/Keras Functional API.
5. Embedding dimension: 64 per tower, hidden layers: 128 → 64.
6. Student Tower and Scholarship Tower each produce a fixed-size embedding vector.

## Custom Components
7. **CosineSimilarity** custom layer — connects the two towers, outputs via sigmoid activation.
8. **WeightedMSE** custom loss — class-aware weighting (In-Between weighted 1.5× for balanced gradients).
9. **ClassDistributionCallback** custom callback — tracks per-class RMSE/MAE during training.

## Model Output & Inference
10. Model returns continuous relevance scores in range [0, 1] — not binary classification. Higher score = better match.
11. Supports broadcast inference: one student profile against N scholarships returns an array of N scores.
12. Sort scores descending → top-K recommendations.

## Feedback Loop
13. System supports feedback loop — students can provide feedback on recommendations (apply, click, view, reject).
14. Feedback is recorded with weights and used to improve / retrain the model.