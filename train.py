"""Two-Tower Recommendation System — Training Script

Trains the two-tower model with proper callbacks and monitoring.
Saves the best model in .keras format for production use.
"""

import os
import sys

import tensorflow as tf
from tensorflow import keras

from data import create_datasets
from model import (
    build_model,
    ClassDistributionCallback,
    CosineSimilarity,
    WeightedMSE,
    prepare_input_batch,
)


def main(
    students_csv: str = "./datasets_two_tower/students.csv",
    scholarships_csv: str = "./datasets_two_tower/scholarships.csv",
    pairs_csv: str = "./datasets_two_tower/pairs.csv",
    batch_size: int = 2048,
    epochs: int = 50,
    embedding_dim: int = 64,
    learning_rate: float = 0.001,
    model_save_path: str = "./two_tower_model.keras",
    log_dir: str = "./logs",
):
    """Run the full training pipeline.

    Args:
        students_csv: Path to students CSV.
        scholarships_csv: Path to scholarships CSV.
        pairs_csv: Path to pairs CSV.
        batch_size: Training batch size.
        epochs: Maximum number of epochs.
        embedding_dim: Dimension of tower embeddings.
        learning_rate: Initial learning rate.
        model_save_path: Where to save the trained model.
        log_dir: TensorBoard log directory.
    """
    print("=" * 60)
    print("Two-Tower Recommendation System — Training Pipeline")
    print("=" * 60)
    print(f"Configuration:")
    print(f"  Batch size:      {batch_size:,}")
    print(f"  Max epochs:      {epochs}")
    print(f"  Embedding dim:   {embedding_dim}")
    print(f"  Learning rate:   {learning_rate}")
    print(f"  Model save path: {model_save_path}")
    print("=" * 60)

    # ── Step 1: Create datasets ─────────────────────────────
    print("\n[Step 1] Loading data and creating tf.data.Datasets...")
    train_ds, val_ds, test_ds, preprocessors = create_datasets(
        students_csv=students_csv,
        scholarships_csv=scholarships_csv,
        pairs_csv=pairs_csv,
        batch_size=batch_size,
    )

    # Print dataset sizes
    train_size = sum(1 for _ in train_ds.unbatch())
    val_size = sum(1 for _ in val_ds.unbatch())
    test_size = sum(1 for _ in test_ds.unbatch())

    print(f"  Training pairs:   {train_size:,}")
    print(f"  Validation pairs: {val_size:,}")
    print(f"  Test pairs:       {test_size:,}")

    # ── Step 2: Build model ─────────────────────────────────
    print("\n[Step 2] Building two-tower model...")
    model, student_tower, scholarship_tower = build_model(
        preprocessors=preprocessors,
        embedding_dim=embedding_dim,
        learning_rate=learning_rate,
    )

    model.summary()

    # ── Step 3: Set up callbacks ────────────────────────────
    print("\n[Step 3] Setting up callbacks...")

    callbacks = [
        # Early stopping to prevent overfitting
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            min_delta=0.001,
            restore_best_weights=True,
            verbose=1,
        ),
        # Reduce learning rate when plateauing
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        ),
        # Per-class metrics tracking
        ClassDistributionCallback(
            validation_data=val_ds,
            log_frequency=1,
        ),
        # TensorBoard logging
        keras.callbacks.TensorBoard(
            log_dir=log_dir,
            histogram_freq=0,
            write_graph=True,
            update_freq="epoch",
        ),
        # Model checkpoint (save best)
        keras.callbacks.ModelCheckpoint(
            filepath=model_save_path.replace(".keras", "_checkpoint.keras"),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
    ]

    # ── Step 4: Calculate steps ─────────────────────────────
    # Our datasets are finite (from_generator), so we must specify steps
    steps_per_epoch = train_size // batch_size
    validation_steps = val_size // batch_size

    print(f"\n  Steps per epoch: {steps_per_epoch}")
    print(f"  Validation steps: {validation_steps}")

    # ── Step 5: Train model ─────────────────────────────────
    print("\n[Step 5] Starting training...")
    print("=" * 60)

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        callbacks=callbacks,
        verbose=1,
    )

    # ── Step 6: Evaluate on test set ────────────────────────
    print("\n[Step 6] Evaluating on held-out test set...")
    print("=" * 60)

    # Collect test predictions
    all_preds = []
    all_true = []

    for batch in test_ds:
        inputs, y_true = batch
        y_pred = model.predict_on_batch(list(inputs))
        pred_arr = y_pred.numpy() if hasattr(y_pred, 'numpy') else y_pred
        true_arr = y_true.numpy() if hasattr(y_true, 'numpy') else y_true
        all_preds.append(pred_arr.flatten())
        all_true.append(true_arr.flatten())

    y_pred_all = np.concatenate(all_preds)
    y_true_all = np.concatenate(all_true)

    # Overall metrics
    overall_rmse = np.sqrt(np.mean((y_true_all - y_pred_all) ** 2))
    overall_mae = np.mean(np.abs(y_true_all - y_pred_all))

    print(f"\nTest Set Results:")
    print(f"  Overall RMSE: {overall_rmse:.4f}  (Target: <= 0.15)")
    print(f"  Overall MAE:  {overall_mae:.4f}  (Target: <= 0.02)")

    # Per-class metrics
    match_idx = y_true_all >= 0.7
    inbetween_idx = (y_true_all >= 0.3) & (y_true_all < 0.7)
    not_match_idx = y_true_all < 0.3

    if match_idx.sum() > 0:
        rmse_m = np.sqrt(np.mean((y_true_all[match_idx] - y_pred_all[match_idx]) ** 2))
        mae_m = np.mean(np.abs(y_true_all[match_idx] - y_pred_all[match_idx]))
        print(f"  Match (n={match_idx.sum():,}):     RMSE={rmse_m:.4f}, MAE={mae_m:.4f}")

    if inbetween_idx.sum() > 0:
        rmse_ib = np.sqrt(np.mean((y_true_all[inbetween_idx] - y_pred_all[inbetween_idx]) ** 2))
        mae_ib = np.mean(np.abs(y_true_all[inbetween_idx] - y_pred_all[inbetween_idx]))
        print(f"  In-Between (n={inbetween_idx.sum():,}): RMSE={rmse_ib:.4f}, MAE={mae_ib:.4f}")

    if not_match_idx.sum() > 0:
        rmse_nm = np.sqrt(np.mean((y_true_all[not_match_idx] - y_pred_all[not_match_idx]) ** 2))
        mae_nm = np.mean(np.abs(y_true_all[not_match_idx] - y_pred_all[not_match_idx]))
        print(f"  Not Match (n={not_match_idx.sum():,}): RMSE={rmse_nm:.4f}, MAE={mae_nm:.4f}")

    # Accuracy at binary threshold 0.5
    binary_preds = (y_pred_all >= 0.5).astype(np.float32)
    binary_true = (y_true_all >= 0.5).astype(np.float32)
    accuracy = np.mean(binary_preds == binary_true)
    print(f"\n  Binary Accuracy (threshold=0.5): {accuracy:.4f}  (Target: >= 0.85)")

    # ── Step 7: Save model ──────────────────────────────────
    print(f"\n[Step 7] Saving model to {model_save_path}...")
    model.save(model_save_path)
    print("Model saved successfully!")

    # Also save the preprocessors for inference
    import pickle
    preprocessor_path = model_save_path.replace(".keras", "_preprocessors.pkl")
    with open(preprocessor_path, "wb") as f:
        pickle.dump(preprocessors, f)
    print(f"Preprocessors saved to {preprocessor_path}")

    # ── Final Summary ───────────────────────────────────────
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"\nBest model saved at: {model_save_path}")
    print(f"Preprocessors saved at: {preprocessor_path}")
    print(f"TensorBoard logs at: {log_dir}")
    print("\nTo view TensorBoard:")
    print(f"  tensorboard --logdir {log_dir}")
    print("\nTo load the saved model:")
    print("  model = keras.models.load_model('" + model_save_path + "',")
    print("      custom_objects={")
    print("          'CosineSimilarity': CosineSimilarity,")
    print("          'WeightedMSE': WeightedMSE,")
    print("      })")


if __name__ == "__main__":
    import numpy as np

    # Parse command-line overrides if provided
    kwargs = {}
    for arg in sys.argv[1:]:
        key, _, value = arg.partition("=")
        key = key.lstrip("-").replace("-", "_")
        if key == "batch_size":
            kwargs["batch_size"] = int(value)
        elif key == "epochs":
            kwargs["epochs"] = int(value)
        elif key == "embedding_dim":
            kwargs["embedding_dim"] = int(value)
        elif key == "learning_rate":
            kwargs["learning_rate"] = float(value)

    main(**kwargs)