"""Two-Tower Recommendation System — Model Definition

Defines the two-tower neural network architecture using the Keras Functional API.
Includes custom components: CosineSimilarity layer, WeightedMSE loss,
and ClassDistributionCallback for per-class metric tracking.
"""

from typing import Dict, List, Optional, Tuple

import keras
import numpy as np
import tensorflow as tf
from tensorflow import keras as tf_keras


# ============================================================
# Custom Layers
# ============================================================


@keras.saving.register_keras_serializable(package="ScholarshipID")
class CosineSimilarity(keras.layers.Layer):
    """Custom layer computing cosine similarity between two embedding vectors.

    Takes two 64-dimensional embedding vectors (student and scholarship),
    L2-normalizes them, computes cosine similarity, and applies sigmoid
    to produce a relevance score in [0, 1].
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs, training=None):
        """Compute cosine similarity + sigmoid.

        Args:
            inputs: Tuple of (student_embedding, scholarship_embedding)
                Each is a tensor of shape (batch_size, embedding_dim)

        Returns:
            Tensor of shape (batch_size, 1) with values in [0, 1]
        """
        student_emb, scholarship_emb = inputs

        # L2-normalize both embeddings
        student_norm = tf.nn.l2_normalize(student_emb, axis=-1)
        scholarship_norm = tf.nn.l2_normalize(scholarship_emb, axis=-1)

        # Cosine similarity (element-wise product -> sum over embedding dim)
        cosine_sim = tf.reduce_sum(student_norm * scholarship_norm, axis=-1, keepdims=True)

        # Sigmoid maps [-1, 1] to roughly [0.27, 0.73]
        score = tf.sigmoid(cosine_sim)

        return score

    def get_config(self):
        config = super().get_config()
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


# ============================================================
# Loss Function
# ============================================================


@keras.saving.register_keras_serializable(package="ScholarshipID")
class WeightedMSE(keras.losses.Loss):
    """Weighted Mean Squared Error loss with class-aware weighting.

    Applies higher weight to In-Between (0.3-0.7) predictions since they
    are harder to predict and need stronger gradient signal.

    | Class | Relevance Range | Weight |
    |-------|-----------------|--------|
    | Match | >= 0.7          | 1.0    |
    | In-Between | 0.3 - 0.7 | 1.5    |
    | Not Match | < 0.3       | 1.0    |
    """

    def __init__(self, match_weight: float = 1.0, inbetween_weight: float = 1.5, not_match_weight: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.match_weight = match_weight
        self.inbetween_weight = inbetween_weight
        self.not_match_weight = not_match_weight

    def call(self, y_true, y_pred):
        """Compute weighted MSE.

        Args:
            y_true: Ground truth relevance scores, shape (batch_size, 1)
            y_pred: Predicted relevance scores, shape (batch_size, 1)

        Returns:
            Scalar weighted MSE loss.
        """
        # Squared error
        mse = tf.square(y_true - y_pred)

        # Determine class membership
        # Match: y_true >= 0.7
        match_mask = tf.cast(y_true >= 0.7, tf.float32)
        # In-Between: 0.3 <= y_true < 0.7
        inbetween_mask = tf.cast(
            tf.logical_and(y_true >= 0.3, y_true < 0.7),
            tf.float32,
        )
        # Not Match: y_true < 0.3
        not_match_mask = tf.cast(y_true < 0.3, tf.float32)

        # Apply weights
        weighted_mse = (
            match_mask * self.match_weight * mse
            + inbetween_mask * self.inbetween_weight * mse
            + not_match_mask * self.not_match_weight * mse
        )

        return tf.reduce_mean(weighted_mse)

    def get_config(self):
        config = super().get_config()
        config.update({
            "match_weight": self.match_weight,
            "inbetween_weight": self.inbetween_weight,
            "not_match_weight": self.not_match_weight,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


# ============================================================
# Custom Callback: ClassDistribution
# ============================================================


class ClassDistributionCallback(tf_keras.callbacks.Callback):
    """Custom callback that tracks per-class RMSE and MAE during training.

    After each epoch, evaluates predictions on the validation set and
    prints metrics broken down by relevance class.
    """

    def __init__(self, validation_data: tf.data.Dataset, log_frequency: int = 1):
        super().__init__()
        self.validation_data = validation_data
        self.log_frequency = log_frequency
        self.history = {
            "match_rmse": [],
            "match_mae": [],
            "inbetween_rmse": [],
            "inbetween_mae": [],
            "not_match_rmse": [],
            "not_match_mae": [],
        }

    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None):
        if (epoch + 1) % self.log_frequency != 0:
            return

        # Collect predictions and targets
        all_preds = []
        all_true = []

        for batch in self.validation_data:
            inputs, y_true = batch
            y_pred = self.model.predict_on_batch(list(inputs))
            # In TF 2.21+, predict_on_batch returns numpy arrays directly
            pred_arr = y_pred.numpy() if hasattr(y_pred, 'numpy') else y_pred
            true_arr = y_true.numpy() if hasattr(y_true, 'numpy') else y_true
            all_preds.append(pred_arr.flatten())
            all_true.append(true_arr.flatten())

        if not all_preds:
            return

        y_pred_all = np.concatenate(all_preds)
        y_true_all = np.concatenate(all_true)

        # Per-class metrics
        match_idx = y_true_all >= 0.7
        inbetween_idx = (y_true_all >= 0.3) & (y_true_all < 0.7)
        not_match_idx = y_true_all < 0.3

        metrics_str_parts = [f"Epoch {epoch + 1} per-class metrics:"]

        if match_idx.sum() > 0:
            rmse_m = np.sqrt(np.mean((y_true_all[match_idx] - y_pred_all[match_idx]) ** 2))
            mae_m = np.mean(np.abs(y_true_all[match_idx] - y_pred_all[match_idx]))
            self.history["match_rmse"].append(rmse_m)
            self.history["match_mae"].append(mae_m)
            metrics_str_parts.append(f"  match: RMSE={rmse_m:.4f}, MAE={mae_m:.4f}")

        if inbetween_idx.sum() > 0:
            rmse_ib = np.sqrt(np.mean((y_true_all[inbetween_idx] - y_pred_all[inbetween_idx]) ** 2))
            mae_ib = np.mean(np.abs(y_true_all[inbetween_idx] - y_pred_all[inbetween_idx]))
            self.history["inbetween_rmse"].append(rmse_ib)
            self.history["inbetween_mae"].append(mae_ib)
            metrics_str_parts.append(f"  in_between: RMSE={rmse_ib:.4f}, MAE={mae_ib:.4f}")

        if not_match_idx.sum() > 0:
            rmse_nm = np.sqrt(np.mean((y_true_all[not_match_idx] - y_pred_all[not_match_idx]) ** 2))
            mae_nm = np.mean(np.abs(y_true_all[not_match_idx] - y_pred_all[not_match_idx]))
            self.history["not_match_rmse"].append(rmse_nm)
            self.history["not_match_mae"].append(mae_nm)
            metrics_str_parts.append(f"  not_match: RMSE={rmse_nm:.4f}, MAE={mae_nm:.4f}")

        # Log to stdout
        combined = "\n".join(metrics_str_parts)
        print(f"\n{combined}\n")


# ============================================================
# Tower Builders
# ============================================================


def _get_keras():
    """Get the keras module (works with both standalone and tf.keras)."""
    return keras


def build_student_tower(
    preprocessors: Dict,
    embedding_dim: int = 64,
) -> tf_keras.Model:
    """Build the student tower model.

    The student tower encodes a student profile into an embedding vector.

    Input features:
        - 6 categorical features (StringLookup -> Embedding)
        - 8 numerical features (Normalization -> Dense)
        - 4 boolean features (direct float input)
        - 1 language proficiency vector (12-dim)

    Tower architecture:
        All Features -> Concatenate -> Dense(128, relu) -> BatchNorm
            -> Dense(64, relu) -> BatchNorm -> Student Embedding

    Args:
        preprocessors: Dict from create_datasets() containing lookups/normalizers.
        embedding_dim: Output embedding dimension (default 64).

    Returns:
        keras.Model: Student tower model.
    """
    lookups = preprocessors["student_lookups"]
    normalizers = preprocessors["student_normalizers"]
    categorical_cols = preprocessors["student_categorical_cols"]
    numerical_cols = preprocessors["student_numerical_cols"]
    boolean_cols = preprocessors["student_boolean_cols"]

    # Embedding dimensions: sqrt(cardinality) rule of thumb, capped at 16
    categorical_emb_dims = {
        "nationality": 8,       # 27 countries -> sqrt(27) ~ 5, but bumped to 8
        "high_school_track": 4,  # 5 tracks -> sqrt(5) ~ 2, bumped to 4
        "school_tier": 4,        # 7 tiers -> sqrt(7) ~ 3, bumped to 4
        "family_income_category": 4,  # 5 categories -> sqrt(5) ~ 2, bumped to 4
        "intended_career_track": 4,   # 6 tracks -> sqrt(6) ~ 2, bumped to 4
        "olympiad_level": 4,          # 6 levels -> sqrt(6) ~ 2, bumped to 4
    }

    input_layers = []
    embedding_layers = []
    concat_inputs = []

    # --- Categorical features ---
    for col in categorical_cols:
        inp = tf_keras.layers.Input(shape=(), dtype=tf.string, name=col)
        input_layers.append(inp)

        # Lookup integer indices
        if col in lookups:
            indexed = lookups[col](inp)
        else:
            lookup = tf_keras.layers.StringLookup(
                mask_token=None,
                num_oov_indices=1,
                output_mode="int",
            )
            lookup.adapt(tf.constant([""], dtype=tf.string))
            indexed = lookup(inp)

        vocab_size = lookups[col].vocabulary_size() if col in lookups else 1
        emb_dim = categorical_emb_dims.get(col, 4)

        embedding = tf_keras.layers.Embedding(
            input_dim=vocab_size,
            output_dim=emb_dim,
            name=f"emb_{col}",
        )(indexed)
        embedding_layers.append(embedding)
        concat_inputs.append(embedding)

    # --- Numerical features ---
    for col in numerical_cols:
        inp = tf_keras.layers.Input(shape=(), dtype=tf.float32, name=col)
        input_layers.append(inp)

        if col in normalizers:
            normalized = normalizers[col](inp)
        else:
            normalized = inp

        reshaped = tf_keras.layers.Reshape((1,), name=f"{col}_reshape")(normalized)
        concat_inputs.append(reshaped)

    # --- Boolean features ---
    for col in boolean_cols:
        inp = tf_keras.layers.Input(shape=(), dtype=tf.float32, name=col)
        input_layers.append(inp)
        reshaped = tf_keras.layers.Reshape((1,), name=f"{col}_reshape")(inp)
        concat_inputs.append(reshaped)

    # --- Language proficiency vector ---
    lang_dim = preprocessors.get("language_vector_dim", 12)
    lang_inp = tf_keras.layers.Input(shape=(lang_dim,), dtype=tf.float32, name="language_vector")
    input_layers.append(lang_inp)
    concat_inputs.append(lang_inp)

    # --- Concatenate all features ---
    if len(concat_inputs) > 1:
        concatenated = tf_keras.layers.Concatenate(name="student_concat")(concat_inputs)
    else:
        concatenated = concat_inputs[0]

    # --- Dense layers ---
    x = tf_keras.layers.Dense(128, activation="relu", name="student_dense_128")(concatenated)
    x = tf_keras.layers.BatchNormalization(name="student_bn_128")(x)

    x = tf_keras.layers.Dense(64, activation="relu", name="student_dense_64")(x)
    x = tf_keras.layers.BatchNormalization(name="student_bn_64")(x)

    # Final embedding
    embedding_output = tf_keras.layers.Dense(
        embedding_dim,
        activation=None,
        name="student_embedding",
    )(x)

    model = tf_keras.Model(
        inputs=input_layers,
        outputs=embedding_output,
        name="student_tower",
    )

    return model


def build_scholarship_tower(
    preprocessors: Dict,
    embedding_dim: int = 64,
) -> tf_keras.Model:
    """Build the scholarship tower model.

    The scholarship tower encodes a scholarship profile into an embedding vector.

    Input features:
        - 4 categorical features (StringLookup -> Embedding)
        - 6 numerical features (Normalization -> Dense)
        - 7 boolean features (direct float input)
        - 1 list field binary vector (46-dim: 27 countries + 5 tracks + 14 fields)

    Tower architecture:
        All Features -> Concatenate -> Dense(128, relu) -> BatchNorm
            -> Dense(64, relu) -> BatchNorm -> Scholarship Embedding

    Args:
        preprocessors: Dict from create_datasets() containing lookups/normalizers.
        embedding_dim: Output embedding dimension (default 64).

    Returns:
        keras.Model: Scholarship tower model.
    """
    lookups = preprocessors["scholarship_lookups"]
    normalizers = preprocessors["scholarship_normalizers"]
    categorical_cols = preprocessors["scholarship_categorical_cols"]
    numerical_cols = preprocessors["scholarship_numerical_cols"]
    boolean_cols = preprocessors["scholarship_boolean_cols"]

    categorical_emb_dims = {
        "host_region": 4,                # 6 regions -> sqrt(6) ~ 2, bumped to 4
        "preferred_school_tier": 4,      # 7 tiers -> sqrt(7) ~ 3, bumped to 4
        "career_track_preference": 4,     # 7 options (6+None) -> sqrt(7) ~ 3, bumped to 4
        "max_family_income_category": 4,  # 5 categories -> sqrt(5) ~ 2, bumped to 4
    }

    input_layers = []
    concat_inputs = []

    # --- Categorical features ---
    for col in categorical_cols:
        inp = tf_keras.layers.Input(shape=(), dtype=tf.string, name=f"scholarship_{col}")
        input_layers.append(inp)

        if col in lookups:
            indexed = lookups[col](inp)
        else:
            lookup = tf_keras.layers.StringLookup(
                vocabulary=[],
                mask_token=None,
                num_oov_indices=1,
                output_mode="int",
            )
            indexed = lookup(inp)

        vocab_size = lookups[col].vocabulary_size() if col in lookups else 1
        emb_dim = categorical_emb_dims.get(col, 4)

        embedding = tf_keras.layers.Embedding(
            input_dim=vocab_size,
            output_dim=emb_dim,
            name=f"scholarship_emb_{col}",
        )(indexed)
        concat_inputs.append(embedding)

    # --- Numerical features ---
    for col in numerical_cols:
        inp = tf_keras.layers.Input(shape=(), dtype=tf.float32, name=f"scholarship_{col}")
        input_layers.append(inp)

        if col in normalizers:
            normalized = normalizers[col](inp)
        else:
            normalized = inp

        reshaped = tf_keras.layers.Reshape((1,), name=f"scholarship_{col}_reshape")(normalized)
        concat_inputs.append(reshaped)

    # --- Boolean features ---
    for col in boolean_cols:
        inp = tf_keras.layers.Input(shape=(), dtype=tf.float32, name=f"scholarship_{col}")
        input_layers.append(inp)
        reshaped = tf_keras.layers.Reshape((1,), name=f"scholarship_{col}_reshape")(inp)
        concat_inputs.append(reshaped)

    # --- List field binary vector ---
    list_dim = preprocessors.get("list_vector_dim", 46)
    list_inp = tf_keras.layers.Input(shape=(list_dim,), dtype=tf.float32, name="scholarship_list_vector")
    input_layers.append(list_inp)
    concat_inputs.append(list_inp)

    # --- Concatenate all features ---
    if len(concat_inputs) > 1:
        concatenated = tf_keras.layers.Concatenate(name="scholarship_concat")(concat_inputs)
    else:
        concatenated = concat_inputs[0]

    # --- Dense layers ---
    x = tf_keras.layers.Dense(128, activation="relu", name="scholarship_dense_128")(concatenated)
    x = tf_keras.layers.BatchNormalization(name="scholarship_bn_128")(x)

    x = tf_keras.layers.Dense(64, activation="relu", name="scholarship_dense_64")(x)
    x = tf_keras.layers.BatchNormalization(name="scholarship_bn_64")(x)

    # Final embedding
    embedding_output = tf_keras.layers.Dense(
        embedding_dim,
        activation=None,
        name="scholarship_embedding",
    )(x)

    model = tf_keras.Model(
        inputs=input_layers,
        outputs=embedding_output,
        name="scholarship_tower",
    )

    return model


# ============================================================
# Model Assembly
# ============================================================


def build_model(
    preprocessors: Dict,
    embedding_dim: int = 64,
    learning_rate: float = 0.001,
) -> tf_keras.Model:
    """Build and compile the full two-tower recommendation model.

    Args:
        preprocessors: Dict from create_datasets() containing lookups/normalizers
            and feature column definitions.
        embedding_dim: Output embedding dimension (default 64).
        learning_rate: Initial learning rate for Adam optimizer (default 0.001).

    Returns:
        Compiled keras.Model for two-tower recommendation.
        Call with: model.predict([student_features_dict, scholarship_features_dict])
    """
    # Build towers
    student_tower = build_student_tower(preprocessors, embedding_dim)
    scholarship_tower = build_scholarship_tower(preprocessors, embedding_dim)

    # Use the tower's input tensors directly — they are already in the correct order
    # matching the order in which they were defined in build_student_tower()
    all_student_inputs = student_tower.inputs

    # Similarly for scholarship tower
    all_scholar_inputs = scholarship_tower.inputs

    # Pass through towers
    student_embedding = student_tower(all_student_inputs)
    scholarship_embedding = scholarship_tower(all_scholar_inputs)

    # Cosine similarity connection
    score = CosineSimilarity(name="cosine_similarity")([student_embedding, scholarship_embedding])

    # Build model
    all_inputs = all_student_inputs + all_scholar_inputs
    model = tf_keras.Model(
        inputs=all_inputs,
        outputs=score,
        name="two_tower_model",
    )

    # Compile
    model.compile(
        optimizer=tf_keras.optimizers.Adam(learning_rate=learning_rate),
        loss=WeightedMSE(),
        metrics=[
            tf_keras.metrics.RootMeanSquaredError(name="rmse"),
            tf_keras.metrics.MeanAbsoluteError(name="mae"),
        ],
    )

    return model, student_tower, scholarship_tower


# ============================================================
# Utility: Prepare batch for model call
# ============================================================


def prepare_input_batch(
    batch: Tuple,
    preprocessors: Dict,
) -> Tuple[List[tf.Tensor], tf.Tensor]:
    """Convert a dataset batch into the flattened input list the model expects.

    The model expects all student inputs (in order) followed by all scholarship inputs.
    This function takes a batch from the tf.data.Dataset and rearranges it.

    Args:
        batch: A tuple ((student_features_dict, scholarship_features_dict), relevance)
        preprocessors: Dict from create_datasets()

    Returns:
        (input_list, relevance_tensor)
    """
    (student_features, scholarship_features), relevance = batch

    student_inputs = []
    # Student categorical (in column order)
    for col in preprocessors["student_categorical_cols"]:
        student_inputs.append(student_features[col])
    # Student numerical
    for col in preprocessors["student_numerical_cols"]:
        student_inputs.append(student_features[col])
    # Student boolean
    for col in preprocessors["student_boolean_cols"]:
        student_inputs.append(student_features[col])
    # Student language vector
    student_inputs.append(student_features["language_vector"])

    scholarship_inputs = []
    # Scholarship categorical
    for col in preprocessors["scholarship_categorical_cols"]:
        scholarship_inputs.append(scholarship_features[col])
    # Scholarship numerical
    for col in preprocessors["scholarship_numerical_cols"]:
        scholarship_inputs.append(scholarship_features[col])
    # Scholarship boolean
    for col in preprocessors["scholarship_boolean_cols"]:
        scholarship_inputs.append(scholarship_features[col])
    # Scholarship list vector
    scholarship_inputs.append(scholarship_features["list_vector"])

    return student_inputs + scholarship_inputs, relevance