"""
Trainer untuk Two-Tower model dengan TensorFlow/Keras.

Komponen kustom:
- L2Normalize         : custom Keras Layer (di student_tower.py)
- sampled_softmax_loss_weighted : custom loss function
- BestModelCallback   : custom Keras Callback — save model terbaik + log metrics
"""
import os
import numpy as np
import tensorflow as tf


# ── Custom Loss Function ──────────────────────────────────────────────────────

def sampled_softmax_loss_weighted(
    stu_emb: tf.Tensor,
    sch_emb: tf.Tensor,
    weights: tf.Tensor,
    temperature: float = 0.1,
) -> tf.Tensor:
    """Custom loss: sampled softmax dengan per-sample weights.

    Args:
        stu_emb    : L2-normalized student embeddings, shape (N, 128)
        sch_emb    : L2-normalized scholarship embeddings, shape (N, 128)
        weights    : per-sample weights dari feedback type, shape (N,)
        temperature: softmax temperature

    Returns:
        Scalar loss.
    """
    logits = tf.matmul(stu_emb, sch_emb, transpose_b=True) / temperature  # (N, N)
    n      = tf.shape(stu_emb)[0]
    labels = tf.eye(n)

    loss_rows = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
    loss_cols = tf.nn.softmax_cross_entropy_with_logits(
        labels=tf.transpose(labels), logits=tf.transpose(logits)
    )
    per_sample = (loss_rows + loss_cols) / 2.0

    # Normalize weights supaya loss scale stabil
    w_norm = weights / (tf.reduce_mean(weights) + 1e-9)
    return tf.reduce_mean(w_norm * per_sample)


# ── Custom Callback ───────────────────────────────────────────────────────────

class BestModelCallback(tf.keras.callbacks.Callback):
    """Custom Keras Callback: simpan model terbaik berdasarkan Recall@K.

    - Menyimpan student_tower dan scholarship_tower dalam format .keras (full model)
      setiap kali Recall@K meningkat.
    - Mencetak log metrics per epoch ke console.
    - Menyimpan history metrics ke self.history untuk plotting.
    """

    def __init__(
        self,
        student_tower: tf.keras.Model,
        scholarship_tower: tf.keras.Model,
        checkpoint_dir: str,
        monitor: str = "val_recall",
    ):
        super().__init__()
        self.student_tower     = student_tower
        self.scholarship_tower = scholarship_tower
        self.checkpoint_dir    = checkpoint_dir
        self.monitor           = monitor
        self.best_value        = -np.inf
        self.best_epoch        = 0
        self.history           = {}

        os.makedirs(checkpoint_dir, exist_ok=True)

    def on_epoch_end(self, epoch: int, logs: dict = None):
        logs = logs or {}

        # Simpan ke history
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

        # Cek apakah metric membaik
        current = logs.get(self.monitor, None)
        if current is not None and current > self.best_value:
            self.best_value = current
            self.best_epoch = epoch + 1

            # Simpan full model dalam format .keras
            stu_path = os.path.join(self.checkpoint_dir, "student_tower_best.keras")
            sch_path = os.path.join(self.checkpoint_dir, "scholarship_tower_best.keras")
            self.student_tower.save(stu_path)
            self.scholarship_tower.save(sch_path)

            print(f"  ✓ Best {self.monitor}={current:.4f} — model saved (epoch {epoch + 1})")

    def on_train_end(self, logs=None):
        print(f"\nTraining selesai. Best {self.monitor}={self.best_value:.4f} "
              f"at epoch {self.best_epoch}")


# ── Trainer ───────────────────────────────────────────────────────────────────

class Trainer:
    def __init__(
        self,
        student_tower: tf.keras.Model,
        scholarship_tower: tf.keras.Model,
        optimizer: tf.keras.optimizers.Optimizer,
        temperature: float = 0.1,
        checkpoint_dir: str = "outputs/checkpoints",
    ):
        self.student_tower     = student_tower
        self.scholarship_tower = scholarship_tower
        self.optimizer         = optimizer
        self.temperature       = temperature
        self.checkpoint_dir    = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

    @tf.function
    def _train_step(self, stu_feat, sch_feat, weights):
        with tf.GradientTape() as tape:
            stu_emb = self.student_tower(stu_feat, training=True)
            sch_emb = self.scholarship_tower(sch_feat, training=True)
            loss    = sampled_softmax_loss_weighted(stu_emb, sch_emb, weights,
                                                    self.temperature)
        all_vars = (self.student_tower.trainable_variables +
                    self.scholarship_tower.trainable_variables)
        grads = tape.gradient(loss, all_vars)
        self.optimizer.apply_gradients(zip(grads, all_vars))
        return loss

    def train_epoch(self, dataset: tf.data.Dataset) -> float:
        losses = []
        for stu_feat, sch_feat, weights in dataset:
            loss = self._train_step(stu_feat, sch_feat, weights)
            losses.append(float(loss))
        return float(np.mean(losses))

    def eval_epoch(self, dataset: tf.data.Dataset) -> float:
        losses = []
        for stu_feat, sch_feat, weights in dataset:
            stu_emb = self.student_tower(stu_feat, training=False)
            sch_emb = self.scholarship_tower(sch_feat, training=False)
            loss    = sampled_softmax_loss_weighted(stu_emb, sch_emb, weights,
                                                    self.temperature)
            losses.append(float(loss))
        return float(np.mean(losses))

    def save_keras(self, tag: str = "best"):
        """Simpan full model dalam format .keras (TF production format)."""
        stu_path = os.path.join(self.checkpoint_dir, f"student_tower_{tag}.keras")
        sch_path = os.path.join(self.checkpoint_dir, f"scholarship_tower_{tag}.keras")
        self.student_tower.save(stu_path)
        self.scholarship_tower.save(sch_path)
        return stu_path, sch_path

    def load_keras(self, student_path: str, scholarship_path: str):
        """Load full model dari format .keras."""
        from src.models.student_tower import L2Normalize
        custom_objects = {"L2Normalize": L2Normalize}
        self.student_tower     = tf.keras.models.load_model(
            student_path, custom_objects=custom_objects)
        self.scholarship_tower = tf.keras.models.load_model(
            scholarship_path, custom_objects=custom_objects)
