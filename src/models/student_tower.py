"""Student Tower: encode student profile → 128-dim L2-normalized embedding."""
import tensorflow as tf


class L2Normalize(tf.keras.layers.Layer):
    """L2-normalize embedding supaya dot product = cosine similarity."""
    def call(self, x):
        return tf.math.l2_normalize(x, axis=-1)


def build_student_tower(input_dim: int = 506) -> tf.keras.Model:
    """
    Input(input_dim) → Dense(256, relu) → Dense(128, relu) → L2Normalize → emb(128)
    """
    inp = tf.keras.Input(shape=(input_dim,), name="student_tower_input")
    x   = tf.keras.layers.Dense(256, activation="relu", name="student_tower_dense_256")(inp)
    x   = tf.keras.layers.Dense(128, activation="relu", name="student_tower_dense_128")(x)
    out = L2Normalize(name="student_tower_l2norm")(x)
    return tf.keras.Model(inputs=inp, outputs=out, name="student_tower")
