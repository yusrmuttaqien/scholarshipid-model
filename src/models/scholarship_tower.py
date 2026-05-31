"""Scholarship Tower: encode scholarship profile → 128-dim L2-normalized embedding."""
import tensorflow as tf
from .student_tower import L2Normalize


def build_scholarship_tower(input_dim: int = 509) -> tf.keras.Model:
    """
    Input(input_dim) → Dense(256, relu) → Dense(128, relu) → L2Normalize → emb(128)
    """
    inp = tf.keras.Input(shape=(input_dim,), name="scholarship_tower_input")
    x   = tf.keras.layers.Dense(256, activation="relu", name="scholarship_tower_dense_256")(inp)
    x   = tf.keras.layers.Dense(128, activation="relu", name="scholarship_tower_dense_128")(x)
    out = L2Normalize(name="scholarship_tower_l2norm")(x)
    return tf.keras.Model(inputs=inp, outputs=out, name="scholarship_tower")
