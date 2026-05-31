"""TwoTowerModel: wrapper tipis di atas student_tower dan scholarship_tower."""
import tensorflow as tf


class TwoTowerModel:
    """
    Bukan Keras Model subclass — hanya wrapper convenience.
    Training mengakses kedua tower secara langsung via trainer.
    """

    def __init__(self, student_tower: tf.keras.Model,
                 scholarship_tower: tf.keras.Model,
                 temperature: float = 0.1):
        self.student_tower     = student_tower
        self.scholarship_tower = scholarship_tower
        self.temperature       = temperature

    def encode_student(self, features, training: bool = False) -> tf.Tensor:
        return self.student_tower(features, training=training)

    def encode_scholarship(self, features, training: bool = False) -> tf.Tensor:
        return self.scholarship_tower(features, training=training)
