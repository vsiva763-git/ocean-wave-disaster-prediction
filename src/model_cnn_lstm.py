"""Model definitions for multimodal CNN + LSTM fusion.

Two backbones provided:
- `simple` lightweight CNN
- `mobilenet_v2` using pretrained ImageNet weights
"""
from typing import Literal, Tuple

import tensorflow as tf
from tensorflow.keras import layers, models


def _simple_cnn(image_shape: Tuple[int, int, int]) -> tf.keras.Model:
    inputs = layers.Input(shape=image_shape, name="image_input")
    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    return models.Model(inputs, x, name="simple_cnn_backbone")


def _mobilenet_v2(image_shape: Tuple[int, int, int]) -> tf.keras.Model:
    base = tf.keras.applications.MobileNetV2(
        input_shape=image_shape,
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False  # allow fine-tuning later if desired
    inputs = base.input
    x = base.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    return models.Model(inputs, x, name="mobilenet_v2_backbone")


def build_multimodal_model(
    image_shape: Tuple[int, int, int] = (128, 128, 3),
    seq_len: int = 12,
    seq_features: int = 5,
    num_classes: int = 3,
    backbone: Literal["simple", "mobilenet_v2"] = "simple",
) -> tf.keras.Model:
    """Construct the CNN + LSTM fusion model."""
    if backbone == "simple":
        cnn = _simple_cnn(image_shape)
    elif backbone == "mobilenet_v2":
        cnn = _mobilenet_v2(image_shape)
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")

    image_inputs = cnn.input
    image_features = cnn.output

    seq_inputs = layers.Input(shape=(seq_len, seq_features), name="sequence_input")
    x_seq = layers.Masking()(seq_inputs)
    x_seq = layers.LSTM(128, return_sequences=True)(x_seq)
    x_seq = layers.LSTM(64)(x_seq)
    x_seq = layers.Dropout(0.3)(x_seq)

    fused = layers.Concatenate(name="fusion")([image_features, x_seq])
    x = layers.Dense(128, activation="relu")(fused)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(64, activation="relu")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="risk_output")(x)

    model = models.Model(inputs=[image_inputs, seq_inputs], outputs=outputs, name="cnn_lstm_fusion")
    return model


def hazard_probability_index(probabilities: tf.Tensor) -> tf.Tensor:
    """Compute Hazard Probability Index (HPI) as a weighted softmax expectation.

    Weights: NORMAL=0.0, MODERATE=0.5, GIANT=1.0
    """
    weights = tf.constant([0.0, 0.5, 1.0], dtype=probabilities.dtype)
    return tf.reduce_sum(probabilities * weights, axis=-1, name="hazard_probability_index")
