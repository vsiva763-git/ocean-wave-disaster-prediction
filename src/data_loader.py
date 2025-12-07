"""Data loading utilities for multimodal ocean wave prediction.

- Loads tabular metadata from CSV (labels + sequence features).
- Loads satellite images and aligns them with metadata rows.
- Reshapes time-series into (batch, seq_len, n_features).
- Provides optional normalization for sequences and image scaling to [0, 1].
"""
import os
from typing import List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

from preprocess import build_sequence_feature_columns

IMAGE_DEFAULT_SIZE = (128, 128)
SEQUENCE_LENGTH = 12
SEQUENCE_FEATURES = 5
LABEL_MAP = {0: "NORMAL", 1: "MODERATE", 2: "GIANT"}


def load_dataset_csv(csv_path: str) -> pd.DataFrame:
    """Load dataset metadata CSV. Expects columns for image filename, label, and sequence features."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    df = pd.read_csv(csv_path)
    return df


def load_image(img_path: str, image_size: Tuple[int, int] = IMAGE_DEFAULT_SIZE) -> np.ndarray:
    """Load and resize a single image, returning float32 array scaled to [0, 1]."""
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image not found: {img_path}")
    bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Failed to read image: {img_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, image_size)
    return resized.astype(np.float32) / 255.0


def load_image_batch(
    image_dir: str,
    image_filenames: List[str],
    image_size: Tuple[int, int] = IMAGE_DEFAULT_SIZE,
) -> np.ndarray:
    """Load a batch of images from a directory in the order of the provided filenames."""
    images = []
    for name in image_filenames:
        full_path = os.path.join(image_dir, name)
        images.append(load_image(full_path, image_size))
    return np.stack(images, axis=0)


def reshape_sequences(
    seq_values: np.ndarray,
    seq_len: int = SEQUENCE_LENGTH,
    n_features: int = SEQUENCE_FEATURES,
) -> np.ndarray:
    """Ensure sequences are shaped as (N, seq_len, n_features)."""
    arr = np.asarray(seq_values)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    expected = seq_len * n_features
    if arr.shape[1] != expected:
        raise ValueError(f"Expected {expected} features per sample, got {arr.shape[1]}")
    return arr.reshape(-1, seq_len, n_features)


def normalize_sequences(
    sequences: np.ndarray,
    scaler: Optional[StandardScaler] = None,
) -> Tuple[np.ndarray, StandardScaler]:
    """Normalize sequences feature-wise using StandardScaler.

    Flattens sequence time dimension, fits/uses scaler, then reshapes back.
    """
    batch, seq_len, n_features = sequences.shape
    flat = sequences.reshape(batch, seq_len * n_features)
    if scaler is None:
        scaler = StandardScaler()
        scaled = scaler.fit_transform(flat)
    else:
        scaled = scaler.transform(flat)
    return scaled.reshape(batch, seq_len, n_features), scaler


def build_tf_dataset(
    images: np.ndarray,
    sequences: np.ndarray,
    labels: np.ndarray,
    batch_size: int = 32,
    shuffle: bool = True,
) -> tf.data.Dataset:
    """Create a tf.data.Dataset from numpy arrays."""
    ds = tf.data.Dataset.from_tensor_slices(((images, sequences), labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(images), reshuffle_each_iteration=True)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def split_features_labels(
    df: pd.DataFrame,
    image_col: str = "image",
    label_col: str = "label",
    seq_feature_cols: Optional[List[str]] = None,
) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """Extract filenames, sequence matrix, and labels from the dataframe."""
    if seq_feature_cols is None:
        seq_feature_cols = build_sequence_feature_columns(
            seq_len=SEQUENCE_LENGTH, n_features=SEQUENCE_FEATURES
        )
    image_names = df[image_col].tolist()
    sequences = df[seq_feature_cols].to_numpy()
    labels = df[label_col].to_numpy()
    return image_names, sequences, labels


def decode_predictions(probabilities: np.ndarray) -> List[str]:
    """Convert softmax probabilities to human-readable labels."""
    idx = np.argmax(probabilities, axis=1)
    return [LABEL_MAP.get(i, str(i)) for i in idx]
