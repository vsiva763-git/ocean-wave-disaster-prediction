"""Preprocessing utilities for multimodal ocean wave dataset."""
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

LABELS = ["NORMAL", "MODERATE", "GIANT"]


def encode_labels(labels: List[str]) -> Tuple[np.ndarray, LabelEncoder]:
    """Encode string labels to integers."""
    encoder = LabelEncoder()
    encoded = encoder.fit_transform(labels)
    return encoded, encoder


def decode_labels(encoded: np.ndarray, encoder: LabelEncoder) -> List[str]:
    return encoder.inverse_transform(encoded)


def stratified_split(
    df: pd.DataFrame,
    test_size: float = 0.15,
    val_size: float = 0.15,
    label_col: str = "label",
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create train/val/test splits with stratification."""
    train_df, temp_df = train_test_split(
        df, test_size=test_size + val_size, stratify=df[label_col], random_state=random_state
    )
    relative_val_size = val_size / (test_size + val_size)
    val_df, test_df = train_test_split(
        temp_df, test_size=1 - relative_val_size, stratify=temp_df[label_col], random_state=random_state
    )
    return train_df, val_df, test_df


def get_class_weights(labels: np.ndarray) -> Dict[int, float]:
    """Compute class weights to address imbalance."""
    unique = np.unique(labels)
    weights = compute_class_weight(class_weight="balanced", classes=unique, y=labels)
    return {cls: w for cls, w in zip(unique, weights)}


def build_sequence_feature_columns(seq_len: int = 12, n_features: int = 5) -> List[str]:
    """Generate column names for flattened sequence features.

    Example output for 12 timesteps and 5 features: ["t0_Hs", "t0_Hmax", ..., "t11_PeakWaveDirection"].
    """
    feature_names = ["Hs", "Hmax", "SST", "WindSpeed", "PeakWaveDirection"]
    cols: List[str] = []
    for t in range(seq_len):
        for feat in feature_names[:n_features]:
            cols.append(f"t{t}_{feat}")
    return cols


def assemble_sequences(df: pd.DataFrame, seq_cols: List[str]) -> np.ndarray:
    """Extract and reshape flattened sequence columns into (N, seq_len, n_features)."""
    seq = df[seq_cols].to_numpy()
    # infer seq_len and n_features
    total = seq.shape[1]
    # n_features is len of feature_names inferred by repeating pattern length
    # assume five features per timestep (consistent order)
    n_features = 5
    seq_len = total // n_features
    return seq.reshape(-1, seq_len, n_features)
