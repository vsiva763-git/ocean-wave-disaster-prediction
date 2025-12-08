"""Utilities to prepare real-time data as model-ready tensors.

Converts live data feeds into the sequence format expected by the
CNN+LSTM multimodal model.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from joblib import load

LOGGER = logging.getLogger(__name__)


def prepare_sequence_tensor(
    data: pd.DataFrame,
    feature_columns: List[str],
    seq_len: int = 12,
    scaler_path: Optional[str] = None,
) -> Tuple[np.ndarray, Optional[object]]:
    """Prepare time-series data as model-ready sequence tensor.
    
    Args:
        data: DataFrame with time index and feature columns
        feature_columns: List of feature names in expected order
        seq_len: Sequence length expected by model
        scaler_path: Path to fitted StandardScaler (optional)
    
    Returns:
        Tuple of:
            - sequence array with shape (1, seq_len, n_features)
            - scaler object (if loaded, else None)
    """
    # Sort by time
    data = data.sort_index()
    
    # Select only the required features
    available_features = [f for f in feature_columns if f in data.columns]
    missing_features = [f for f in feature_columns if f not in data.columns]
    
    if missing_features:
        LOGGER.warning(f"Missing features: {missing_features}")
    
    # Fill missing features with zeros or NaN
    for feat in missing_features:
        data[feat] = 0.0
    
    # Ensure correct column order
    feature_data = data[feature_columns].copy()
    
    # Handle missing values
    feature_data = feature_data.interpolate(method='linear').fillna(method='bfill').fillna(method='ffill').fillna(0.0)
    
    # Take the most recent seq_len timesteps
    if len(feature_data) < seq_len:
        LOGGER.warning(f"Data has only {len(feature_data)} timesteps, need {seq_len}. Padding with zeros.")
        # Pad with zeros at the beginning
        pad_length = seq_len - len(feature_data)
        pad_df = pd.DataFrame(
            np.zeros((pad_length, len(feature_columns))),
            columns=feature_columns,
        )
        feature_data = pd.concat([pad_df, feature_data], ignore_index=True)
    else:
        # Take last seq_len rows
        feature_data = feature_data.tail(seq_len)
    
    # Convert to numpy array
    sequence = feature_data.values  # shape: (seq_len, n_features)
    
    # Apply scaler if provided
    scaler = None
    if scaler_path:
        try:
            scaler = load(scaler_path)
            # Reshape for scaler, scale, then reshape back
            original_shape = sequence.shape
            sequence = scaler.transform(sequence)
            LOGGER.info("Applied scaler to sequence data")
        except Exception as exc:
            LOGGER.warning(f"Failed to load/apply scaler: {exc}")
    
    # Add batch dimension
    sequence = np.expand_dims(sequence, axis=0)  # shape: (1, seq_len, n_features)
    
    return sequence, scaler


def merge_data_sources(
    open_meteo_df: Optional[pd.DataFrame] = None,
    ndbc_df: Optional[pd.DataFrame] = None,
    feature_columns: List[str] = None,
) -> pd.DataFrame:
    """Merge multiple real-time data sources into unified DataFrame.
    
    Args:
        open_meteo_df: Data from Open-Meteo Marine API
        ndbc_df: Data from NDBC buoy
        feature_columns: Target feature columns
    
    Returns:
        Merged DataFrame with time index and feature columns
    """
    if feature_columns is None:
        feature_columns = ["Hs", "Hmax", "SST", "WindSpeed", "PeakWaveDirection"]
    
    # Start with an empty DataFrame
    result = pd.DataFrame()
    
    # Merge sources, preferring more accurate/recent data
    # Priority: NDBC (direct observations) > Open-Meteo (forecast)
    
    if ndbc_df is not None and not ndbc_df.empty:
        result = ndbc_df.copy()
        LOGGER.info(f"Using NDBC data as primary source ({len(result)} timesteps)")
    
    if open_meteo_df is not None and not open_meteo_df.empty:
        if result.empty:
            result = open_meteo_df.copy()
            LOGGER.info(f"Using Open-Meteo data as primary source ({len(result)} timesteps)")
        else:
            # Fill missing values from Open-Meteo
            for col in feature_columns:
                if col in open_meteo_df.columns and col not in result.columns:
                    result[col] = open_meteo_df[col]
                elif col in open_meteo_df.columns and col in result.columns:
                    # Fill NaN values in result with Open-Meteo data
                    result[col] = result[col].fillna(open_meteo_df[col])
            LOGGER.info(f"Merged Open-Meteo data (now {len(result)} timesteps)")
    
    # Ensure all required features exist
    for col in feature_columns:
        if col not in result.columns:
            result[col] = np.nan
    
    # Keep only the required features
    result = result[feature_columns]
    
    return result


def prepare_mock_image_tensor(
    image_size: Tuple[int, int] = (128, 128),
) -> np.ndarray:
    """Create a mock satellite image tensor for testing.
    
    In production, this would load actual satellite imagery.
    
    Args:
        image_size: Image dimensions (height, width)
    
    Returns:
        Image array with shape (1, height, width, 3)
    """
    # Create a simple gradient pattern as placeholder
    height, width = image_size
    img = np.zeros((height, width, 3))
    
    # Simple blue gradient (representing ocean)
    for i in range(height):
        img[i, :, 2] = 0.3 + 0.4 * (i / height)  # Blue channel
        img[i, :, 0] = 0.1  # Red channel
        img[i, :, 1] = 0.2  # Green channel
    
    # Add batch dimension
    img = np.expand_dims(img, axis=0)  # shape: (1, height, width, 3)
    
    return img.astype(np.float32)


def build_model_input(
    sequence_df: pd.DataFrame,
    feature_columns: List[str],
    seq_len: int = 12,
    image_size: Tuple[int, int] = (128, 128),
    scaler_path: Optional[str] = None,
    image_array: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build complete model input from real-time data.
    
    Args:
        sequence_df: DataFrame with time-series features
        feature_columns: List of feature names in order
        seq_len: Sequence length for LSTM
        image_size: Image dimensions
        scaler_path: Path to fitted scaler
        image_array: Optional actual image array (otherwise uses mock)
    
    Returns:
        Tuple of (image_tensor, sequence_tensor) ready for model.predict()
    """
    # Prepare sequence
    seq_tensor, _ = prepare_sequence_tensor(
        sequence_df,
        feature_columns,
        seq_len,
        scaler_path,
    )
    
    # Prepare image
    if image_array is not None:
        # Ensure correct shape
        if image_array.ndim == 3:
            image_array = np.expand_dims(image_array, axis=0)
        img_tensor = image_array
    else:
        img_tensor = prepare_mock_image_tensor(image_size)
    
    return img_tensor, seq_tensor


def format_prediction_result(
    probabilities: np.ndarray,
    class_names: List[str] = None,
) -> Dict:
    """Format model prediction as structured result.
    
    Args:
        probabilities: Model output probabilities (softmax)
        class_names: List of class names
    
    Returns:
        Dictionary with prediction details
    """
    if class_names is None:
        class_names = ["NORMAL", "MODERATE", "GIANT"]
    
    predicted_class_idx = int(np.argmax(probabilities))
    predicted_class = class_names[predicted_class_idx]
    
    # Calculate Hazard Probability Index (weighted average)
    # Weights: NORMAL=0, MODERATE=0.5, GIANT=1.0
    weights = np.array([0.0, 0.5, 1.0])
    hpi = float(np.sum(probabilities * weights))
    
    result = {
        "predicted_class": predicted_class,
        "predicted_class_index": predicted_class_idx,
        "hazard_probability_index": round(hpi, 4),
        "probabilities": {
            class_names[i]: round(float(probabilities[i]), 4)
            for i in range(len(class_names))
        },
        "confidence": round(float(probabilities[predicted_class_idx]), 4),
    }
    
    return result
