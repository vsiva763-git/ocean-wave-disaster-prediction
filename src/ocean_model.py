"""Enhanced Multi-Modal CNN + LSTM Hybrid Model for Ocean Wave and Tsunami Prediction.

This model combines:
- CNN branch: Processes satellite imagery / visual patterns
- LSTM branch: Processes time-series ocean data
- Fusion layer: Combines both modalities for final prediction

Focused on accurate prediction for Kanyakumari region.
"""
from __future__ import annotations

from typing import List, Literal, Optional, Tuple
import numpy as np

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint


def build_cnn_backbone(
    image_shape: Tuple[int, int, int],
    backbone: Literal["simple", "enhanced", "mobilenet_v2"] = "enhanced",
) -> tf.keras.Model:
    """Build CNN backbone for image feature extraction.
    
    Args:
        image_shape: Input image dimensions (height, width, channels)
        backbone: Architecture type
            - "simple": Basic 3-layer CNN
            - "enhanced": Deeper CNN with residual connections
            - "mobilenet_v2": Transfer learning with MobileNetV2
    
    Returns:
        Keras model that extracts image features
    """
    inputs = layers.Input(shape=image_shape, name="image_input")
    
    if backbone == "simple":
        x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        
        x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        
        x = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.3)(x)
        
    elif backbone == "enhanced":
        # Initial convolution
        x = layers.Conv2D(32, (7, 7), strides=2, padding="same")(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.MaxPooling2D((3, 3), strides=2, padding="same")(x)
        
        # Residual block 1
        x = _residual_block(x, 64, downsample=False)
        x = _residual_block(x, 64, downsample=False)
        
        # Residual block 2
        x = _residual_block(x, 128, downsample=True)
        x = _residual_block(x, 128, downsample=False)
        
        # Residual block 3
        x = _residual_block(x, 256, downsample=True)
        x = _residual_block(x, 256, downsample=False)
        
        # Global pooling
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.4)(x)
        
    elif backbone == "mobilenet_v2":
        # Use pretrained MobileNetV2
        base = tf.keras.applications.MobileNetV2(
            input_shape=image_shape,
            include_top=False,
            weights="imagenet",
        )
        base.trainable = False  # Freeze base layers initially
        
        x = base(inputs)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.3)(x)
        
    else:
        raise ValueError(f"Unknown backbone: {backbone}")
    
    return models.Model(inputs, x, name=f"{backbone}_cnn_backbone")


def _residual_block(x, filters: int, downsample: bool = False):
    """Create a residual block with optional downsampling."""
    stride = 2 if downsample else 1
    shortcut = x
    
    # First convolution
    x = layers.Conv2D(filters, (3, 3), strides=stride, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    
    # Second convolution
    x = layers.Conv2D(filters, (3, 3), strides=1, padding="same")(x)
    x = layers.BatchNormalization()(x)
    
    # Shortcut connection
    if downsample or shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, (1, 1), strides=stride, padding="same")(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)
    
    x = layers.Add()([x, shortcut])
    x = layers.Activation("relu")(x)
    
    return x


def build_lstm_branch(
    seq_len: int,
    seq_features: int,
    lstm_units: List[int] = None,
    bidirectional: bool = True,
    attention: bool = True,
) -> tf.keras.Model:
    """Build LSTM branch for time-series processing.
    
    Args:
        seq_len: Number of time steps
        seq_features: Number of features per time step
        lstm_units: List of LSTM layer sizes
        bidirectional: Whether to use bidirectional LSTMs
        attention: Whether to add attention mechanism
    
    Returns:
        Keras model that processes sequence data
    """
    if lstm_units is None:
        lstm_units = [128, 64]
    
    inputs = layers.Input(shape=(seq_len, seq_features), name="sequence_input")
    
    # Masking for variable-length sequences
    x = layers.Masking(mask_value=0.0)(inputs)
    
    # LSTM layers
    for i, units in enumerate(lstm_units):
        return_sequences = (i < len(lstm_units) - 1) or attention
        
        lstm_layer = layers.LSTM(
            units,
            return_sequences=return_sequences,
            dropout=0.2,
            recurrent_dropout=0.1,
            kernel_regularizer=regularizers.l2(1e-4),
        )
        
        if bidirectional:
            x = layers.Bidirectional(lstm_layer, name=f"bilstm_{i}")(x)
        else:
            x = lstm_layer(x)
        
        x = layers.BatchNormalization()(x)
    
    # Attention mechanism
    if attention:
        x = _attention_layer(x)
    
    x = layers.Dropout(0.3)(x)
    
    return models.Model(inputs, x, name="lstm_branch")


def _attention_layer(x):
    """Self-attention layer for sequence data."""
    # Compute attention scores
    attention_scores = layers.Dense(1, activation="tanh")(x)
    attention_weights = layers.Softmax(axis=1)(attention_scores)
    
    # Weighted sum
    attended = layers.Multiply()([x, attention_weights])
    output = layers.Lambda(lambda t: tf.reduce_sum(t, axis=1))(attended)
    
    return output


class MultiModalOceanModel:
    """Multi-modal CNN + LSTM model for ocean wave and tsunami prediction."""
    
    def __init__(
        self,
        image_shape: Tuple[int, int, int] = (128, 128, 3),
        seq_len: int = 24,
        seq_features: int = 8,
        num_classes: int = 4,
        class_names: List[str] = None,
        cnn_backbone: str = "enhanced",
        lstm_units: List[int] = None,
        use_attention: bool = True,
    ):
        """Initialize the multi-modal model.
        
        Args:
            image_shape: Shape of input images (H, W, C)
            seq_len: Number of time steps in sequence
            seq_features: Number of features per time step
            num_classes: Number of output classes
            class_names: Names of output classes
            cnn_backbone: CNN architecture type
            lstm_units: LSTM layer sizes
            use_attention: Whether to use attention in LSTM
        """
        self.image_shape = image_shape
        self.seq_len = seq_len
        self.seq_features = seq_features
        self.num_classes = num_classes
        self.class_names = class_names or ["NORMAL", "MODERATE", "HIGH_RISK", "TSUNAMI_WARNING"]
        self.cnn_backbone = cnn_backbone
        self.lstm_units = lstm_units or [128, 64]
        self.use_attention = use_attention
        
        self.model = None
        self._build_model()
    
    def _build_model(self):
        """Construct the full multi-modal model."""
        # CNN branch for image processing
        cnn_model = build_cnn_backbone(self.image_shape, self.cnn_backbone)
        
        # LSTM branch for sequence processing
        lstm_model = build_lstm_branch(
            self.seq_len,
            self.seq_features,
            self.lstm_units,
            bidirectional=True,
            attention=self.use_attention,
        )
        
        # Get features from both branches
        image_features = cnn_model.output
        sequence_features = lstm_model.output
        
        # Fusion layer
        fused = layers.Concatenate(name="multimodal_fusion")([
            image_features,
            sequence_features,
        ])
        
        # Fusion processing
        x = layers.Dense(256, activation="relu", kernel_regularizer=regularizers.l2(1e-4))(fused)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.4)(x)
        
        x = layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(1e-4))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        
        x = layers.Dense(64, activation="relu")(x)
        
        # Output layer
        outputs = layers.Dense(
            self.num_classes,
            activation="softmax",
            name="risk_classification"
        )(x)
        
        # Create combined model
        self.model = models.Model(
            inputs=[cnn_model.input, lstm_model.input],
            outputs=outputs,
            name="multimodal_ocean_predictor",
        )
    
    def compile(
        self,
        learning_rate: float = 1e-3,
        optimizer: str = "adam",
    ):
        """Compile the model for training."""
        if optimizer == "adam":
            opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer == "adamw":
            opt = tf.keras.optimizers.AdamW(learning_rate=learning_rate)
        else:
            opt = optimizer
        
        self.model.compile(
            optimizer=opt,
            loss="categorical_crossentropy",
            metrics=[
                "accuracy",
                tf.keras.metrics.Precision(name="precision"),
                tf.keras.metrics.Recall(name="recall"),
                tf.keras.metrics.AUC(name="auc"),
            ],
        )
    
    def train(
        self,
        x_train: Tuple[np.ndarray, np.ndarray],
        y_train: np.ndarray,
        x_val: Tuple[np.ndarray, np.ndarray] = None,
        y_val: np.ndarray = None,
        epochs: int = 100,
        batch_size: int = 32,
        checkpoint_path: str = None,
    ) -> tf.keras.callbacks.History:
        """Train the model.
        
        Args:
            x_train: Tuple of (images, sequences) for training
            y_train: Training labels (one-hot encoded)
            x_val: Validation data
            y_val: Validation labels
            epochs: Maximum training epochs
            batch_size: Batch size
            checkpoint_path: Path to save best model
        
        Returns:
            Training history
        """
        callbacks = [
            EarlyStopping(
                monitor="val_loss" if x_val is not None else "loss",
                patience=15,
                restore_best_weights=True,
            ),
            ReduceLROnPlateau(
                monitor="val_loss" if x_val is not None else "loss",
                factor=0.5,
                patience=5,
                min_lr=1e-6,
            ),
        ]
        
        if checkpoint_path:
            callbacks.append(ModelCheckpoint(
                checkpoint_path,
                monitor="val_loss" if x_val is not None else "loss",
                save_best_only=True,
            ))
        
        validation_data = None
        if x_val is not None and y_val is not None:
            validation_data = (x_val, y_val)
        
        history = self.model.fit(
            x_train,
            y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1,
        )
        
        return history
    
    def predict(
        self,
        images: np.ndarray,
        sequences: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Make predictions.
        
        Args:
            images: Image tensor (batch, H, W, C)
            sequences: Sequence tensor (batch, seq_len, features)
        
        Returns:
            Tuple of:
                - probabilities: Class probabilities
                - predicted_classes: Predicted class indices
                - hazard_index: Hazard probability index (0-1)
        """
        probabilities = self.model.predict([images, sequences], verbose=0)
        predicted_classes = np.argmax(probabilities, axis=-1)
        hazard_index = self._compute_hazard_index(probabilities)
        
        return probabilities, predicted_classes, hazard_index
    
    def _compute_hazard_index(self, probabilities: np.ndarray) -> np.ndarray:
        """Compute Hazard Probability Index (HPI).
        
        HPI is a weighted combination of class probabilities:
        - NORMAL: 0.0
        - MODERATE: 0.33
        - HIGH_RISK: 0.67
        - TSUNAMI_WARNING: 1.0
        """
        weights = np.array([0.0, 0.33, 0.67, 1.0])
        return np.sum(probabilities * weights, axis=-1)
    
    def predict_single(
        self,
        image: np.ndarray = None,
        sequence: np.ndarray = None,
    ) -> dict:
        """Make prediction for a single sample.
        
        Args:
            image: Single image (H, W, C) or None for mock
            sequence: Single sequence (seq_len, features) or None for mock
        
        Returns:
            Dictionary with prediction results
        """
        # Handle missing inputs with mock data
        if image is None:
            image = np.random.rand(*self.image_shape).astype(np.float32)
        if sequence is None:
            sequence = np.random.rand(self.seq_len, self.seq_features).astype(np.float32)
        
        # Add batch dimension
        images = np.expand_dims(image, axis=0)
        sequences = np.expand_dims(sequence, axis=0)
        
        # Predict
        probs, pred_class, hpi = self.predict(images, sequences)
        
        return {
            "predicted_class": self.class_names[pred_class[0]],
            "predicted_class_index": int(pred_class[0]),
            "probabilities": {
                name: float(probs[0, i])
                for i, name in enumerate(self.class_names)
            },
            "hazard_probability_index": float(hpi[0]),
            "confidence": float(np.max(probs[0])),
        }
    
    def save(self, path: str):
        """Save model to file."""
        self.model.save(path)
    
    @classmethod
    def load(cls, path: str) -> "MultiModalOceanModel":
        """Load model from file."""
        instance = cls.__new__(cls)
        instance.model = tf.keras.models.load_model(path)
        
        # Extract configuration from model
        instance.image_shape = instance.model.input[0].shape[1:]
        instance.seq_len = instance.model.input[1].shape[1]
        instance.seq_features = instance.model.input[1].shape[2]
        instance.num_classes = instance.model.output.shape[1]
        instance.class_names = ["NORMAL", "MODERATE", "HIGH_RISK", "TSUNAMI_WARNING"]
        
        return instance
    
    def summary(self):
        """Print model summary."""
        return self.model.summary()


def create_mock_prediction(
    wave_height: float,
    wind_speed: float,
    earthquake_magnitude: float = 0,
    earthquake_distance: float = 10000,
) -> dict:
    """Create a rule-based mock prediction when model is not available.
    
    This uses simple rules based on ocean parameters to provide
    reasonable predictions for demonstration purposes.
    
    Args:
        wave_height: Current wave height in meters
        wind_speed: Wind speed in km/h
        earthquake_magnitude: Recent earthquake magnitude (0 if none)
        earthquake_distance: Distance to earthquake epicenter in km
    
    Returns:
        Prediction dictionary
    """
    class_names = ["NORMAL", "MODERATE", "HIGH_RISK", "TSUNAMI_WARNING"]
    
    # Initialize base probabilities
    probs = [0.7, 0.2, 0.08, 0.02]  # Default: mostly normal
    
    # Adjust based on wave height
    if wave_height >= 6.0:
        probs = [0.05, 0.15, 0.50, 0.30]
    elif wave_height >= 4.0:
        probs = [0.10, 0.30, 0.50, 0.10]
    elif wave_height >= 2.5:
        probs = [0.25, 0.50, 0.20, 0.05]
    elif wave_height >= 1.5:
        probs = [0.50, 0.35, 0.12, 0.03]
    
    # Adjust based on wind speed
    if wind_speed >= 100:
        probs[2] += 0.15
        probs[0] -= 0.15
    elif wind_speed >= 60:
        probs[1] += 0.10
        probs[0] -= 0.10
    
    # Major factor: earthquake-induced tsunami risk
    if earthquake_magnitude >= 8.0 and earthquake_distance < 2000:
        probs = [0.02, 0.08, 0.30, 0.60]
    elif earthquake_magnitude >= 7.0 and earthquake_distance < 1500:
        probs = [0.05, 0.20, 0.45, 0.30]
    elif earthquake_magnitude >= 6.5 and earthquake_distance < 1000:
        probs = [0.15, 0.35, 0.35, 0.15]
    
    # Normalize probabilities
    total = sum(probs)
    probs = [p / total for p in probs]
    
    # Determine prediction
    predicted_idx = probs.index(max(probs))
    
    # Compute HPI
    weights = [0.0, 0.33, 0.67, 1.0]
    hpi = sum(p * w for p, w in zip(probs, weights))
    
    return {
        "predicted_class": class_names[predicted_idx],
        "predicted_class_index": predicted_idx,
        "probabilities": {
            name: round(prob, 4)
            for name, prob in zip(class_names, probs)
        },
        "hazard_probability_index": round(hpi, 4),
        "confidence": round(max(probs), 4),
        "model_type": "rule_based",
    }
