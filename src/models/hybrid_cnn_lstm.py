"""Enhanced Multimodal CNN-LSTM Hybrid Model for Ocean Wave and Tsunami Prediction.

This module implements a sophisticated deep learning architecture that combines:
1. Convolutional Neural Networks (CNN) for spatial pattern recognition
2. Long Short-Term Memory (LSTM) networks for temporal sequence modeling
3. Attention mechanisms for feature importance weighting
4. Multi-task learning for wave height and tsunami risk prediction

Optimized for Kanyakumari coastal region monitoring.
"""
from typing import Dict, List, Literal, Optional, Tuple, Union
import numpy as np

# Try to import TensorFlow
TF_AVAILABLE = False
tf = None
layers = None
models = None
Model = None
Adam = None
EarlyStopping = None
ReduceLROnPlateau = None
ModelCheckpoint = None

try:
    import tensorflow as tf
    from tensorflow.keras import layers, models
    from tensorflow.keras.models import Model
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
    TF_AVAILABLE = True
except ImportError:
    pass


# Define a placeholder AttentionLayer class when TensorFlow is not available
if TF_AVAILABLE:
    class AttentionLayer(layers.Layer):
        """Custom attention layer for sequence-to-sequence attention."""

        def __init__(self, units: int = 64, **kwargs):
            super().__init__(**kwargs)
            self.units = units

        def build(self, input_shape):
            self.W = self.add_weight(
                name='attention_weight',
                shape=(input_shape[-1], self.units),
                initializer='glorot_uniform',
                trainable=True
            )
            self.b = self.add_weight(
                name='attention_bias',
                shape=(self.units,),
                initializer='zeros',
                trainable=True
            )
            self.u = self.add_weight(
                name='attention_context',
                shape=(self.units, 1),
                initializer='glorot_uniform',
                trainable=True
            )
            super().build(input_shape)

        def call(self, inputs):
            # Compute attention scores
            score = tf.tanh(tf.tensordot(inputs, self.W, axes=1) + self.b)
            attention_weights = tf.nn.softmax(
                tf.tensordot(score, self.u, axes=1), axis=1)
            # Weighted sum
            context = tf.reduce_sum(inputs * attention_weights, axis=1)
            return context, attention_weights

        def get_config(self):
            config = super().get_config()
            config.update({'units': self.units})
            return config
else:
    # Placeholder class when TensorFlow is not available
    class AttentionLayer:
        """Placeholder AttentionLayer when TensorFlow is not available."""

        def __init__(self, *args, **kwargs):
            raise ImportError("TensorFlow is required for AttentionLayer")


def build_cnn_backbone(
    image_shape: Tuple[int, int, int] = (128, 128, 3),
    backbone_type: Literal["simple", "mobilenet", "resnet"] = "simple"
) -> "Model":
    """Build CNN backbone for spatial feature extraction.

    Args:
        image_shape: Input image dimensions (height, width, channels)
        backbone_type: Type of CNN backbone to use

    Returns:
        Keras Model for feature extraction
    """
    if not TF_AVAILABLE:
        raise ImportError("TensorFlow is required for model building")

    inputs = layers.Input(shape=image_shape, name="image_input")

    if backbone_type == "simple":
        # Custom lightweight CNN
        x = layers.Conv2D(32, (3, 3), activation='relu',
                          padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.25)(x)

        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.25)(x)

        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.25)(x)

        x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.3)(x)

    elif backbone_type == "mobilenet":
        # MobileNetV2 transfer learning
        base = tf.keras.applications.MobileNetV2(
            input_shape=image_shape,
            include_top=False,
            weights='imagenet'
        )
        base.trainable = False  # Freeze for initial training
        x = base(inputs)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.3)(x)

    elif backbone_type == "resnet":
        # ResNet50 transfer learning
        base = tf.keras.applications.ResNet50(
            input_shape=image_shape,
            include_top=False,
            weights='imagenet'
        )
        base.trainable = False
        x = base(inputs)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.3)(x)

    else:
        raise ValueError(f"Unknown backbone type: {backbone_type}")

    return Model(inputs, x, name=f"{backbone_type}_cnn_backbone")


def build_lstm_backbone(
    seq_len: int = 24,
    seq_features: int = 8,
    lstm_units: List[int] = [128, 64],
    use_attention: bool = True,
    bidirectional: bool = True
) -> "Model":
    """Build LSTM backbone for temporal sequence modeling.

    Args:
        seq_len: Length of input sequences (time steps)
        seq_features: Number of features per time step
        lstm_units: List of LSTM units per layer
        use_attention: Whether to use attention mechanism
        bidirectional: Whether to use bidirectional LSTM

    Returns:
        Keras Model for sequence feature extraction
    """
    if not TF_AVAILABLE:
        raise ImportError("TensorFlow is required for model building")

    inputs = layers.Input(shape=(seq_len, seq_features), name="sequence_input")

    # Masking for variable length sequences
    x = layers.Masking(mask_value=0.0)(inputs)

    # Stacked LSTM layers
    for i, units in enumerate(lstm_units[:-1]):
        if bidirectional:
            x = layers.Bidirectional(
                layers.LSTM(units, return_sequences=True,
                            dropout=0.2, recurrent_dropout=0.1),
                name=f"bilstm_{i}"
            )(x)
        else:
            x = layers.LSTM(units, return_sequences=True, dropout=0.2,
                            recurrent_dropout=0.1, name=f"lstm_{i}")(x)
        x = layers.BatchNormalization()(x)

    # Final LSTM layer
    if use_attention:
        if bidirectional:
            x = layers.Bidirectional(
                layers.LSTM(lstm_units[-1],
                            return_sequences=True, dropout=0.2),
                name="bilstm_attention"
            )(x)
        else:
            x = layers.LSTM(
                lstm_units[-1], return_sequences=True, dropout=0.2, name="lstm_attention")(x)

        # Apply attention
        x, attention_weights = AttentionLayer(units=64, name="attention")(x)
    else:
        if bidirectional:
            x = layers.Bidirectional(
                layers.LSTM(lstm_units[-1],
                            return_sequences=False, dropout=0.2),
                name="bilstm_final"
            )(x)
        else:
            x = layers.LSTM(
                lstm_units[-1], return_sequences=False, dropout=0.2, name="lstm_final")(x)

    x = layers.Dropout(0.3)(x)

    return Model(inputs, x, name="lstm_backbone")


def build_multimodal_hybrid_model(
    image_shape: Tuple[int, int, int] = (128, 128, 3),
    seq_len: int = 24,
    seq_features: int = 8,
    cnn_backbone: Literal["simple", "mobilenet", "resnet"] = "simple",
    lstm_units: List[int] = [128, 64],
    use_attention: bool = True,
    bidirectional: bool = True,
    num_wave_classes: int = 4,
    num_tsunami_classes: int = 3,
    fusion_dense_units: List[int] = [256, 128, 64],
    include_regression_head: bool = True
) -> "Model":
    """Build the complete multimodal CNN-LSTM hybrid model.

    This model combines:
    - CNN backbone for spatial patterns (satellite imagery, heatmaps)
    - LSTM backbone for temporal sequences (wave height, pressure, etc.)
    - Multi-task heads for classification and regression

    Args:
        image_shape: Input image dimensions
        seq_len: Sequence length (time steps)
        seq_features: Features per time step
        cnn_backbone: Type of CNN backbone
        lstm_units: LSTM layer units
        use_attention: Use attention in LSTM
        bidirectional: Use bidirectional LSTM
        num_wave_classes: Wave severity classes
        num_tsunami_classes: Tsunami risk classes
        fusion_dense_units: Dense layer units after fusion
        include_regression_head: Include wave height regression

    Returns:
        Complete multimodal Keras Model
    """
    if not TF_AVAILABLE:
        raise ImportError("TensorFlow is required for model building")

    # Build backbones
    cnn = build_cnn_backbone(image_shape, cnn_backbone)
    lstm = build_lstm_backbone(
        seq_len, seq_features, lstm_units, use_attention, bidirectional)

    # Get inputs and features
    image_input = cnn.input
    image_features = cnn.output

    seq_input = lstm.input
    seq_features_out = lstm.output

    # Fusion layer - concatenate CNN and LSTM features
    fused = layers.Concatenate(name="multimodal_fusion")(
        [image_features, seq_features_out])

    # Shared dense layers
    x = fused
    for i, units in enumerate(fusion_dense_units):
        x = layers.Dense(units, activation='relu', name=f"fusion_dense_{i}")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.4)(x)

    # Output heads
    outputs = []

    # Wave severity classification (NORMAL, MODERATE, HIGH, EXTREME)
    wave_class_output = layers.Dense(
        num_wave_classes,
        activation='softmax',
        name='wave_severity'
    )(x)
    outputs.append(wave_class_output)

    # Tsunami risk classification (NONE, LOW, HIGH)
    tsunami_class_output = layers.Dense(
        num_tsunami_classes,
        activation='softmax',
        name='tsunami_risk'
    )(x)
    outputs.append(tsunami_class_output)

    # Wave height regression (optional)
    if include_regression_head:
        wave_height_output = layers.Dense(
            1,
            activation='linear',
            name='wave_height_meters'
        )(x)
        outputs.append(wave_height_output)

    # Build final model
    model = Model(
        inputs=[image_input, seq_input],
        outputs=outputs,
        name="multimodal_cnn_lstm_hybrid"
    )

    return model


def compile_model(
    model: "Model",
    learning_rate: float = 0.001,
    include_regression: bool = True
) -> "Model":
    """Compile the model with appropriate losses and metrics.

    Args:
        model: The Keras model to compile
        learning_rate: Initial learning rate
        include_regression: Whether model includes regression head

    Returns:
        Compiled model
    """
    if not TF_AVAILABLE:
        raise ImportError("TensorFlow is required")

    losses = {
        'wave_severity': 'categorical_crossentropy',
        'tsunami_risk': 'categorical_crossentropy',
    }

    loss_weights = {
        'wave_severity': 1.0,
        'tsunami_risk': 1.5,  # Higher weight for tsunami prediction
    }

    metrics = {
        'wave_severity': ['accuracy'],
        'tsunami_risk': ['accuracy'],
    }

    if include_regression:
        losses['wave_height_meters'] = 'mse'
        loss_weights['wave_height_meters'] = 0.5
        metrics['wave_height_meters'] = ['mae']

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss=losses,
        loss_weights=loss_weights,
        metrics=metrics
    )

    return model


def get_training_callbacks(
    checkpoint_path: str = 'models/best_model.keras',
    patience: int = 10
) -> List:
    """Get standard training callbacks.

    Args:
        checkpoint_path: Path to save best model
        patience: Early stopping patience

    Returns:
        List of Keras callbacks
    """
    if not TF_AVAILABLE:
        return []

    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=patience // 2,
            min_lr=1e-6,
            verbose=1
        ),
        ModelCheckpoint(
            checkpoint_path,
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
    ]

    return callbacks


def compute_hazard_probability_index(
    wave_probs: np.ndarray,
    tsunami_probs: np.ndarray,
    wave_weights: List[float] = [0.0, 0.3, 0.7, 1.0],
    tsunami_weights: List[float] = [0.0, 0.5, 1.0]
) -> np.ndarray:
    """Compute combined Hazard Probability Index (HPI).

    The HPI is a weighted combination of wave severity and tsunami risk
    probabilities, normalized to [0, 1].

    Args:
        wave_probs: Wave severity class probabilities [batch, num_classes]
        tsunami_probs: Tsunami risk class probabilities [batch, num_classes]
        wave_weights: Weights for each wave class (NORMAL=0, EXTREME=1)
        tsunami_weights: Weights for each tsunami class (NONE=0, HIGH=1)

    Returns:
        Combined HPI scores [batch,]
    """
    wave_weights = np.array(wave_weights)
    tsunami_weights = np.array(tsunami_weights)

    wave_hpi = np.sum(wave_probs * wave_weights, axis=-1)
    tsunami_hpi = np.sum(tsunami_probs * tsunami_weights, axis=-1)

    # Combined HPI (tsunami weighted higher due to severity)
    combined_hpi = 0.4 * wave_hpi + 0.6 * tsunami_hpi

    return combined_hpi


class OceanWavePredictor:
    """High-level predictor class for ocean wave and tsunami prediction.

    This class wraps the multimodal model and provides easy-to-use
    prediction methods.
    """

    # Class labels
    WAVE_CLASSES = ['NORMAL', 'MODERATE', 'HIGH', 'EXTREME']
    TSUNAMI_CLASSES = ['NONE', 'LOW', 'HIGH']

    def __init__(
        self,
        model_path: Optional[str] = None,
        model: Optional["Model"] = None,
        image_shape: Tuple[int, int, int] = (128, 128, 3),
        seq_len: int = 24,
        seq_features: int = 8
    ):
        """Initialize the predictor.

        Args:
            model_path: Path to saved model weights
            model: Pre-built Keras model
            image_shape: Expected image input shape
            seq_len: Expected sequence length
            seq_features: Expected features per time step
        """
        self.image_shape = image_shape
        self.seq_len = seq_len
        self.seq_features = seq_features
        self.model = model

        if model_path and TF_AVAILABLE:
            self.load_model(model_path)

    def load_model(self, model_path: str):
        """Load model from file."""
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow required to load model")

        self.model = tf.keras.models.load_model(
            model_path,
            custom_objects={'AttentionLayer': AttentionLayer}
        )

    def predict(
        self,
        image_input: np.ndarray,
        sequence_input: np.ndarray,
        return_details: bool = True
    ) -> Dict:
        """Make predictions on input data.

        Args:
            image_input: Image tensor [batch, H, W, C]
            sequence_input: Sequence tensor [batch, seq_len, features]
            return_details: Whether to return detailed probabilities

        Returns:
            Dictionary with predictions
        """
        if self.model is None:
            # Return mock predictions if no model
            return self._mock_prediction(image_input.shape[0])

        # Ensure proper shape
        if len(image_input.shape) == 3:
            image_input = np.expand_dims(image_input, 0)
        if len(sequence_input.shape) == 2:
            sequence_input = np.expand_dims(sequence_input, 0)

        # Get predictions
        predictions = self.model.predict(
            [image_input, sequence_input], verbose=0)

        # Parse outputs (depends on model configuration)
        if len(predictions) == 3:
            wave_probs, tsunami_probs, wave_height = predictions
        else:
            wave_probs, tsunami_probs = predictions[:2]
            wave_height = None

        # Compute derived metrics
        wave_class_idx = np.argmax(wave_probs, axis=-1)
        tsunami_class_idx = np.argmax(tsunami_probs, axis=-1)
        hpi = compute_hazard_probability_index(wave_probs, tsunami_probs)

        result = {
            'wave_class': [self.WAVE_CLASSES[i] for i in wave_class_idx],
            'tsunami_class': [self.TSUNAMI_CLASSES[i] for i in tsunami_class_idx],
            'hazard_probability_index': hpi.tolist(),
        }

        if wave_height is not None:
            result['predicted_wave_height_m'] = wave_height.flatten().tolist()

        if return_details:
            result['wave_probabilities'] = {
                cls: wave_probs[:, i].tolist()
                for i, cls in enumerate(self.WAVE_CLASSES)
            }
            result['tsunami_probabilities'] = {
                cls: tsunami_probs[:, i].tolist()
                for i, cls in enumerate(self.TSUNAMI_CLASSES)
            }

        # Single sample case - simplify output
        if wave_probs.shape[0] == 1:
            result = {
                'wave_class': result['wave_class'][0],
                'tsunami_class': result['tsunami_class'][0],
                'hazard_probability_index': result['hazard_probability_index'][0],
                'wave_probabilities': {k: v[0] for k, v in result.get('wave_probabilities', {}).items()},
                'tsunami_probabilities': {k: v[0] for k, v in result.get('tsunami_probabilities', {}).items()},
            }
            if 'predicted_wave_height_m' in result:
                result['predicted_wave_height_m'] = result['predicted_wave_height_m'][0]

        return result

    def _mock_prediction(self, batch_size: int = 1) -> Dict:
        """Generate mock predictions for testing."""
        # Simulate normal conditions with slight variation
        wave_probs = np.random.dirichlet([10, 3, 1, 0.5], size=batch_size)
        tsunami_probs = np.random.dirichlet([15, 2, 0.5], size=batch_size)
        wave_height = np.random.uniform(0.5, 2.5, size=batch_size)

        wave_class_idx = np.argmax(wave_probs, axis=-1)
        tsunami_class_idx = np.argmax(tsunami_probs, axis=-1)
        hpi = compute_hazard_probability_index(wave_probs, tsunami_probs)

        result = {
            'wave_class': [self.WAVE_CLASSES[i] for i in wave_class_idx],
            'tsunami_class': [self.TSUNAMI_CLASSES[i] for i in tsunami_class_idx],
            'hazard_probability_index': hpi.tolist(),
            'predicted_wave_height_m': wave_height.tolist(),
            'wave_probabilities': {
                cls: wave_probs[:, i].tolist()
                for i, cls in enumerate(self.WAVE_CLASSES)
            },
            'tsunami_probabilities': {
                cls: tsunami_probs[:, i].tolist()
                for i, cls in enumerate(self.TSUNAMI_CLASSES)
            },
            'is_mock': True
        }

        if batch_size == 1:
            result = {
                'wave_class': result['wave_class'][0],
                'tsunami_class': result['tsunami_class'][0],
                'hazard_probability_index': result['hazard_probability_index'][0],
                'predicted_wave_height_m': result['predicted_wave_height_m'][0],
                'wave_probabilities': {k: v[0] for k, v in result['wave_probabilities'].items()},
                'tsunami_probabilities': {k: v[0] for k, v in result['tsunami_probabilities'].items()},
                'is_mock': True
            }

        return result


# Model summary function
def print_model_summary():
    """Print summary of the default model architecture."""
    if not TF_AVAILABLE:
        print("TensorFlow not available. Install with: pip install tensorflow")
        return

    model = build_multimodal_hybrid_model()
    model = compile_model(model)
    print("\n" + "="*80)
    print("MULTIMODAL CNN-LSTM HYBRID MODEL FOR OCEAN WAVE & TSUNAMI PREDICTION")
    print("="*80)
    model.summary()

    print("\n" + "-"*80)
    print("Output Heads:")
    print("-"*80)
    print(
        f"  1. Wave Severity Classification: {OceanWavePredictor.WAVE_CLASSES}")
    print(
        f"  2. Tsunami Risk Classification: {OceanWavePredictor.TSUNAMI_CLASSES}")
    print(f"  3. Wave Height Regression: Continuous (meters)")
    print("="*80 + "\n")


if __name__ == "__main__":
    print_model_summary()
