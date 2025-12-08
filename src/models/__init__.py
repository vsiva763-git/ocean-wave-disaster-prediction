"""
Ocean Wave Prediction Models Package

Contains the multimodal CNN-LSTM hybrid architecture for
wave height prediction and tsunami risk assessment.
"""

from .hybrid_cnn_lstm import (
    build_multimodal_hybrid_model,
    build_cnn_backbone,
    build_lstm_backbone,
    compile_model,
    get_training_callbacks,
    compute_hazard_probability_index,
    OceanWavePredictor,
    AttentionLayer,
    TF_AVAILABLE,
)

__all__ = [
    'build_multimodal_hybrid_model',
    'build_cnn_backbone',
    'build_lstm_backbone',
    'compile_model',
    'get_training_callbacks',
    'compute_hazard_probability_index',
    'OceanWavePredictor',
    'AttentionLayer',
    'TF_AVAILABLE',
]
