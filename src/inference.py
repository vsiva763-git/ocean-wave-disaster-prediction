"""Inference script for single-sample or batch prediction.

Supports optional serial output to Arduino-compatible board.
"""
import argparse
import json
import os
from typing import List, Tuple

import numpy as np
import tensorflow as tf
from joblib import load

from data_loader import decode_predictions, load_image, normalize_sequences, reshape_sequences
from model_cnn_lstm import hazard_probability_index

try:
    import serial  # type: ignore
except ImportError:
    serial = None


LABELS = ["NORMAL", "MODERATE", "GIANT"]


def parse_sequence(seq_str: str, seq_len: int, n_features: int) -> np.ndarray:
    """Parse comma-separated numeric sequence into expected shape."""
    values = [float(x) for x in seq_str.split(",")]
    return reshape_sequences(np.array(values)[None, :], seq_len, n_features)


def run_inference(
    model_path: str,
    scaler_path: str,
    image_path: str,
    sequence_values: np.ndarray,
    image_size: Tuple[int, int] = (128, 128),
) -> Tuple[str, float, List[float]]:
    model = tf.keras.models.load_model(model_path)
    scaler = load(scaler_path)

    seq_norm, _ = normalize_sequences(sequence_values, scaler)
    img = load_image(image_path, image_size)

    probs = model.predict([np.expand_dims(img, 0), seq_norm], verbose=0)[0]
    hpi = float(hazard_probability_index(tf.convert_to_tensor([probs])).numpy()[0])
    label = decode_predictions(np.expand_dims(probs, 0))[0]
    return label, hpi, probs.tolist()


def send_serial(port: str, baud: int, message: str) -> None:
    if serial is None:
        raise ImportError("pyserial is required for serial output. Install with pip install pyserial.")
    with serial.Serial(port, baudrate=baud, timeout=2) as ser:
        ser.write((message + "\n").encode("utf-8"))


def main():
    parser = argparse.ArgumentParser(description="Run inference for a single multimodal sample")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--scaler_path", required=True)
    parser.add_argument("--image_path", required=True)
    parser.add_argument(
        "--sequence",
        required=True,
        help="Comma-separated flattened sequence of length seq_len * n_features",
    )
    parser.add_argument("--seq_len", type=int, default=12)
    parser.add_argument("--seq_features", type=int, default=5)
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--serial_port", type=str, default=None, help="Optional serial port for Arduino")
    parser.add_argument("--baud", type=int, default=115200)
    args = parser.parse_args()

    seq = parse_sequence(args.sequence, args.seq_len, args.seq_features)
    label, hpi, probs = run_inference(
        args.model_path, args.scaler_path, args.image_path, seq, (args.image_size, args.image_size)
    )
    result = {"label": label, "hazard_index": hpi, "probabilities": probs}
    print(json.dumps(result, indent=2))

    if args.serial_port:
        msg = f"LABEL:{label};HPI:{hpi:.3f};P_GIANT:{probs[2]:.3f}"
        send_serial(args.serial_port, args.baud, msg)
        print(f"Sent to serial {args.serial_port}: {msg}")


if __name__ == "__main__":
    main()
