"""Evaluation utilities for the trained model."""
import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from joblib import load
from sklearn.metrics import classification_report, confusion_matrix

from data_loader import build_tf_dataset, load_dataset_csv, load_image_batch, normalize_sequences
from model_cnn_lstm import hazard_probability_index
from preprocess import assemble_sequences, build_sequence_feature_columns, encode_labels


def plot_confusion(cm: np.ndarray, class_names: list, save_path: str) -> None:
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def evaluate(args: argparse.Namespace) -> None:
    df = load_dataset_csv(args.csv_path)
    seq_cols = args.seq_columns
    if seq_cols is None:
        seq_cols = build_sequence_feature_columns(seq_len=args.seq_len, n_features=args.seq_features)
    else:
        seq_cols = seq_cols.split(",")

    # Label encoding (fit on full set to align with training mapping)
    encoded_labels, encoder = encode_labels(df[args.label_col].astype(str).tolist())
    labels = encoder.transform(df[args.label_col].astype(str))
    class_names = list(encoder.classes_)

    sequences = assemble_sequences(df, seq_cols)
    scaler = load(args.scaler_path)
    sequences, _ = normalize_sequences(sequences, scaler)

    images = load_image_batch(args.image_dir, df[args.image_col].tolist(), (args.image_size, args.image_size))

    ds = build_tf_dataset(images, sequences, labels, args.batch_size, shuffle=False)

    model = tf.keras.models.load_model(args.model_path)
    preds = model.predict(ds)
    pred_labels = np.argmax(preds, axis=1)

    report = classification_report(labels, pred_labels, target_names=class_names, output_dict=True)
    cm = confusion_matrix(labels, pred_labels)

    hpi = hazard_probability_index(tf.convert_to_tensor(preds)).numpy()

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "classification_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    np.save(os.path.join(args.output_dir, "confusion_matrix.npy"), cm)
    plot_confusion(cm, class_names, os.path.join(args.output_dir, "confusion_matrix.png"))

    # Save predictions CSV
    prob_cols = [f"prob_{cls}" for cls in class_names]
    out_rows = []
    for fname, true_lbl, pred_lbl, probs, h in zip(
        df[args.image_col].tolist(), labels, pred_labels, preds, hpi
    ):
        row = {
            "image": fname,
            "true_label": class_names[true_lbl],
            "pred_label": class_names[pred_lbl],
            "hazard_index": float(h),
        }
        row.update({pc: float(p) for pc, p in zip(prob_cols, probs)})
        out_rows.append(row)

    import pandas as pd

    pd.DataFrame(out_rows).to_csv(os.path.join(args.output_dir, "predictions.csv"), index=False)

    print("Evaluation complete. Macro F1:", report["macro avg"]["f1-score"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained multimodal model")
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--scaler_path", type=str, required=True, help="Path to sequence scaler pickle")
    parser.add_argument("--output_dir", type=str, default="reports")
    parser.add_argument("--image_col", type=str, default="image")
    parser.add_argument("--label_col", type=str, default="label")
    parser.add_argument("--seq_columns", type=str, default=None)
    parser.add_argument("--seq_len", type=int, default=12)
    parser.add_argument("--seq_features", type=int, default=5)
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=32)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(args)
