"""Training script for CNN + LSTM fusion model."""
import argparse
import json
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from joblib import dump

from data_loader import (
    build_tf_dataset,
    load_dataset_csv,
    load_image_batch,
    normalize_sequences,
)
from model_cnn_lstm import build_multimodal_model
from preprocess import assemble_sequences, build_sequence_feature_columns, encode_labels, stratified_split


def plot_history(history: tf.keras.callbacks.History, output_dir: str) -> None:
    """Save training curves to disk."""
    metrics = history.history
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(metrics["loss"], label="train")
    if "val_loss" in metrics:
        plt.plot(metrics["val_loss"], label="val")
    plt.title("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(metrics.get("accuracy", []), label="train")
    if "val_accuracy" in metrics:
        plt.plot(metrics["val_accuracy"], label="val")
    plt.title("Accuracy")
    plt.legend()

    plt.tight_layout()
    fig_path = os.path.join(output_dir, "training_curves.png")
    plt.savefig(fig_path)
    plt.close()


def train(args: argparse.Namespace) -> None:
    os.makedirs(args.output_dir, exist_ok=True)

    df = load_dataset_csv(args.csv_path)
    seq_cols = args.seq_columns
    if seq_cols is None:
        seq_cols = build_sequence_feature_columns(seq_len=args.seq_len, n_features=args.seq_features)
    else:
        seq_cols = seq_cols.split(",")

    train_df, val_df, test_df = stratified_split(
        df, test_size=args.test_size, val_size=args.val_size, label_col=args.label_col, random_state=args.seed
    )

    # Label encoding
    _, encoder = encode_labels(df[args.label_col].astype(str).tolist())
    class_names = list(encoder.classes_)
    train_labels = encoder.transform(train_df[args.label_col].astype(str))
    val_labels = encoder.transform(val_df[args.label_col].astype(str))
    test_labels = encoder.transform(test_df[args.label_col].astype(str))

    # Sequences
    train_seq = assemble_sequences(train_df, seq_cols)
    val_seq = assemble_sequences(val_df, seq_cols)
    test_seq = assemble_sequences(test_df, seq_cols)

    train_seq, scaler = normalize_sequences(train_seq)
    val_seq, _ = normalize_sequences(val_seq, scaler)
    test_seq, _ = normalize_sequences(test_seq, scaler)

    # Images
    image_size = (args.image_size, args.image_size)
    train_images = load_image_batch(args.image_dir, train_df[args.image_col].tolist(), image_size)
    val_images = load_image_batch(args.image_dir, val_df[args.image_col].tolist(), image_size)
    test_images = load_image_batch(args.image_dir, test_df[args.image_col].tolist(), image_size)

    # Datasets
    train_ds = build_tf_dataset(train_images, train_seq, train_labels, args.batch_size, shuffle=True)
    val_ds = build_tf_dataset(val_images, val_seq, val_labels, args.batch_size, shuffle=False)
    test_ds = build_tf_dataset(test_images, test_seq, test_labels, args.batch_size, shuffle=False)

    model = build_multimodal_model(
        image_shape=(args.image_size, args.image_size, 3),
        seq_len=args.seq_len,
        seq_features=args.seq_features,
        num_classes=len(class_names),
        backbone="mobilenet_v2" if args.use_mobilenet else "simple",
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=args.early_stopping, restore_best_weights=True, monitor="val_loss"),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(args.output_dir, "best_model.h5"),
            monitor="val_loss",
            save_best_only=True,
        ),
        tf.keras.callbacks.TensorBoard(log_dir=os.path.join(args.output_dir, "logs")),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    # Save artifacts
    model.save(os.path.join(args.output_dir, "last_model.h5"))
    plot_history(history, args.output_dir)

    with open(os.path.join(args.output_dir, "history.json"), "w", encoding="utf-8") as f:
        json.dump(history.history, f, indent=2)

    with open(os.path.join(args.output_dir, "label_mapping.json"), "w", encoding="utf-8") as f:
        json.dump({"classes": class_names}, f, indent=2)

    dump(scaler, os.path.join(args.output_dir, "sequence_scaler.pkl"))

    # Evaluate on test split
    test_metrics = model.evaluate(test_ds, return_dict=True)
    with open(os.path.join(args.output_dir, "test_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2)

    print("Training complete. Test metrics:", test_metrics)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CNN+LSTM fusion model for ocean wave risk.")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to dataset CSV")
    parser.add_argument("--image_dir", type=str, required=True, help="Directory containing image files")
    parser.add_argument("--output_dir", type=str, default="models", help="Directory to save models and logs")
    parser.add_argument("--image_col", type=str, default="image", help="Column name for image filenames")
    parser.add_argument("--label_col", type=str, default="label", help="Column name for labels")
    parser.add_argument(
        "--seq_columns",
        type=str,
        default=None,
        help="Comma-separated list of sequence columns if custom. Defaults to generated tX_Feature columns.",
    )
    parser.add_argument("--seq_len", type=int, default=12, help="Sequence length (timesteps)")
    parser.add_argument("--seq_features", type=int, default=5, help="Number of features per timestep")
    parser.add_argument("--image_size", type=int, default=128, help="Image side length (square)")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--early_stopping", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--test_size", type=float, default=0.15, help="Test split size")
    parser.add_argument("--val_size", type=float, default=0.15, help="Validation split size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--use_mobilenet", action="store_true", help="Use MobileNetV2 backbone")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    # For reproducibility
    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)
    train(args)
