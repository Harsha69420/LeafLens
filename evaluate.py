"""
evaluate.py - Model Evaluation, Metrics Computation, and Visualizations.
LeafLens: Plant Disease Screening Using CNN and Transfer Learning
"""

import os
import json
import argparse
os.environ.setdefault("KERAS_HOME", os.environ.get("TEMP", "/tmp"))
os.environ.setdefault("MPLCONFIGDIR", os.environ.get("TEMP", "/tmp"))
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support, accuracy_score
import tensorflow as tf

from model import load_model_artifact


def plot_and_save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list,
    save_path: str,
    normalize: bool = True,
):
    """
    Renders and saves a formatted confusion matrix heatmap using Seaborn and Matplotlib.
    """
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm_display = cm.astype("float") / (cm.sum(axis=1)[:, np.newaxis] + 1e-12)
        fmt = ".2%"
        title = "Normalized Confusion Matrix"
    else:
        cm_display = cm
        fmt = "d"
        title = "Confusion Matrix (Sample Counts)"

    # Clean display labels
    clean_labels = [name.replace("Tomato___", "").replace("_", " ") for name in class_names]

    plt.figure(figsize=(9, 7))
    sns.set_theme(style="whitegrid")
    sns.heatmap(
        cm_display,
        annot=True,
        fmt=fmt,
        cmap="YlGnBu",
        xticklabels=clean_labels,
        yticklabels=clean_labels,
        cbar=True,
    )
    plt.title(f"LeafLens Model Evaluation: {title}", fontsize=13, fontweight="bold", pad=12)
    plt.ylabel("True Class", fontsize=11, fontweight="bold")
    plt.xlabel("Predicted Class", fontsize=11, fontweight="bold")
    plt.xticks(rotation=40, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[LeafLens] Confusion matrix saved to: {save_path}")


def plot_and_save_training_curves(history_dict: dict, save_path: str):
    """
    Plots training and validation accuracy and loss side-by-side.
    """
    epochs = range(1, len(history_dict.get("accuracy", [])) + 1)
    if not epochs:
        print("[WARN] No epoch history available to plot.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.set_theme(style="whitegrid")

    # Accuracy subplot
    axes[0].plot(epochs, history_dict.get("accuracy", []), "b-o", label="Training Accuracy", linewidth=2)
    if "val_accuracy" in history_dict:
        axes[0].plot(epochs, history_dict.get("val_accuracy", []), "g--s", label="Validation Accuracy", linewidth=2)
    axes[0].set_title("Classification Accuracy Over Epochs", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Epoch", fontsize=10)
    axes[0].set_ylabel("Accuracy", fontsize=10)
    axes[0].legend(loc="lower right")
    axes[0].set_ylim([0, 1.02])

    # Loss subplot
    axes[1].plot(epochs, history_dict.get("loss", []), "r-o", label="Training Loss", linewidth=2)
    if "val_loss" in history_dict:
        axes[1].plot(epochs, history_dict.get("val_loss", []), "m--s", label="Validation Loss", linewidth=2)
    axes[1].set_title("Cross-Entropy Loss Over Epochs", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Epoch", fontsize=10)
    axes[1].set_ylabel("Loss", fontsize=10)
    axes[1].legend(loc="upper right")

    plt.suptitle("LeafLens MobileNetV2 Transfer Learning Progression", fontsize=14, fontweight="bold")
    plt.tight_layout()

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[LeafLens] Training curves saved to: {save_path}")


def evaluate_dataset(
    model: tf.keras.Model,
    test_ds: tf.data.Dataset,
    class_names: list,
    output_dir: str = "artifacts",
) -> dict:
    """
    Runs model inference over test_ds, computes accuracy, precision, recall, F1,
    saves confusion matrix plot, training curves, and returns metrics dict.
    """
    os.makedirs(output_dir, exist_ok=True)

    y_true = []
    y_pred = []
    y_probs = []

    print("[LeafLens] Generating predictions across evaluation dataset...")
    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_probs.extend(preds)
        y_pred.extend(np.argmax(preds, axis=1))
        y_true.extend(labels.numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_probs = np.array(y_probs)

    # Core scikit-learn metrics
    overall_acc = float(accuracy_score(y_true, y_pred))
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    p_weight, r_weight, f1_weight, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)

    # Detailed per-class classification report
    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    metrics = {
        "accuracy": round(overall_acc, 4),
        "macro_precision": round(float(p_macro), 4),
        "macro_recall": round(float(r_macro), 4),
        "macro_f1": round(float(f1_macro), 4),
        "weighted_precision": round(float(p_weight), 4),
        "weighted_recall": round(float(r_weight), 4),
        "weighted_f1": round(float(f1_weight), 4),
        "num_test_samples": int(len(y_true)),
        "class_names": class_names,
        "per_class": report_dict,
    }

    # Save metrics JSON
    metrics_path = os.path.join(output_dir, "evaluation_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"[LeafLens] Evaluation metrics saved to: {metrics_path}")

    # Plot Confusion Matrix
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    plot_and_save_confusion_matrix(y_true, y_pred, class_names, cm_path)

    # Print summary
    print("\n" + "=" * 50)
    print("        LEAFLENS MODEL EVALUATION SUMMARY")
    print("=" * 50)
    print(f"Overall Accuracy:       {overall_acc * 100:.2f}%")
    print(f"Macro Precision:        {p_macro * 100:.2f}%")
    print(f"Macro Recall:           {r_macro * 100:.2f}%")
    print(f"Macro F1-Score:         {f1_macro * 100:.2f}%")
    print(f"Weighted F1-Score:      {f1_weight * 100:.2f}%")
    print("=" * 50)
    print("\nClassification Report:\n")
    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate LeafLens Model")
    parser.add_argument("--model", type=str, default="models/best_model.keras", help="Path to trained model")
    parser.add_argument("--data", type=str, default="data/plantvillage_subset", help="Directory with class folders")
    parser.add_argument("--output", type=str, default="artifacts", help="Output directory for plots and metrics")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for evaluation")
    parser.add_argument("--split", type=str, default="test", choices=["test", "all"], help="Evaluate held-out test split or all samples")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"[ERROR] Trained model file not found at {args.model}")
        exit(1)

    if not os.path.exists(args.data):
        print(f"[ERROR] Dataset directory not found at {args.data}")
        exit(1)

    model = load_model_artifact(args.model)

    if args.split == "test":
        from data_loader import load_stratified_datasets
        print(f"[LeafLens] Loading stratified holdout test split from {args.data}...")
        _, _, test_ds, class_names = load_stratified_datasets(
            data_dir=args.data,
            batch_size=args.batch_size,
            train_split=0.70,
            val_split=0.15,
            test_split=0.15,
        )
        evaluate_dataset(model, test_ds, class_names, output_dir=args.output)
    else:
        print(f"[LeafLens] Loading all dataset samples from {args.data}...")
        dataset = tf.keras.utils.image_dataset_from_directory(
            args.data,
            image_size=(224, 224),
            batch_size=args.batch_size,
            shuffle=False,
        )
        class_names = dataset.class_names
        evaluate_dataset(model, dataset, class_names, output_dir=args.output)
