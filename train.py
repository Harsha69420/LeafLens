"""
train.py - Model Training Pipeline Using MobileNetV2 and Transfer Learning.
LeafLens: Plant Disease Screening Using CNN and Transfer Learning
"""

import os
import json
import argparse
os.environ.setdefault("MPLCONFIGDIR", os.environ.get("TEMP", "/tmp"))
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
import tensorflow as tf

from model import build_mobilenetv2_model, unfreeze_for_fine_tuning, save_model_artifact
from data_loader import download_plantvillage_subset, load_stratified_datasets, detect_dataset_classes
from evaluate import evaluate_dataset, plot_and_save_training_curves


def run_training_pipeline(
    data_dir: str = "data/plantvillage_subset",
    models_dir: str = "models",
    artifacts_dir: str = "artifacts",
    batch_size: int = 32,
    epochs_stage1: int = 12,
    epochs_stage2: int = 8,
    learning_rate: float = 5e-4,
    fine_tune: bool = True,
    download_if_missing: bool = True,
    max_images_per_class: int = 150,
):
    """
    Executes the end-to-end training and evaluation pipeline:
    1. Ensures dataset availability (auto-downloads PlantVillage tomato subset if needed).
    2. Dynamically detects classes.
    3. Partitions data via stratified sampling (train/val/test).
    4. Applies data augmentation to training images.
    5. Builds MobileNetV2 with frozen backbone and custom classification head.
    6. Trains head with EarlyStopping and ModelCheckpoint.
    7. Optionally fine-tunes top backbone layers with reduced learning rate.
    8. Evaluates on holdout test set, saves plots and metrics.
    """
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(artifacts_dir, exist_ok=True)

    # 1. Dataset verification or download
    detected_classes = detect_dataset_classes(data_dir)
    if not detected_classes and download_if_missing:
        print(f"[LeafLens] Dataset not found in '{data_dir}'. Auto-downloading PlantVillage subset...")
        download_plantvillage_subset(
            output_dir=data_dir,
            max_images_per_class=max_images_per_class,
        )
        detected_classes = detect_dataset_classes(data_dir)

    if not detected_classes:
        raise RuntimeError(
            f"No classes found in '{data_dir}'. Please verify the dataset path or enable internet access to download."
        )

    print(f"\n[LeafLens] Automatically detected classes ({len(detected_classes)}):")
    for idx, cname in enumerate(detected_classes):
        print(f"  [{idx}] {cname}")

    # 2. Stratified Data Loading
    train_ds, val_ds, test_ds, classes = load_stratified_datasets(
        data_dir=data_dir,
        batch_size=batch_size,
        train_split=0.70,
        val_split=0.15,
        test_split=0.15,
    )

    num_classes = len(classes)

    # Save class names mapping for inference
    class_map_path = os.path.join(models_dir, "class_names.json")
    with open(class_map_path, "w") as f:
        json.dump(classes, f, indent=4)
    print(f"[LeafLens] Class mappings saved to: {class_map_path}")

    # Class weights to address Early Blight misclassification (class index 1)
    class_weights = {0: 1.0, 1: 1.4, 2: 1.0, 3: 1.0, 4: 1.0}
    print(f"[LeafLens] Utilizing class weighting: {class_weights}")

    # 3. Model Construction (Transfer Learning with MobileNetV2)
    print("\n[LeafLens] Building MobileNetV2 transfer learning model...")
    model = build_mobilenetv2_model(
        num_classes=num_classes,
        learning_rate=learning_rate,
        dropout_rate=0.35,
        freeze_backbone=True,
    )
    model.summary()

    # 4. Training Callbacks
    best_model_path = os.path.join(models_dir, "best_model.keras")
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=best_model_path,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    # 5. Stage 1: Feature Extraction (Train classification head)
    print("\n" + "=" * 60)
    print(" STAGE 1: Training Classification Head (Backbone Frozen)")
    print("=" * 60)

    history1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs_stage1,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1,
    )

    combined_history = {
        "accuracy": list(history1.history.get("accuracy", [])),
        "val_accuracy": list(history1.history.get("val_accuracy", [])),
        "loss": list(history1.history.get("loss", [])),
        "val_loss": list(history1.history.get("val_loss", [])),
    }

    # 6. Stage 2: Fine-Tuning (Optional unfreezing)
    if fine_tune and epochs_stage2 > 0:
        print("\n" + "=" * 60)
        print(" STAGE 2: Fine-Tuning Top 40 MobileNetV2 Backbone Layers")
        print("=" * 60)
        model = unfreeze_for_fine_tuning(model, unfreeze_layers=40, fine_tune_lr=2e-5)

        callbacks_stage2 = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=4,
                restore_best_weights=True,
                verbose=1,
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=best_model_path,
                monitor="val_accuracy",
                save_best_only=True,
                verbose=1,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=2,
                min_lr=1e-7,
                verbose=1,
            ),
        ]

        history2 = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs_stage1 + epochs_stage2,
            initial_epoch=len(history1.history["loss"]),
            callbacks=callbacks_stage2,
            class_weight=class_weights,
            verbose=1,
        )

        combined_history["accuracy"].extend(history2.history.get("accuracy", []))
        combined_history["val_accuracy"].extend(history2.history.get("val_accuracy", []))
        combined_history["loss"].extend(history2.history.get("loss", []))
        combined_history["val_loss"].extend(history2.history.get("val_loss", []))

    # Save final model if not already saved by checkpoint
    if not os.path.exists(best_model_path):
        save_model_artifact(model, best_model_path)

    # Save combined training curves
    curves_path = os.path.join(artifacts_dir, "training_curves.png")
    plot_and_save_training_curves(combined_history, curves_path)

    # Save raw training history json
    history_json_path = os.path.join(artifacts_dir, "training_history.json")
    with open(history_json_path, "w") as f:
        json.dump(combined_history, f, indent=4)

    # 7. Comprehensive Holdout Test Evaluation
    print("\n" + "=" * 60)
    print(" STAGE 3: Holdout Test Set Evaluation")
    print("=" * 60)
    best_model = tf.keras.models.load_model(best_model_path, compile=False)
    metrics = evaluate_dataset(
        model=best_model,
        test_ds=test_ds,
        class_names=classes,
        output_dir=artifacts_dir,
    )

    print("\n[LeafLens] Training pipeline completed successfully!")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LeafLens Plant Disease Screening Model")
    parser.add_argument("--data_dir", type=str, default="data/plantvillage_subset", help="Dataset directory")
    parser.add_argument("--models_dir", type=str, default="models", help="Directory to save model artifacts")
    parser.add_argument("--artifacts_dir", type=str, default="artifacts", help="Directory to save metrics and plots")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs_stage1", type=int, default=12, help="Head training epochs")
    parser.add_argument("--epochs_stage2", type=int, default=8, help="Fine tuning epochs")
    parser.add_argument("--lr", type=float, default=5e-4, help="Stage 1 learning rate")
    parser.add_argument("--no_finetune", action="store_true", help="Disable stage 2 fine tuning")
    parser.add_argument("--max_images", type=int, default=150, help="Max images per class to download if missing")
    args = parser.parse_args()

    run_training_pipeline(
        data_dir=args.data_dir,
        models_dir=args.models_dir,
        artifacts_dir=args.artifacts_dir,
        batch_size=args.batch_size,
        epochs_stage1=args.epochs_stage1,
        epochs_stage2=0 if args.no_finetune else args.epochs_stage2,
        learning_rate=args.lr,
        fine_tune=not args.no_finetune,
        max_images_per_class=args.max_images,
    )
