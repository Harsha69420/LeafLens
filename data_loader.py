"""
data_loader.py - Dataset Management, Download, Preprocessing, and Stratified Splitting.
LeafLens: Plant Disease Screening Using CNN and Transfer Learning
"""

import os
import glob
import json
import urllib.request
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import tensorflow as tf

from model import get_data_augmentation, IMG_SIZE

DEFAULT_CLASSES = [
    "Tomato___healthy",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Bacterial_spot",
    "Tomato___Leaf_Mold",
]

GITHUB_API_DIR_URL = "https://api.github.com/repos/spMohanty/PlantVillage-Dataset/contents/raw/color"
GITHUB_RAW_BASE_URL = "https://raw.githubusercontent.com/spMohanty/PlantVillage-Dataset/master/raw/color"


def download_plantvillage_subset(
    output_dir: str = "data/plantvillage_subset",
    classes: list = None,
    max_images_per_class: int = 200,
) -> str:
    """
    Downloads authentic PlantVillage leaf images from the official spMohanty/PlantVillage-Dataset
    repository on GitHub. Limits to `max_images_per_class` for rapid, reliable training.
    """
    if classes is None:
        classes = DEFAULT_CLASSES

    os.makedirs(output_dir, exist_ok=True)
    print(f"[LeafLens] Initiating PlantVillage dataset download to: {output_dir}")
    print(f"[LeafLens] Target classes: {classes}")
    print(f"[LeafLens] Samples per class: up to {max_images_per_class}")

    headers = {"User-Agent": "LeafLens-Downloader/1.0"}

    for class_name in classes:
        class_dir = os.path.join(output_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)

        existing_files = [f for f in os.listdir(class_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        if len(existing_files) >= max_images_per_class:
            print(f"[LeafLens] Class '{class_name}' already populated ({len(existing_files)} images). Skipping download.")
            continue

        api_url = f"{GITHUB_API_DIR_URL}/{class_name}"
        try:
            req = urllib.request.Request(api_url, headers=headers)
            with urllib.request.urlopen(req, timeout=20) as response:
                file_entries = json.loads(response.read().decode("utf-8"))
        except Exception as e:
            print(f"[WARN] Failed to query GitHub API for {class_name}: {e}")
            continue

        image_files = [item for item in file_entries if item.get("name", "").lower().endswith((".jpg", ".jpeg", ".png"))]
        target_files = image_files[:max_images_per_class]

        print(f"[LeafLens] Downloading {len(target_files)} authentic images for '{class_name}'...")
        count = 0
        for item in target_files:
            file_name = item["name"]
            dest_path = os.path.join(class_dir, file_name)
            if os.path.exists(dest_path):
                count += 1
                continue

            raw_url = item.get("download_url") or f"{GITHUB_RAW_BASE_URL}/{class_name}/{file_name}"
            try:
                img_req = urllib.request.Request(raw_url, headers=headers)
                with urllib.request.urlopen(img_req, timeout=15) as img_resp:
                    content = img_resp.read()
                    with open(dest_path, "wb") as f_out:
                        f_out.write(content)
                count += 1
            except Exception as dl_err:
                print(f"[WARN] Could not fetch {file_name}: {dl_err}")

        print(f"[LeafLens] Successfully saved {count} images in {class_dir}")

    return output_dir


def detect_dataset_classes(data_dir: str) -> list:
    """
    Scans directory for subdirectories containing valid leaf images and returns
    sorted list of detected class names.
    """
    if not os.path.exists(data_dir):
        return []

    classes = []
    for entry in sorted(os.listdir(data_dir)):
        entry_path = os.path.join(data_dir, entry)
        if os.path.isdir(entry_path):
            images = [f for f in os.listdir(entry_path) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
            if images:
                classes.append(entry)

    return sorted(classes)


def load_stratified_datasets(
    data_dir: str,
    batch_size: int = 32,
    train_split: float = 0.70,
    val_split: float = 0.15,
    test_split: float = 0.15,
    seed: int = 42,
):
    """
    Discovers all images and classes, performs stratified train/val/test split,
    and returns TensorFlow datasets with training data augmentation.
    """
    classes = detect_dataset_classes(data_dir)
    if not classes:
        raise ValueError(f"No valid image classes found in {data_dir}")

    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}

    filepaths = []
    labels = []

    for cls_name in classes:
        cls_dir = os.path.join(data_dir, cls_name)
        pattern = os.path.join(cls_dir, "*.*")
        for img_path in glob.glob(pattern):
            if img_path.lower().endswith((".jpg", ".jpeg", ".png")):
                filepaths.append(img_path)
                labels.append(class_to_idx[cls_name])

    filepaths = np.array(filepaths)
    labels = np.array(labels)

    print(f"[LeafLens] Discovered {len(filepaths)} total images across {len(classes)} classes.")

    # 1. Stratified split: Train vs (Val + Test)
    val_test_ratio = val_split + test_split
    X_train, X_temp, y_train, y_temp = train_test_split(
        filepaths,
        labels,
        test_size=val_test_ratio,
        stratify=labels,
        random_state=seed,
    )

    # 2. Stratified split: Val vs Test
    relative_test_ratio = test_split / val_test_ratio
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=relative_test_ratio,
        stratify=y_temp,
        random_state=seed,
    )

    print(f"[LeafLens] Stratified Dataset Split:")
    print(f"  - Training samples:   {len(X_train)} ({len(X_train)/len(filepaths)*100:.1f}%)")
    print(f"  - Validation samples: {len(X_val)} ({len(X_val)/len(filepaths)*100:.1f}%)")
    print(f"  - Testing samples:    {len(X_test)} ({len(X_test)/len(filepaths)*100:.1f}%)")

    # Image loading and decoding helper
    def parse_image_file(path, label):
        img_bytes = tf.io.read_file(path)
        img = tf.io.decode_image(img_bytes, channels=3, expand_animations=False)
        img = tf.image.resize(img, IMG_SIZE)
        img = tf.cast(img, tf.float32)
        return img, label

    # Data augmentation for training only
    aug_pipeline = get_data_augmentation()

    def augment_train_sample(img, label):
        img_aug = aug_pipeline(img, training=True)
        return img_aug, label

    # Build tf.data pipelines
    train_ds = (
        tf.data.Dataset.from_tensor_slices((X_train, y_train))
        .map(parse_image_file, num_parallel_calls=tf.data.AUTOTUNE)
        .map(augment_train_sample, num_parallel_calls=tf.data.AUTOTUNE)
        .shuffle(buffer_size=min(len(X_train), 1000), seed=seed)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    val_ds = (
        tf.data.Dataset.from_tensor_slices((X_val, y_val))
        .map(parse_image_file, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    test_ds = (
        tf.data.Dataset.from_tensor_slices((X_test, y_test))
        .map(parse_image_file, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    return train_ds, val_ds, test_ds, classes
