"""
model.py - CNN Architecture, Transfer Learning with MobileNetV2, and Preprocessing.
LeafLens: Plant Disease Screening Using CNN and Transfer Learning
"""

import os
os.environ.setdefault("KERAS_HOME", os.environ.get("TEMP", "/tmp"))
os.environ.setdefault("MPLCONFIGDIR", os.environ.get("TEMP", "/tmp"))
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
import numpy as np
from PIL import Image

# Import tensorflow lazily or safely
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import mobilenet_v2

# Fixed input shape for MobileNetV2
IMG_SIZE = (224, 224)
INPUT_SHAPE = (224, 224, 3)


def get_data_augmentation():
    """
    Keras Sequential data augmentation pipeline applied strictly to training images.
    Uses reflection padding to avoid distortion artifacts and subtle contrast adjustment.
    """
    return tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal_and_vertical", name="aug_random_flip"),
            layers.RandomRotation(0.12, fill_mode="reflect", name="aug_random_rotation"),
            layers.RandomZoom((-0.1, 0.1), fill_mode="reflect", name="aug_random_zoom"),
            layers.RandomTranslation(0.08, 0.08, fill_mode="reflect", name="aug_random_translation"),
            layers.RandomContrast(0.1, name="aug_random_contrast"),
        ],
        name="data_augmentation",
    )


def preprocess_input_image(image: Image.Image) -> np.ndarray:
    """
    Identical preprocessing for single PIL Image for inference.
    1. Converts image to RGB mode.
    2. Resizes to 224x224.
    3. Converts to numpy float32 array in [0, 255].
    4. Adds batch dimension: returns shape (1, 224, 224, 3).
    Note: MobileNetV2 preprocessing ([-1, 1] scaling) is embedded directly
    in the model's first layer for end-to-end consistency.
    """
    if image.mode != "RGB":
        image = image.convert("RGB")
    image_resized = image.resize(IMG_SIZE, Image.Resampling.BILINEAR)
    img_array = np.array(image_resized, dtype=np.float32)
    return np.expand_dims(img_array, axis=0)


def build_mobilenetv2_model(
    num_classes: int,
    input_shape: tuple = INPUT_SHAPE,
    learning_rate: float = 5e-4,
    dropout_rate: float = 0.35,
    freeze_backbone: bool = True,
) -> tf.keras.Model:
    """
    Constructs the LeafLens transfer learning model:
    - Pretrained MobileNetV2 backbone (ImageNet weights)
    - Global Average Pooling
    - Batch Normalization
    - Refined 2-stage dense feature projection (384 -> 128 units with L2 regularization)
    - Balanced Dropout for regularization
    - Dense softmax classification layer (num_classes)
    """
    # 1. Base Pretrained Backbone
    base_model = mobilenet_v2.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet",
    )

    # 2. Backbone Freezing
    base_model.trainable = not freeze_backbone

    # 3. Functional API assembly
    inputs = tf.keras.Input(shape=input_shape, name="leaf_image_input")

    # MobileNetV2 preprocessing layer embedded inside model for end-to-end consistency
    x = layers.Lambda(mobilenet_v2.preprocess_input, name="mobilenetv2_preprocessing")(inputs)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D(name="global_avg_pooling")(x)
    x = layers.BatchNormalization(name="batch_norm")(x)
    
    # Hierarchical feature projection
    x = layers.Dense(384, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-4), name="dense_features_1")(x)
    x = layers.Dropout(dropout_rate, name="dropout_1")(x)
    x = layers.Dense(128, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-4), name="dense_features_2")(x)
    x = layers.Dropout(max(0.1, dropout_rate - 0.1), name="dropout_2")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="disease_prediction")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="LeafLens_MobileNetV2")

    # Compile model
    optimizer = optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def unfreeze_for_fine_tuning(
    model: tf.keras.Model,
    unfreeze_layers: int = 40,
    fine_tune_lr: float = 2e-5,
) -> tf.keras.Model:
    """
    Unfreezes the top N layers of the MobileNetV2 backbone for fine-tuning.
    Keeps BatchNormalization frozen to preserve running statistics.
    Recompiles with a reduced learning rate.
    """
    # Find base model layer
    base_model = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model) or (hasattr(layer, "layers") and len(layer.layers) > 10):
            base_model = layer
            break

    if base_model is None:
        print("[WARN] MobileNetV2 backbone submodel not directly found. Setting top layers trainable.")
        for layer in model.layers[-unfreeze_layers:]:
            layer.trainable = True
    else:
        base_model.trainable = True
        # Freeze all layers except the last `unfreeze_layers`
        for layer in base_model.layers[:-unfreeze_layers]:
            layer.trainable = False
        for layer in base_model.layers[-unfreeze_layers:]:
            if not isinstance(layer, layers.BatchNormalization):
                layer.trainable = True

    model.compile(
        optimizer=optimizers.Adam(learning_rate=fine_tune_lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def save_model_artifact(model: tf.keras.Model, save_path: str):
    """Safely saves the trained model in Keras format."""
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    model.save(save_path)
    print(f"[LeafLens] Model successfully saved to: {save_path}")


def load_model_artifact(model_path: str) -> tf.keras.Model:
    """Loads a saved Keras model artifact."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")
    model = tf.keras.models.load_model(model_path, compile=False)
    return model
