"""
gradcam.py - Gradient-Weighted Class Activation Mapping (Grad-CAM) for LeafLens.
Explains MobileNetV2 CNN disease screening predictions by localizing salient foliar regions.
"""

import os
os.environ.setdefault("MPLCONFIGDIR", os.environ.get("TEMP", "/tmp"))
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.cm as cm
import matplotlib.pyplot as plt


def find_target_conv_layer(model: tf.keras.Model) -> str:
    """
    Locates the deepest convolutional or activation layer inside MobileNetV2 or the model.
    Typically 'out_relu' or the last Conv2D in MobileNetV2.
    """
    # Check if nested base_model exists
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.Model) or (hasattr(layer, "layers") and len(layer.layers) > 10):
            for sublayer in reversed(layer.layers):
                if "relu" in sublayer.name.lower() or isinstance(sublayer, (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D)):
                    return f"{layer.name}/{sublayer.name}"
        if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D)):
            return layer.name
        if "conv" in layer.name.lower() or "relu" in layer.name.lower():
            return layer.name

    return "out_relu"


def generate_gradcam_heatmap(
    model: tf.keras.Model,
    img_array: np.ndarray,
    class_idx: int = None,
    conv_layer_name: str = None,
) -> np.ndarray:
    """
    Computes a Grad-CAM heatmap for an input image and a specified class index.

    Args:
        model: Trained Keras model.
        img_array: Preprocessed numpy array of shape (1, 224, 224, 3).
        class_idx: Target class index. If None, uses top predicted class.
        conv_layer_name: Name of target convolutional layer.

    Returns:
        heatmap: 2D numpy array of shape (H, W) normalized between 0 and 1.
    """
    # Find base model if nested
    base_model = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model) or (hasattr(layer, "layers") and len(layer.layers) > 10):
            base_model = layer
            break

    # If base_model is a submodel inside model
    if base_model is not None:
        if conv_layer_name is None:
            # 'out_relu' is the standard last activation layer of MobileNetV2
            conv_layer_name = "out_relu"

        try:
            target_layer = base_model.get_layer(conv_layer_name)
        except ValueError:
            # Fallback to last conv2d
            target_layer = [l for l in base_model.layers if isinstance(l, (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D))][-1]
            conv_layer_name = target_layer.name

        # Create feature extractor for backbone
        backbone_grad_model = tf.keras.models.Model(
            inputs=[base_model.input],
            outputs=[target_layer.output, base_model.output]
        )

        # Build classification head model
        # Find layers after base_model
        head_input = tf.keras.Input(shape=base_model.output.shape[1:])
        x = head_input
        found_base = False
        for l in model.layers:
            if l == base_model:
                found_base = True
                continue
            if found_base:
                x = l(x)
        head_model = tf.keras.models.Model(inputs=[head_input], outputs=[x])

        with tf.GradientTape() as tape:
            # Scale input to [-1, 1] as expected by the MobileNetV2 backbone
            scaled_input = tf.keras.applications.mobilenet_v2.preprocess_input(tf.cast(img_array, tf.float32))
            conv_outputs, backbone_out = backbone_grad_model(scaled_input)
            tape.watch(conv_outputs)
            predictions = head_model(backbone_out)
            if class_idx is None:
                class_idx = tf.argmax(predictions[0])
            loss = predictions[:, class_idx]

        grads = tape.gradient(loss, conv_outputs)

    else:
        # Flat model
        if conv_layer_name is None:
            conv_layer_name = find_target_conv_layer(model)
        grad_model = tf.keras.models.Model(
            inputs=[model.inputs],
            outputs=[model.get_layer(conv_layer_name).output, model.output]
        )

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            if class_idx is None:
                class_idx = tf.argmax(predictions[0])
            loss = predictions[:, class_idx]

        grads = tape.gradient(loss, conv_outputs)

    # Global Average Pooling of the gradients (importance weights)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weight the convolution feature maps
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Apply ReLU (only features that contribute positively to the class)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
    return heatmap.numpy()


def overlay_gradcam(
    original_pil_img: Image.Image,
    heatmap: np.ndarray,
    alpha: float = 0.45,
    colormap_name: str = "jet",
) -> Image.Image:
    """
    Superimposes the Grad-CAM heatmap onto the original PIL image.

    Args:
        original_pil_img: PIL Image of the leaf.
        heatmap: 2D numpy array normalized between 0 and 1.
        alpha: Heatmap blend opacity (0.0 to 1.0).
        colormap_name: Matplotlib colormap name.

    Returns:
        PIL Image with superimposed Grad-CAM heatmap.
    """
    img = original_pil_img.convert("RGB").resize((224, 224))
    img_arr = np.array(img, dtype=np.float32)

    # Resize heatmap to match image dimensions
    heatmap_uint8 = np.uint8(255 * np.clip(heatmap, 0, 1))
    heatmap_pil = Image.fromarray(heatmap_uint8).resize((224, 224), Image.Resampling.BILINEAR)
    heatmap_resized = np.array(heatmap_pil, dtype=np.float32) / 255.0

    # Colorize heatmap
    try:
        colormap = plt.colormaps[colormap_name]
    except Exception:
        colormap = cm.get_cmap(colormap_name)
    heatmap_colored = colormap(heatmap_resized)[:, :, :3]  # Discard alpha
    heatmap_colored = np.uint8(255 * heatmap_colored)

    # Superimpose heatmap
    blended = (1.0 - alpha) * img_arr + alpha * heatmap_colored
    blended = np.clip(blended, 0, 255).astype(np.uint8)

    return Image.fromarray(blended)
