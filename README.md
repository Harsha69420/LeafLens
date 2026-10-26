<div align="center">

# LeafLens

**Plant Disease Screening Using Deep Convolutional Neural Networks and Transfer Learning**

[![Python Versions](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![MobileNetV2](https://img.shields.io/badge/Backbone-MobileNetV2-2E7D32?logo=google&logoColor=white)](https://keras.io/api/applications/mobilenet/)
[![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Holdout Test Accuracy](https://img.shields.io/badge/Test%20Accuracy-80.00%25-2E7D32)](artifacts/evaluation_metrics.json)
[![Macro F1-Score](https://img.shields.io/badge/Macro%20F1-79.87%25-2E7D32)](artifacts/evaluation_metrics.json)

</div>

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [Supported Tomato Pathologies](#supported-tomato-pathologies)
- [Application Screenshots](#application-screenshots)
- [Model Architecture & Pipeline](#model-architecture--pipeline)
- [Explainable AI (Grad-CAM)](#explainable-ai-grad-cam)
- [Dataset & Data Leakage Prevention](#dataset--data-leakage-prevention)
- [Training & Optimization Strategy](#training--optimization-strategy)
- [Holdout Test Results & Benchmarks](#holdout-test-results--benchmarks)
- [Installation & Environment Setup](#installation--environment-setup)
- [Usage Guide](#usage-guide)
- [Repository Structure](#repository-structure)
- [Troubleshooting](#troubleshooting)
- [Limitations & Future Work](#limitations--future-work)
- [Educational Disclaimer](#educational-disclaimer)
- [License & Acknowledgements](#license--acknowledgements)

---

## Project Overview

**LeafLens** is an end-to-end computer vision and deep learning screening application engineered to classify foliar pathologies in tomato crops (*Solanum lycopersicum*). Built upon **MobileNetV2** transfer learning, LeafLens pairs edge-efficient convolutional feature extraction with **Gradient-Weighted Class Activation Mapping (Grad-CAM)** to provide transparent, interpretable visual evidence alongside probabilistic disease predictions.

### Intended Use
LeafLens serves as an educational screening prototype and algorithmic decision-support tool. It assists students, agronomists, and developers in exploring deep learning transfer learning workflows, convolutional attention mechanisms, and foliar disease diagnosis.

### Operational Scope & Limitations
LeafLens is calibrated on laboratory-curated leaf photographs from the PlantVillage dataset. Its performance on complex in-field farm photographs (featuring cluttered soil backgrounds, motion blur, overlapping leaves, extreme sunlight glare, or co-occurring pest damage) has not been clinically validated.

---

## Key Features

- **Five-Class Foliar Disease Classification:** Detects healthy foliage and four major tomato pathologies (Bacterial Spot, Early Blight, Late Blight, and Leaf Mold).
- **MobileNetV2 Transfer Learning:** Pretrained on ImageNet to exploit rich hierarchical low-level edge and texture detectors while maintaining a lightweight footprint (~2.8M parameters).
- **Dual Foliage Input Modes:** Upload user-provided leaf photographs (`PNG`, `JPG`, `JPEG`) or select from authentic preloaded PlantVillage dataset samples directly in the UI.
- **Full Five-Class Probability Breakdown:** Visualizes confidence percentages across all five target categories using an interactive Altair horizontal bar chart.
- **Explainable AI via Grad-CAM:** Generates spatial salience heatmaps from the final convolutional layer (`out_relu`), with an interactive opacity slider ($0.20 \text{ to } 0.80$) to overlay directly over the leaf.
- **Integrated Model Evaluation Dashboard:** Surfaces holdout test KPI metric cards (Accuracy, Macro Precision, Macro Recall, Macro F1-Score), high-resolution normalized confusion matrix, and two-stage training history curves.
- **Agronomic Guidance & Clinical Symptoms:** Displays pathogen profiles, field symptom checklists, and cautious cultural next steps.
- **Zero-Leakage Partitioning:** Enforces strict stratified splitting (70% train / 15% validation / 15% test) with dynamic data augmentation confined strictly to training tensors.
- **Responsive Botanical UI:** Built with custom Streamlit styling utilizing an earthy botanical palette (`#1b4332`, `#2d6a4f`, `#d8f3dc`) and modular tabs.

---

## Supported Tomato Pathologies

The system automatically detects and maps five classes based on the dataset directory structure:

| Class Index | Exact Label Identifier | Clinical Pathology Name | Pathogen Type | Clinical Severity |
|:---:|:---|:---|:---|:---|
| `0` | `Tomato___Bacterial_spot` | Bacterial Spot (*Xanthomonas spp.*) | Bacteria | Moderate to High |
| `1` | `Tomato___Early_blight` | Early Blight (*Alternaria solani*) | Fungus | Moderate to High |
| `2` | `Tomato___Late_blight` | Late Blight (*Phytophthora infestans*) | Oomycete | Critical |
| `3` | `Tomato___Leaf_Mold` | Leaf Mold (*Passalora fulva*) | Fungus | Moderate |
| `4` | `Tomato___healthy` | Healthy Foliage | None | None (Healthy) |

*The class-to-index mapping is dynamically generated during training and persisted in `models/class_names.json` to ensure consistency across training, evaluation, and production inference.*

---

## Application Screenshots

The LeafLens Streamlit frontend organizes screening, explainability, and evaluation workflows into structured, responsive views.

### 1. Foliar Disease Screening & Probabilistic Prediction
Users can inspect foliage by selecting authentic sample leaves or uploading photos to obtain real-time diagnostic classifications and class probability distributions.

| Foliar Screening Dashboard | Disease Prediction & Probability Spread |
|:---:|:---:|
| ![LeafLens Dashboard](docs/screenshots/01-dashboard.png) | ![Disease Prediction](docs/screenshots/02-disease-prediction.png) |
| *Figure 1: Main screening interface with leaf preview and classification controls.* | *Figure 2: Diagnosis result card with confidence percentage and 5-class distribution.* |

### 2. Explainable AI & Agronomic Guidance
To verify that the model attends to authentic foliar lesions rather than background cues, LeafLens visualizes spatial activation heatmaps alongside actionable agronomic profiles.

| Grad-CAM Salience Overlay | Agronomic Management Profile |
|:---:|:---:|
| ![Grad-CAM Salience Overlay](docs/screenshots/03-gradcam.png) | ![Agronomic Guidance](docs/screenshots/04-bacterial-spot-agronomic-guidance.png) |
| *Figure 3: Side-by-side comparison of preprocessed input and Grad-CAM salience heatmap.* | *Figure 4: Agronomic profile displaying symptom checklists and management actions.* |

### 3. Model Evaluation & Benchmarks
The built-in analytics dashboard displays holdout test benchmarks, confusion matrices, and optimization comparisons directly in the user interface.

| Model Evaluation Dashboard | Baseline vs. Improved Comparison |
|:---:|:---:|
| ![Model Evaluation Dashboard](docs/screenshots/05-model-evaluation-dashboard.png) | ![Performance Comparison](docs/screenshots/06-performance-comparison.png) |
| *Figure 5: Test KPI cards, normalized confusion matrix, and training dynamics.* | *Figure 6: Before-and-after performance comparison across holdout metrics.* |

---

## Model Architecture & Pipeline

LeafLens implements a transfer learning architecture with a custom regularized classification head and an internal preprocessing layer:

```
Input Leaf Tensor: (224 × 224 × 3 RGB, Range [0, 255])
           │
           ▼
Internal Keras Lambda Layer: (x / 127.5) - 1.0  ===>  Range: [-1.0, 1.0]
           │
           ▼
Pretrained MobileNetV2 Backbone (1280 feature maps, ImageNet weights)
           │
           ▼
GlobalAveragePooling2D: (1280-dimensional feature vector)
           │
           ▼
Batch Normalization (Channel Stabilizer)
           │
           ▼
Dense Layer 1: 384 units, ReLU, L2 Regularization (1e-4)
           │
           ▼
Dropout Layer 1: rate = 0.35
           │
           ▼
Dense Layer 2: 128 units, ReLU, L2 Regularization (1e-4)
           │
           ▼
Dropout Layer 2: rate = 0.25
           │
           ▼
Dense Layer (Softmax): 5 Disease Class Probabilities
```

### Preprocessing Consistency
MobileNetV2 requires inputs normalized to $[-1.0, 1.0]$. To eliminate double-scaling bugs and ensure inference precisely mirrors training:
- Input images are loaded as standard PIL RGB images and resized to $224 \times 224 \times 3$.
- Mathematical conversion `(x / 127.5) - 1.0` is encapsulated inside a Keras `Lambda` layer as the very first layer of the model.
- Raw float32 arrays in $[0, 255]$ are passed directly to `model.predict()`.

### Parameter Summary
- **Total Parameters:** 2,804,933 (~10.70 MB)
- **Trainable Parameters (Stage 1):** 544,389 (Head only)
- **Trainable Parameters (Stage 2):** 1,605,829 (Head + Top 40 backbone layers, excluding BatchNorm)
- **Non-Trainable Parameters:** 1,199,104

---

## Explainable AI (Grad-CAM)

To inspect and visually verify foliar regions influencing the model's predictions, LeafLens implements **Gradient-Weighted Class Activation Mapping (Grad-CAM)**:

1. **Feature Extraction:** Extracts the 2D spatial feature activations $A^k$ of the deepest convolutional layer in MobileNetV2 (`out_relu`, dimension $7 \times 7 \times 1280$).
2. **Gradient Backpropagation:** Computes the gradient of the predicted class logit $y^c$ with respect to each activation map $A^k$:
   $$\alpha_k^c = \frac{1}{Z} \sum_{i=1}^{U} \sum_{j=1}^{V} \frac{\partial y^c}{\partial A_{i,j}^k}$$
3. **Rectified Weighting:** Computes the weighted combination and applies ReLU to isolate features positively contributing to the predicted disease:
   $$L_{\text{Grad-CAM}}^c = \text{ReLU}\left( \sum_{k} \alpha_k^c A^k \right)$$
4. **Bilinear Upsampling & Overlay:** Normalizes the $(7, 7)$ map to $[0, 1]$, bilinearly interpolates to $(224, 224)$, applies the Jet colormap, and blends with the original image via alpha compositing:
   $$I_{\text{blended}} = \alpha \cdot \text{Colormap} + (1 - \alpha) \cdot I_{\text{original}}$$

> [!NOTE]
> **Interpretability Scope:** Grad-CAM highlights coarse convolutional activation regions that strongly correlate with the model's classification score. These visual overlays serve as a diagnostic check to verify whether attention is centered on foliar lesions rather than background artifacts; they do not constitute formal proof of causal botanical reasoning.

---

## Dataset & Data Leakage Prevention

### Source Dataset
LeafLens utilizes authentic leaf imagery from the public **PlantVillage Dataset**:
- **Dataset Repository:** [spMohanty/PlantVillage-Dataset](https://github.com/spMohanty/PlantVillage-Dataset)
- **Scope:** The project downloads a curated subset of **600 authentic tomato images** across five classes (120 images per class), rather than the entire multi-gigabyte repository.

### Data Partitioning
Partitions are constructed with fixed-seed stratification (`seed=42`) using `scikit-learn`'s `train_test_split`:

| Subset Split | Percentage | Number of Images | Per-Class Allocation |
|:---|:---:|:---:|:---:|
| **Training Set** | 70% | 420 | 84 images per class |
| **Validation Set** | 15% | 90 | 18 images per class |
| **Holdout Test Set** | 15% | 90 | 18 images per class |
| **Total Authentic Images** | **100%** | **600** | **120 images per class** |

### Leakage Prevention Protocol
1. **Pristine Holdout Split:** The 90-image holdout test set is partitioned upfront and remains untouched during all training and hyperparameter tuning phases.
2. **Training-Only Data Augmentation:** Spatial transformations (random rotations of $\pm 12\%$, random zooms of $\pm 10\%$, random horizontal/vertical flips, spatial translations of $\pm 8\%$, and random contrast of $\pm 10\%$) are applied dynamically inside the training `tf.data.Dataset` stream. Validation and test sets are never augmented.

---

## Training & Optimization Strategy

The training pipeline in [`train.py`](train.py) executes in two phases:

### Stage 1: Classification Head Training (Frozen Backbone)
- **Objective:** Train newly initialized classification head without disturbing ImageNet feature extractors.
- **Optimizer:** Adam with initial learning rate $\eta = 5 \times 10^{-4}$.
- **Loss Function:** `SparseCategoricalCrossentropy` with class-weighted penalties.
- **Epochs:** 12 epochs with batch size 32.
- **Callbacks:** `ModelCheckpoint` tracking `val_accuracy`, and `ReduceLROnPlateau` (factor 0.5, patience 2, min learning rate $10^{-6}$).

### Stage 2: Controlled Backbone Fine-Tuning
- **Objective:** Adapt top convolutional representations to foliar pathogen morphology.
- **Unfrozen Layers:** Top 40 layers of MobileNetV2.
- **Frozen Batch Normalization:** All `BatchNormalization` layers inside MobileNetV2 are locked in inference mode (`layer.trainable = False`). This prevents running mean and variance corruption caused by small batch statistics ($N=32$).
- **Optimizer:** Adam with fine-tuning learning rate $\eta = 2 \times 10^{-5}$.
- **Epochs:** 8 epochs with batch size 32.

### Early Blight Class-Weighting Strategy
Because Early Blight lesions exhibit concentric dark rings that visually mimic Bacterial Spot specks and Late Blight necrosis, baseline models suffer from false-negative under-recall. LeafLens introduces a calibrated loss multiplier for class index 1:
$$\text{Class Weights} = \{0: 1.0, \; 1: 1.4, \; 2: 1.0, \; 3: 1.0, \; 4: 1.0\}$$
This penalizes Early Blight omissions by $40\%$, elevating Early Blight test recall from **38.89% to 66.67%**.

---

## Holdout Test Results & Benchmarks

All metrics below are verified from [`artifacts/evaluation_metrics.json`](artifacts/evaluation_metrics.json), evaluated strictly on the **90-image holdout test set** (18 images per class, `seed=42`):

### Baseline vs. Improved Model Comparison

| Evaluation Metric | Baseline Model | Improved LeafLens Model | Absolute Delta | Relative Gain |
|:---|:---:|:---:|:---:|:---:|
| **Overall Test Accuracy** | **72.22%** | **80.00%** | **+7.78%** | **+10.8%** |
| **Macro F1-Score** | **71.42%** | **79.87%** | **+8.45%** | **+11.8%** |
| **Macro Precision** | **74.53%** | **81.20%** | **+6.67%** | **+8.9%** |
| **Macro Recall** | **72.22%** | **80.00%** | **+7.78%** | **+10.8%** |
| **Early Blight Recall** | **38.89%** | **66.67%** | **+27.78%** | **+71.4%** |
| **Early Blight F1-Score** | **51.85%** | **66.67%** | **+14.82%** | **+28.6%** |
| **Bacterial Spot Recall** | **77.78%** | **94.44%** | **+16.66%** | **+21.4%** |
| **Bacterial Spot F1-Score** | **71.11%** | **80.95%** | **+9.84%** | **+13.8%** |
| **Healthy Leaf Recall** | **83.33%** | **94.44%** | **+11.11%** | **+13.3%** |
| **Healthy Leaf F1-Score** | **85.71%** | **91.89%** | **+6.18%** | **+7.2%** |
| **Late Blight Precision** | **91.67%** | **93.33%** | **+1.66%** | **+1.8%** |

*Both models were evaluated on the identical 90-image holdout test split (`test_ds`, $N=90$, seed 42).*

### Per-Class Performance Breakdown (Improved Model)

| Class Identifier | Precision | Recall | F1-Score | Support (Holdout Images) |
|:---|:---:|:---:|:---:|:---:|
| `Tomato___Bacterial_spot` | 70.83% | 94.44% | 80.95% | 18 |
| `Tomato___Early_blight` | 66.67% | 66.67% | 66.67% | 18 |
| `Tomato___Late_blight` | 93.33% | 77.78% | 84.85% | 18 |
| `Tomato___Leaf_Mold` | 85.71% | 66.67% | 75.00% | 18 |
| `Tomato___healthy` | 89.47% | 94.44% | 91.89% | 18 |
| **Macro Average** | **81.20%** | **80.00%** | **79.87%** | **90** |
| **Weighted Average** | **81.20%** | **80.00%** | **79.87%** | **90** |

*Evaluation artifacts persisted in repository:*
- Confusion Matrix: [`artifacts/confusion_matrix.png`](artifacts/confusion_matrix.png)
- Training Dynamics: [`artifacts/training_curves.png`](artifacts/training_curves.png)
- Structured Metrics JSON: [`artifacts/evaluation_metrics.json`](artifacts/evaluation_metrics.json)

---

## Installation & Environment Setup

### 1. Prerequisites & Python Compatibility

LeafLens is recommended and tested for **Python 3.10, 3.11, or 3.12** on Windows, Linux, or macOS:

> [!IMPORTANT]
> **Recommended Python Environment:**
> - **Python 3.10 – 3.12 (Supported & Recommended):** PyPI provides official stable binary wheels for TensorFlow (`tensorflow>=2.15.0,<=2.18.0`) across all supported operating systems.
> - **Python 3.14 (Not Recommended for Production):** Python 3.14 is not officially supported by stable TensorFlow wheels on PyPI. While a pre-release wheel (`tensorflow==2.22.0rc0`) was used during initial local development on Windows, reproducible and stable deployment requires **Python 3.10–3.12**.

### 2. Windows PowerShell Setup Guide

Clone the repository and open a terminal inside the project root:

```powershell
# Clone the repository
git clone https://github.com/<your-username>/LeafLens.git
cd LeafLens

# Create a virtual environment with Python 3.10 - 3.12
python -m venv venv

# If PowerShell blocks script execution, temporarily adjust ExecutionPolicy:
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

# Activate virtual environment
.\venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

Install all verified packages from [`requirements.txt`](requirements.txt):

```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

#### Package Manifest:
```text
tensorflow>=2.15.0
streamlit>=1.30.0
numpy>=1.24.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
pillow>=9.5.0
seaborn>=0.12.0
altair>=5.0.0
requests>=2.28.0
```

---

## Usage Guide

All scripts include sensible defaults; run them directly without mandatory flags.

### 1. Launch the Streamlit Web Application (Standard Usage)
```powershell
streamlit run app.py
```
*Opens automatically at `http://localhost:8501`. Retraining is not triggered; the cached model artifact is loaded instantly.*

### 2. Run the End-to-End Verification Suite
Executes all 7 automated system tests (environment, dataset structure, model loading, class mapping, holdout metrics, Grad-CAM execution, and app syntax):
```powershell
python verify_all.py
```

### 3. Test Predictions & Grad-CAM Across All 5 Classes
Performs single-image inference and generates Grad-CAM overlays for each of the five tomato classes:
```powershell
python test_5_classes.py
```
*Generated overlays are saved to `artifacts/verification_cam/`.*

### 4. Evaluate Checkpoint on Holdout Test Split
Evaluates the trained model against the untouched 90-image test set and updates evaluation artifacts:
```powershell
python evaluate.py --model models/best_model.keras --data data/plantvillage_subset --split test --output artifacts
```

### 5. Download Dataset Subset (Optional)
Downloads authentic tomato leaf images from PlantVillage (120 per class) if not already present:
```powershell
python download_data.py data/plantvillage_subset 120
```

### 6. Retrain the Model from Scratch (Optional)
Executes two-stage training with class re-weighting, validation checkpointing, and test evaluation:
```powershell
python train.py --epochs_stage1 12 --epochs_stage2 8 --batch_size 32
```

---

## Repository Structure

```
LeafLens/
├── app.py                      # Responsive Streamlit web application with botanical theme
├── train.py                    # Two-stage MobileNetV2 transfer learning pipeline
├── model.py                    # CNN architecture, preprocessing Lambda & fine-tuning logic
├── evaluate.py                 # Holdout test evaluation, confusion matrix & curves generator
├── data_loader.py              # Stratified data splitting (70/15/15) & augmentation pipeline
├── disease_info.py             # Agronomic disease descriptions, symptoms & management advice
├── gradcam.py                  # Gradient-Weighted Class Activation Mapping implementation
├── verify_all.py               # Complete 7-test automated verification test suite
├── test_5_classes.py           # Single-image prediction & Grad-CAM visual tester
├── download_data.py            # Multithreaded PlantVillage subset downloader
├── requirements.txt            # Python dependencies with version specifications
├── README.md                   # Technical system documentation
├── data/
│   └── plantvillage_subset/    # 600 authentic PlantVillage images (5 class subdirectories)
├── models/
│   ├── best_model.keras        # Optimized trained model checkpoint (80.00% test accuracy)
│   ├── best_model_baseline.keras # Preserved baseline model checkpoint (72.22% test accuracy)
│   └── class_names.json        # Verified class-to-index JSON mapping
├── artifacts/
│   ├── evaluation_metrics.json # Full test metrics (accuracy, precision, recall, F1)
│   ├── confusion_matrix.png    # High-resolution normalized confusion matrix plot
│   ├── training_curves.png     # Two-stage accuracy and loss progression curves
│   ├── training_history.json   # Epoch-by-epoch metric logs
│   └── verification_cam/       # Verified Grad-CAM heatmaps for all 5 classes
└── docs/
    └── screenshots/            # Verified application interface screenshots
        ├── 01-dashboard.png
        ├── 02-disease-prediction.png
        ├── 03-gradcam.png
        ├── 04-bacterial-spot-agronomic-guidance.png
        ├── 05-model-evaluation-dashboard.png
        └── 06-performance-comparison.png
```

---

## Troubleshooting

### 1. PowerShell Script Execution Error (`PSSecurityException`)
- **Symptom:** `.\venv\Scripts\Activate.ps1 cannot be loaded because running scripts is disabled on this system.`
- **Resolution:** Run PowerShell as Administrator or execute:
  ```powershell
  Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
  .\venv\Scripts\Activate.ps1
  ```

### 2. Python Version & TensorFlow Wheel Resolution
- **Symptom:** `pip install -r requirements.txt` fails to find a matching TensorFlow binary.
- **Resolution:** Verify your Python version using `python --version`. Use a Python 3.10, 3.11, or 3.12 virtual environment. On Python 3.14, stable wheels are not yet published by TensorFlow on PyPI.

### 3. Matplotlib Cache Directory Warning
- **Symptom:** `Matplotlib created a temporary cache directory at ... because C:\Users\... is not writable.`
- **Resolution:** Set the `MPLCONFIGDIR` environment variable before running scripts:
  ```powershell
  $env:MPLCONFIGDIR = "$env:TEMP"
  ```
  *(Note: All project scripts already configure `os.environ.setdefault("MPLCONFIGDIR", ...)` internally).*

### 4. Port Conflict on Streamlit Launch
- **Symptom:** `Port 8501 is already in use.`
- **Resolution:** Specify an alternative port:
  ```powershell
  streamlit run app.py --server.port 8502
  ```

### 5. Missing Model Checkpoint
- **Symptom:** `FileNotFoundError: models/best_model.keras not found.`
- **Resolution:** Run the training script to generate the model artifact:
  ```powershell
  python train.py --epochs_stage1 12 --epochs_stage2 8
  ```

---

## Limitations & Future Work

1. **Dataset Scope:** The current model is trained on 600 authentic PlantVillage images (120 per class). Expanding training to the full multi-thousand image PlantVillage dataset will improve model generalization.
2. **Visual Mimicry Between Diseases:** Early Blight and Bacterial Spot share dark punctate necrosis in their early stages. Incorporating multi-spectral imaging or temporal lesion tracking would reduce inter-class confusion.
3. **In-Field Agricultural Testing:** PlantVillage images feature uniform, controlled laboratory backgrounds. Incorporating noisy, in-field background datasets (such as PlantDoc) will strengthen real-world farm deployment.
4. **Model Quantization:** Future updates could convert the trained Keras model to TensorFlow Lite (`TFLite`) with INT8 quantization for deployment on microcontrollers and offline Android/iOS devices.

---

## Educational Disclaimer

> [!IMPORTANT]
> **LeafLens is an educational and research screening prototype.**
> The predictions, confidence scores, and recommendations generated by this software are provided for educational and academic exploration only. This software does **not** replace laboratory phytopathological analysis, molecular testing, or on-site consultation with certified agricultural extension agents. Never make irreversible chemical application or crop destruction decisions solely based on this screening tool.

---

## License & Acknowledgements

### License
This repository does not currently have an open-source license attached. If you intend to distribute or open-source this codebase, adding a permissive license such as the [MIT License](https://opensource.org/licenses/MIT) or [Apache 2.0 License](https://www.apache.org/licenses/LICENSE-2.0) is recommended.

### Acknowledgements
- **PlantVillage Dataset:** D. P. Hughes and M. Salathé, *"An open access repository of images on plant health to enable the development of mobile disease diagnostics,"* arXiv:1511.08060, 2015. Source repository: [spMohanty/PlantVillage-Dataset](https://github.com/spMohanty/PlantVillage-Dataset).
- **MobileNetV2 Architecture:** M. Sandler, A. Howard, M. Zhu, A. Zhmoginov, and L.-C. Chen, *"MobileNetV2: Inverted Residuals and Linear Bottlenecks,"* CVPR 2018.
- **Grad-CAM:** R. R. Selvaraju, M. Cogswell, A. Das, R. Vedantam, D. Parikh, and D. Batra, *"Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization,"* ICCV 2017.
- **Streamlit:** For providing the interactive Python web application framework.
