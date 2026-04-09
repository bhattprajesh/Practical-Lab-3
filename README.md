# 🐶🐱 Dogs vs Cats — Transfer Learning & Fine-Tuning Lab

---

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Running the Notebook](#running-the-notebook)
- [Results Summary](#results-summary)
- [Key Talking Points](#key-talking-points)
- [Team](#team)

---

## Overview

This lab demonstrates the core deep learning engineering workflow of **transfer learning and fine-tuning**:

1. **Custom CNN** — A 4-block convolutional baseline trained from scratch on ~3,500 images
2. **VGG16 Fine-Tuned** — A two-phase approach: feature extraction (frozen base) followed by selective unfreezing of the last two convolutional blocks

Both models are evaluated on a held-out test set using accuracy, confusion matrix, precision, recall, F1-score, precision-recall curve, and failure case analysis.

---

## Dataset

| Property | Detail |
|---|---|
| **Source** | [Microsoft Download Center](https://www.microsoft.com/en-us/download/details.aspx?id=54765) |
| **Full archive** | `kagglecatsanddogs_5340.zip` (~786 MB, ~25,000 images) |
| **Subset used** | **5,000 images** — 2,500 cats + 2,500 dogs (sampled with `seed=42`) |
| **Split** | 70 % train (3,500) · 15 % validation (750) · 15 % test (750) |
| **Classes** | `cats` · `dogs` |

> ⚠️ The dataset is **not committed to this repository** (too large). The notebook downloads and prepares it automatically on first run. Corrupted JPEG files in the raw archive are detected and skipped via `PIL.Image.verify()`.

---

## Project Structure

```
dogs-vs-cats-finetune/
│
├── dogs_vs_cats_finetune.ipynb   # Main notebook (all code + Markdown)
├── requirements.txt               # Python dependencies
├── .gitignore                     # Excludes dataset, models, plots
└── README.md                      # This file
```

Files generated at runtime (excluded from Git):

```
PetImages/                         # Raw extracted archive
cats_and_dogs_5k/                  # Structured 5k-image dataset
│   ├── train/cats/  train/dogs/
│   ├── validation/cats/  validation/dogs/
│   └── test/cats/  test/dogs/
best_cnn.keras                     # Best Custom CNN checkpoint
best_vgg.keras                     # Best VGG16 checkpoint
*.png                              # All generated plots
```

---

## Setup & Installation

### Prerequisites
- Python **3.9 – 3.13.12**
- pip ≥ 23
- (Optional) NVIDIA GPU with CUDA for faster training

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/dogs-vs-cats-finetune.git
cd dogs-vs-cats-finetune
```

### 2. Create a virtual environment

```bash
python -m venv venv
# macOS / Linux
source venv/bin/activate
# Windows
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **GPU users:** Replace `tensorflow` with `tensorflow[and-cuda]` in `requirements.txt` before installing.

### 4. Launch Jupyter

```bash
jupyter notebook
```

Then open `dogs_vs_cats_finetune.ipynb`.

---

## Running the Notebook

Run all cells **top to bottom**. The notebook is fully self-contained:

| Section | What happens |
|---|---|
| **1. Setup** | Imports libraries, sets seeds |
| **2. Data Acquisition** | Downloads the Microsoft zip, verifies images, samples 5,000, builds train/val/test folders |
| **3. EDA** | Plots class balance, sample images, size distribution, channel intensities, aspect ratios |
| **4a. Custom CNN** | Trains a 4-block CNN for up to 40 epochs with callbacks; saves `best_cnn.keras` |
| **4b. VGG16** | Phase 1 (head only, 20 epochs) + Phase 2 (unfreeze blocks 4–5, 30 epochs); saves `best_vgg.keras` |
| **5. Evaluation** | Loads best checkpoints; computes all metrics + plots |
| **6. Failure Analysis** | Visualises most-confident wrong predictions; finds images both models fail on |
| **7. Conclusions** | Summary table (accuracy, F1, MCC, AP) + written analysis |

> ⏱️ **Estimated runtime:** ~25–40 min on CPU · ~8–12 min on GPU

---

## Results Summary

> *Exact numbers will vary slightly due to random sampling and augmentation.  
> The values below are representative of a typical run.*

| Model | Test Accuracy | Macro F1 | Avg Precision |
|---|---|---|---|
| Custom CNN | ~0.76 | ~0.76 | ~0.83 |
| VGG16 Fine-Tuned | ~0.92 | ~0.92 | ~0.97 |

VGG16 fine-tuned decisively outperforms the custom CNN across every metric, confirming the value of transfer learning on small datasets.

---

## Key Talking Points

Five 💡 **Talking Points** are embedded as highlighted Markdown blocks throughout the notebook:

1. **Motivation for Transfer Learning** *(Introduction)* — Why reusing ImageNet weights beats training from scratch on small datasets
2. **Dataset Design & Sampling Strategy** *(Data Acquisition)* — Corrupt-file filtering, reproducible sampling, and leakage-free splits
3. **Custom CNN Architecture** *(Section 4a)* — Conv-BN-ReLU-Pool design rationale, GlobalAveragePooling, and Dropout
4. **Two-Phase Fine-Tuning Strategy** *(Section 4b)* — Why we freeze first, which VGG16 layers to unfreeze, and why the learning rate must drop to 1e-5
5. **Evaluation Protocol** *(Section 5)* — Always load the best checkpoint; evaluate only on the unseen test set

---



## References

- Chollet, F. (2021). *Deep Learning with Python* (2nd ed.). Manning Publications.
- Simonyan, K., & Zisserman, A. (2015). *Very Deep Convolutional Networks for Large-Scale Image Recognition*. ICLR 2015.
- Microsoft Download Center — [Kaggle Cats and Dogs Dataset](https://www.microsoft.com/en-us/download/details.aspx?id=54765)
- TensorFlow / Keras Documentation — [Transfer Learning Guide](https://www.tensorflow.org/guide/keras/transfer_learning)
