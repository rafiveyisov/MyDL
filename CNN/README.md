<div align="center">

# 🧠 Computer Vision & Deep Learning

**A collection of CNN-based projects spanning image classification, character recognition, and beyond.**

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-22C55E?style=flat-square)](LICENSE)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Project Index](#-project-index)
- [Project Details](#-project-details)
  - [Project 1 — Facial Emotion Recognition](#-project-1--facial-emotion-recognition-resnet18)
  - [Project 2 — EMNIST Alphanumeric Recognizer](#-project-2--emnist-alphanumeric-recognizer)
- [Tech Stack](#️-tech-stack)
- [Getting Started](#-getting-started)
- [Author](#-author)

---

## 🔭 Overview

This repository is a growing collection of deep learning projects built around **Convolutional Neural Networks (CNNs)**. Each project lives in its own self-contained folder with dedicated source code, models, and instructions. New projects are added over time — see the index below for the current list.

---

## 📁 Project Index

| # | Project | Domain | Architecture | Accuracy | Status |
|---|---------|--------|-------------|----------|--------|
| 01 | [Facial Emotion Recognition](#-project-1--facial-emotion-recognition-resnet18) | Image Classification | ResNet18 (Transfer Learning) | ~72–75% | ✅ Complete |
| 02 | [EMNIST Alphanumeric Recognizer](#-project-2--emnist-alphanumeric-recognizer) | Character Recognition | LeNet-5 + Flask | ~83% | ✅ Complete |
| 03 | *(Coming Soon)* | — | — | — | 🔄 Planned |

> **Adding a new project?** Create a new folder, add its entry to this table, and add a section below following the existing template.

---

## 📂 Project Details

---

### 🎭 Project 1 — Facial Emotion Recognition (ResNet18)

> Classifies human facial expressions into 7 emotional categories using Transfer Learning on ResNet18.

**📁 Folder:** `FacialExpression/`

#### Key Features

| Feature | Detail |
|---------|--------|
| Architecture | ResNet18 fine-tuned for grayscale input |
| Output Classes | Angry · Disgust · Fear · Happy · Neutral · Sad · Surprise |
| Accuracy | ~72–75% |
| Extras | Batch prediction grid visualization |

#### Model Pipeline

```
Input (64×64×1)
    └── ResNet18 Backbone (Pre-trained ImageNet weights)
        └── Modified first conv layer (1-channel input)
            └── Custom Fully Connected Head
                └── Output: 7 Classes (Softmax)
```

#### Quick Start

```bash
cd FacialExpression/
jupyter notebook Yolo.ipynb
```

---

### 🔤 Project 2 — EMNIST Alphanumeric Recognizer

> A real-time web application that recognizes handwritten digits and letters drawn on a canvas.

**📁 Folder:** `Emnist/`

#### Key Features

| Feature | Detail |
|---------|--------|
| Architecture | LeNet-5 based CNN |
| Output Classes | 62 characters (0–9, A–Z, a–z) |
| Accuracy | ~83% |
| Interface | Flask web app with interactive drawing canvas |

#### Model Pipeline

```
Input (28×28×1)
    └── Conv2D + ReLU
        └── MaxPool
            └── Conv2D + ReLU
                └── MaxPool
                    └── Flatten
                        └── Dense (500 units)
                            └── Output: 62 Classes (Softmax)
```

#### Quick Start

```bash
cd Emnist/
python app.py
# Open http://localhost:5000 in your browser
```

---

<!--
════════════════════════════════════════════════
  TEMPLATE FOR FUTURE PROJECTS — Copy & fill in
════════════════════════════════════════════════

### 🔷 Project N — [Project Name]

> One-line description of what this project does.

**📁 Folder:** `ProjectFolder/`

#### Key Features

| Feature | Detail |
|---------|--------|
| Architecture | [e.g., VGG16 / EfficientNet / Custom CNN] |
| Output Classes | [e.g., 10 / 100 / binary] |
| Accuracy | ~XX% |
| Extras | [Any extra features] |

#### Model Pipeline

```
Input (H×W×C)
    └── [Layer 1]
        └── [Layer 2]
            └── Output: N Classes
```

#### Quick Start

```bash
cd ProjectFolder/
[run command]
```

---
-->

## 🛠️ Tech Stack

| Category | Libraries / Tools |
|----------|------------------|
| Deep Learning | PyTorch, TensorFlow / Keras |
| Web Framework | Flask |
| Data & Utilities | NumPy, Pillow, Matplotlib, KaggleHub |
| Environment | Python 3.8+, `venv` or `conda` |
| Notebooks | Jupyter Notebook |

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/rafiveyisov/Emotion-Recognition-ResNet.git
cd Emotion-Recognition-ResNet
```

### 2. Set up your environment

```bash
# Option A — venv
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Option B — conda
conda create -n cv-projects python=3.9
conda activate cv-projects
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run a project

Navigate to the relevant project folder and follow its **Quick Start** instructions listed above.

---

## 👤 Author

<div align="center">

**Rafi Veyisov**

[![GitHub](https://img.shields.io/badge/GitHub-@rafiveyisov-181717?style=flat-square&logo=github)](https://github.com/rafiveyisov)

*Deep Learning · Computer Vision · CNN*

</div>

---

<div align="center">
<sub>⭐ If you find this repository useful, consider giving it a star.</sub>
</div>
