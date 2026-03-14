# Facial Emotion Recognition using ResNet18

This project implements a deep learning model to classify human facial expressions into 7 different emotions. It utilizes the **ResNet18** architecture via **Transfer Learning** to achieve high accuracy on the Facial Emotion Recognition dataset.

## 📌 Project Overview
The goal of this project is to recognize emotions from grayscale facial images. The model is trained to identify the following categories:
* Angry
* Disgust
* Fear
* Happy
* Neutral
* Sad
* Surprise

## 🚀 Features
* **Transfer Learning:** Uses pre-trained ResNet18 weights from ImageNet for faster convergence and better feature extraction.
* **Data Augmentation:** Implements random horizontal flips and rotations to improve model generalization.
* **Custom Head:** The original ResNet fully connected layer is replaced with a custom MLP head tailored for 7-class classification.
* **Visualization:** Includes a grid-based visualization tool to display model predictions on local test images.

## 🛠️ Tech Stack
* **Language:** Python
* **Framework:** PyTorch
* **Library:** Torchvision (for models and transforms)
* **Data Handling:** KaggleHub (for automatic dataset download)
* **Visualization:** Matplotlib, PIL

## 📂 Project Structure
* `Yolo.ipynb`: The main Jupyter notebook containing data loading, model definition, training loop, and evaluation.
* `processed_data/`: The directory where the dataset is stored and organized by emotion categories.

## ⚙️ How It Works

### 1. Data Preprocessing
Images are resized to **64x64** pixels and converted to **Grayscale**. We apply normalization to ensure pixel values are within the range of [-1, 1], which helps in stabilizing the training process.

### 2. Model Architecture
We modify the standard ResNet18:
* **Input Layer:** Changed to accept 1-channel (Grayscale) input instead of the default 3-channel (RGB).
* **Fully Connected Layer:** A custom sequence of Linear -> ReLU -> Dropout -> Linear layers is added to map the features to the 7 emotion classes.

### 3. Training
* **Optimizer:** Adam optimizer with a learning rate of `0.0001`.
* **Loss Function:** CrossEntropyLoss.
* **Epochs:** 10 (adjustable based on performance).

## 📊 Usage

### Prerequisites
Install the required libraries:
```bash
pip install torch torchvision matplotlib kagglehub pillow

