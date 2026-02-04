# ♻️ Waste Classification: Transfer Learning & Fine-Tuning Pipeline

![Python](https://img.shields.io/badge/Python-3.x-blue?style=for-the-badge&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=for-the-badge&logo=tensorflow)
![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-red?style=for-the-badge&logo=keras)

**Computer Vision | VGG16 | Domain Adaptation | Fine-Tuning Strategy**

##  Executive Summary
This project implements an automated binary image classification system to categorize waste into **Organic (O)** and **Recyclable (R)** classes. Addressing data scarcity and computational constraints, the solution employs a **Transfer Learning** approach using the **VGG16** architecture pre-trained on ImageNet.

The project differentiates itself through a robust **Two-Stage Training Strategy**:
1.  **Feature Extraction:** Training a custom classification head on frozen features.
2.  **Fine-Tuning:** Unfreezing specific high-level convolutional layers and optimizing them with **SGD and Momentum** to adapt the model to the specific textures of waste materials without catastrophic forgetting.

##  Tech Stack
* **Deep Learning:** TensorFlow (Keras)
* **Backbone:** VGG16 (ImageNet Weights)
* **Data Processing:** `ImageDataGenerator` (Augmentation & Normalization)
* **Utilities:** `zipfile`, `shutil`, `os` for automated data extraction
* **Visualization:** Matplotlib, Seaborn, NumPy

##  System Architecture

### 1. The Backbone (VGG16)

* **Input Shape:** `(224, 224, 3)`
* **Base Configuration:** `include_top=False` (Generic classification head removed).
* **Weights:** Initialized with `imagenet`.

### 2. Custom Classification Head
A custom Deep Neural Network (DNN) was appended to the backbone to map visual features to waste classes:
* **Flatten Layer:** Converts 2D feature maps to 1D vectors.
* **Hidden Layers:** Three stacked `Dense` layers (100 neurons each), activated by `ReLU`.
* **Output Layer:** 1 neuron, `Sigmoid` activation (Binary Classification).

##  Engineering Highlights: The Fine-Tuning Strategy

A critical engineering decision was the implementation of a precise fine-tuning workflow to maximize accuracy while maintaining training stability.

### Phase 1: Transfer Learning (Feature Extraction)
* **State:** The entire VGG16 base is **frozen** (`layer.trainable = False`).
* **Optimizer:** `Adam` (Default learning rate).
* **Objective:** Train *only* the new custom dense layers to interpret the pre-learned generic features of VGG16.

### Phase 2: Surgical Fine-Tuning
* **State:** The base model is set to trainable with specific constraints:
    * **Frozen:** All layers *except* the last 4.
    * **Unfrozen:** The **last 4 layers** of VGG16 are opened for weight updates.
* **Optimizer Strategy:** Switched to **SGD (Stochastic Gradient Descent)** with specific hyperparameters to ensure small, stable updates:
    * `learning_rate`: **0.001** (Low LR to prevent destroying pre-trained weights).
    * `momentum`: **0.9** (To accelerate convergence in relevant directions).

##  Data Pipeline
The data ingestion pipeline handles automated extraction and preprocessing to ensure efficiency:
* **Extraction:** Automatically extracts `waste-classification-data.zip` using `zipfile`.
* **Generators:** Utilizes `ImageDataGenerator` for memory-efficient loading (streaming data from disk rather than loading entirely into RAM).
* **Rescaling:** `1./255` (Pixel Normalization).
* **Target Size:** `(224, 224)` to match VGG16 input requirements.
* **Batch Size:** `32`.
* **Class Mode:** `Binary`.

## 📂 Project Structure
```text
├── waste-classification-data.zip   # Raw compressed dataset
├── Classify_Waste_Products.ipynb   # Main training pipeline
├── output/                         # (Generated) Extracted images
│   ├── TRAIN/                      # Training set (Organic/Recyclable)
│   └── TEST/                       # Testing set (Organic/Recyclable)
└── README.md                       # Documentation
