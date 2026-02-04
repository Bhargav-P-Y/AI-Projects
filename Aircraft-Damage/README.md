#  Aircraft Damage Assessment: Multimodal Classification & Summarization

**Computer Vision | NLP | Transfer Learning | VGG16 | BLIP Transformer**

##  Executive Summary
This project implements an automated pipeline for **Aircraft Damage Detection**, addressing the critical industry need for efficient, error-free safety inspections. The solution utilizes a **multimodal approach**:
1.  **Visual Classification:** Identifying specific damage types ("Dent" vs. "Crack") using a fine-tuned CNN.
2.  **Automated Reporting:** Generating natural language captions and summaries of the damage using a Vision-Language Transformer.

##  Tech Stack
* **Deep Learning:** TensorFlow (Keras), PyTorch
* **Architectures:** VGG16 (Feature Extraction), BLIP (Vision-Language)
* **Libraries:** `transformers`, `pillow`, `matplotlib`, `numpy`, `pandas`
* **Data Processing:** `ImageDataGenerator` (Augmentation), Custom Keras Layers

##  System Architecture

### Part 1: Damage Classification (Computer Vision)
A binary classification engine designed to distinguish between structural dents and cracks.
* **Backbone:** **VGG16** (pre-trained on ImageNet) used as a frozen feature extractor.
* **Custom Head:** The pre-trained classifier was removed and replaced with a custom top architecture:
    * `Flatten` layer to convert 2D feature maps to 1D vectors.
    * `Dense` (512 units, ReLU) + `Dropout` (0.3) for regularization.
    * `Dense` (512 units, ReLU) + `Dropout` (0.3).
    * `Dense` (1 unit, Sigmoid) for binary probability output.
* **Optimization:** Trained with `Adam` optimizer (lr=0.0001) and `Binary Crossentropy` loss.

### Part 2: Captioning & Summarization (Generative AI)
Integrated the **BLIP (Bootstrapping Language-Image Pretraining)** model to generate human-readable damage descriptions.
* **Model:** `Salesforce/blip-image-captioning-base`.
* **Innovation:** Implemented a **Custom Keras Layer (`BlipCaptionSummaryLayer`)** to wrap the PyTorch-based BLIP model within a TensorFlow workflow. This allows for seamless integration of the transformer into the Keras functional API.
* **Capabilities:** Generates both short captions and detailed summaries based on task prompts ("caption" vs "summary").

##  Technical Implementation Highlights

### 1. Data Pipeline & Preprocessing
* **Directory Structure:** Organized data into `train`, `valid`, and `test` directories, each containing subfolders for `dent` and `crack` classes.
* **Augmentation:** Utilized `ImageDataGenerator` to rescale pixel values (`1./255`) for normalization.
* **Generators:** Implemented memory-efficient data loading using `flow_from_directory`:
    * `train_generator`: Shuffled, batch size 32.
    * `valid_generator` & `test_generator`: Non-shuffled for consistent evaluation.
* **Input Shape:** All images resized to `224x224` to match VGG16 requirements.

### 2. Custom Keras Layer for BLIP
I engineered a custom layer to bridge the BLIP model with the Keras ecosystem, enabling task-specific text generation.

##  Performance & Results
* **Classification:** Achieved **92%** accuracy on the test set.
* **Generative Output:**
    * *Caption:* "A close-up of a dent in the fuselage."
    * *Summary:* "This image shows minor structural damage characterized by a shallow dent on the aircraft's exterior panel."

##  Data Source
The model was trained on the **Aircraft Damage Dataset**, containing labeled images of dents and cracks.
* **Source:** [Roboflow Aircraft Dataset](https://universe.roboflow.com/youssef-donia-fhktl/aircraft-damage-detection-1j9qk)
* **License:** CC BY 4.0

##  Usage
1.  **Clone the Repository**
2.  **Install Dependencies**
    ```bash
    pip install tensorflow torch transformers pillow matplotlib
    ```
3.  **Run the Notebook**
    Open `Aircraft_Damage_Classification.ipynb` to execute the full pipeline, from data download to inference.

