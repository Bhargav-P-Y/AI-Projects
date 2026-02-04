#  Amazon ML Challenge 2025: Multimodal Price Prediction
**Ranked Top 12% Nationwide** | **SMAPE Score: 52.44%**

##  Overview
This repository contains the solution for the Amazon ML Challenge 2025. The objective was to predict the price of retail items using a multimodal dataset comprising product images, textual descriptions, and metadata.

The solution leverages an ensemble of **Sentence Transformers (NLP)**, **ResNet50 (CV)**, and **LightGBM** to handle noisy, real-world e-commerce data.

##  Tech Stack
* **Core:** Python, Pandas, NumPy, Scikit-Learn
* **NLP:** HuggingFace Transformers (`sentence-transformers`)
* **Computer Vision:** PyTorch (`torchvision`)
* **Modeling:** LightGBM, XGBoost
* **Utilities:** Regex, Pillow, TQDM

##  Methodology

### 1. Advanced Feature Engineering
We moved beyond raw data by engineering domain-specific features to capture value drivers:
* **Regex-Based Extraction:** Developed custom parsers to extract **Item Pack Quantity (IPQ)** (e.g., "Pack of 6" $\rightarrow$ `6`) and normalize non-standard units (e.g., `lbs` $\rightarrow$ `Pound`).
* **Target Encoding:** Implemented leakage-free Target Encoding for high-cardinality categorical variables (`item_name`).
* **Distribution Normalization:** Applied `np.log1p` followed by `QuantileTransformer` to normalize highly skewed features like Price and IPQ.

### 2. Multimodal Embedding Pipeline
* **Text (Ensemble Strategy):** Concatenated embeddings from two distinct transformer models to capture both speed and semantic depth:
    * `all-MiniLM-L6-v2` (384-dim)
    * `all-mpnet-base-v2` (768-dim)
* **Images (Transfer Learning):** Extracted 2048-dimensional feature vectors using **ResNet50** (pre-trained on ImageNet), removing the final classification head.
* **Handling Missing Data:** Implemented a binary `image_missing` flag for failed downloads, which proved to be a predictive signal.

### 3. Modeling & Evaluation
* **Algorithm:** LightGBM Regressor with `regression_l1` objective.
* **Validation Strategy:** 5-Fold Cross-Validation to ensure stability and prevent overfitting.
* **Metric:** Optimized for **SMAPE** (Symmetric Mean Absolute Percentage Error).
* **Performance:** Achieved a consistent validation SMAPE of **~52.44%** across folds.
