#  Amazon ML Challenge 2025: Multimodal Price Prediction
![Python](https://img.shields.io/badge/Python-3.x-blue?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red?style=for-the-badge&logo=pytorch)
![LightGBM](https://img.shields.io/badge/LightGBM-Gradient%20Boosting-green?style=for-the-badge)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow?style=for-the-badge)
![Status](https://img.shields.io/badge/Rank-Top%2012%25-brightgreen?style=for-the-badge)

**Score:** 52.44% SMAPE (Symmetric Mean Absolute Percentage Error)

##  Executive Summary
This project is an end-to-end Machine Learning pipeline designed to predict the price of retail items based on a multimodal dataset (Images + Text + Tabular Metadata). By leveraging a fusion of **Computer Vision (ResNet50)**, **NLP (Sentence Transformers)**, and **Gradient Boosting (LightGBM)**, the solution effectively handles high-dimensional, noisy e-commerce data.

The model ranked in the **Top 12% nationwide**, demonstrating robust handling of missing data and advanced feature engineering techniques.

##  Tech Stack & Tools
* **Core:** Python, Pandas, NumPy, Scikit-Learn
* **Natural Language Processing:** `sentence-transformers` (HuggingFace)
* **Computer Vision:** `torchvision` (ResNet50), `Pillow`
* **Modeling:** LightGBM, XGBoost
* **Data Engineering:** Regex (Regular Expressions), TQDM, Requests

##  Methodological Approach

### 1. Advanced Feature Engineering
We transformed raw, unstructured data into high-signal numerical features:
* **Regex-Based Extraction:** Engineered custom parsers to extract **Item Pack Quantity (IPQ)** (e.g., "Pack of 6" $\rightarrow$ `6`, "12 count" $\rightarrow$ `12`) and normalized diverse measurement units (e.g., converting `lbs`, `ounces`, `kg` to standard forms).
* **Leakage-Free Target Encoding:** Implemented K-Fold Target Encoding for high-cardinality categorical variables (like `brand` or `item_name`) to capture category-level price trends without overfitting.
* **Log-Transformation:** Applied `np.log1p` to the target variable (`price`) and skewed features to normalize distributions and stabilize gradient descent.

### 2. Multimodal Embedding Pipeline
To capture semantic nuance, I engineered a dual-stream embedding architecture:
* **Text (Ensemble Strategy):** Concatenated embeddings from two distinct transformer models to balance inference speed with semantic depth:
    * `all-MiniLM-L6-v2` (384-dim): Captures broad semantic similarity.
    * `all-mpnet-base-v2` (768-dim): Captures detailed contextual nuance.
* **Images (Transfer Learning):** Utilized a **ResNet50** backbone (pre-trained on ImageNet). I removed the classification head to extract a rich **2048-dimensional** feature vector from the Global Average Pooling layer.
* **Robust Error Handling:** Implemented a binary `image_missing` flag for corrupted or failed downloads, converting a data quality issue into a predictive signal.

### 3. Modeling Strategy
* **Algorithm:** LightGBM Regressor (Gradient Boosting Decision Tree).
* **Objective Function:** `regression_l1` (Mean Absolute Error) was selected as a proxy to optimize for the competition's SMAPE metric.
* **Validation:** Rigorous **5-Fold Cross-Validation** to ensure model stability and generalizability across unseen data.

### 4. Post-Processing & Inference
* **Inverse Transformation:** Predictions were transformed back to the original scale using `np.expm1` to reverse the log-normalization.
* **Constraint Enforcement:** Applied a non-negative constraint (`pred[pred < 0] = 0`) to ensure logical pricing outputs.

