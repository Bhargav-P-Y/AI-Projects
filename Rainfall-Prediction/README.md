#  Australian Rainfall Prediction: Random Forest vs. Logistic Regression

![Python](https://img.shields.io/badge/Python-3.x-blue?style=for-the-badge&logo=python)
![Scikit-Learn](https://img.shields.io/badge/sklearn-Machine%20Learning-orange?style=for-the-badge&logo=scikit-learn)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Engineering-150458?style=for-the-badge&logo=pandas)
![Status](https://img.shields.io/badge/Status-Verified-success?style=for-the-badge)

**Binary Classification | Grid Search CV | Pipeline Optimization**

##  Executive Summary
This project builds a machine learning pipeline to predict whether it will rain today in the Melbourne area based on historical weather data. The solution implements and compares two classifiers—**Random Forest** and **Logistic Regression**—optimizing both using **Grid Search with Stratified Cross-Validation**.

The project addresses the target leakage issue by redefining the problem to predict "RainToday" using features from the current day, making it applicable for real-time analysis.

##  Tech Stack
* **Core:** Python, Pandas
* **Machine Learning:** Scikit-Learn (`RandomForestClassifier`, `LogisticRegression`)
* **Optimization:** `GridSearchCV`, `StratifiedKFold`
* **Preprocessing:** `ColumnTransformer`, `StandardScaler`, `OneHotEncoder`
* **Visualization:** Matplotlib, Seaborn (`ConfusionMatrixDisplay`)

##  Methodology

### 1. Data Engineering
* **Scope Reduction:** Filtered dataset to three geographically close locations: *Melbourne, Melbourne Airport, and Watsonia* to create a localized model.
* **Feature Engineering:**
    * Created a `Season` feature by mapping the `Date` month to Winter, Spring, Summer, or Autumn.
    * Dropped the original `Date` column to reduce noise.
* **Leakage Prevention:** Renamed targets `RainToday` -> `RainYesterday` and `RainTomorrow` -> `RainToday` to align prediction goals with available data.
* **Preprocessing Pipeline:**
    * **Numerical:** Applied `StandardScaler`.
    * **Categorical:** Applied `OneHotEncoder` (handle_unknown='ignore').

### 2. Model Optimization (Grid Search)
Two distinct architectures were tuned using 5-Fold Stratified Cross-Validation:

#### Model A: Random Forest Classifier
* **Hyperparameters Tuned:**
    * `n_estimators`: [50, 100]
    * `max_depth`: [None, 10, 20]
    * `min_samples_split`: [2, 5]
* **Best Configuration:** `max_depth=None`, `n_estimators=100`, `min_samples_split=2`.

#### Model B: Logistic Regression
* **Hyperparameters Tuned:**
    * `solver`: ['liblinear']
    * `penalty`: ['l1', 'l2'] (Regularization)
    * `class_weight`: [None, 'balanced']

##  Performance & Results

The models were evaluated on an unseen test set using Accuracy, Precision, Recall, and F1-Score.

| Model | Accuracy | Correct Predictions | True Positive Rate |
| :--- | :--- | :--- | :--- |
| **Logistic Regression** | **84%** | **178** | **49.72%** |
| **Random Forest** | **83%** | **182** | **50.83%** |

* **Key Insight:** While Logistic Regression had slightly higher overall accuracy, the Random Forest model achieved a higher True Positive Rate (Sensitivity), making it slightly better at catching actual rain events.
* **Feature Importance:** Analysis of the Random Forest model identified **Humidity** as the single most critical predictor for rainfall.

##  Data Source
The dataset is sourced from the **Australian Government's Bureau of Meteorology** via Kaggle.
* **Dataset:** [Weather Dataset Rattle Package](https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package)

##  Usage

1.  **Install Dependencies**
    ```bash
    pip install pandas numpy matplotlib scikit-learn seaborn
    ```

2.  **Run the Notebook**
    Execute `AUSWeather.ipynb` to:
    * Load and clean the weather data.
    * Run the Grid Search optimization for both models.
    * Generate Classification Reports and Confusion Matrices.
    * Visualize Feature Importance.

---
**Author:** Bhargav P Y
