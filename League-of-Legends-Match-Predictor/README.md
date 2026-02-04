#  League of Legends Match Predictor: PyTorch Classification Pipeline

![Python](https://img.shields.io/badge/Python-3.x-blue?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red?style=for-the-badge&logo=pytorch)
![Scikit-Learn](https://img.shields.io/badge/sklearn-Data%20Science-orange?style=for-the-badge&logo=scikit-learn)
![Status](https://img.shields.io/badge/Status-Baseline%20Established-success?style=for-the-badge)

**Binary Classification | Logistic Regression | Hyperparameter Tuning | Model Interpretability**

##  Executive Summary
This project establishes a rigorous **Deep Learning workflow** to predict the outcome of *League of Legends* matches (Win/Loss) based on in-game telemetry data. Unlike simple script-kiddie implementations, this project manually constructs the training pipeline using **PyTorch**, demonstrating a deep understanding of tensor operations, gradient descent, and model serialization.

The solution focuses on **Explainable AI (XAI)** by extracting feature importance weights to understand *why* a match was won, identifying "Gold Earned" as the primary predictive factor.

##  Tech Stack
* **Core Framework:** PyTorch (`torch`, `torch.nn`, `torch.optim`)
* **Data Processing:** Pandas, NumPy, Scikit-Learn (`StandardScaler`)
* **Visualization:** Matplotlib (ROC Curves, Feature Importance Bars)
* **Optimization:** SGD (Stochastic Gradient Descent) with L2 Regularization

##  System Architecture

### 1. Model Definition (`nn.Module`)
A custom PyTorch class was architected to handle the binary classification task:
* **Input Layer:** Dynamic input dimension based on feature set size (8 features).
* **Linear Transformation:** `nn.Linear` maps inputs to a single output logit.
* **Activation:** `torch.sigmoid` forces output between [0, 1] for probability estimation.
* **Loss Function:** `BCELoss` (Binary Cross Entropy) utilized for optimization.

### 2. The Training Pipeline (Manual Implementation)
Instead of using high-level wrappers, the training loop is implemented manually to demonstrate control over the learning process:
* **Batch Processing:** Utilized `DataLoader` and `TensorDataset` for efficient batching (Batch Size: 32).
* **Gradient Management:** Manual calls to `optimizer.zero_grad()` and `loss.backward()` to handle backpropagation.
* **Regularization:** Implemented **L2 Regularization (Weight Decay = 0.01)** within the SGD optimizer to penalize large weights and prevent overfitting.

### 3. Hyperparameter Tuning
A systematic search was conducted to optimize model convergence:
* **Grid Search:** Iterated through learning rates `[0.01, 0.05, 0.1]`.
* **Validation:** Monitored accuracy across epochs to select the optimal rate (`0.01`), ensuring the model reached a stable local minimum.

##  Engineering Highlights

### Feature Importance Extraction
To ensure the model isn't a "black box," I extracted the weights from the linear layer after training to rank features by influence.
* **Top Predictor:** `gold_earned` (Importance: ~0.17)
* **Secondary Predictor:** `kills` (Importance: ~0.12)
* **Insight:** Economic advantage (Gold) is a stronger predictor of victory than raw combat stats (Kills).

### MLOps: Model Serialization
Implemented a production-ready saving mechanism:
* **State Dictionary:** Saved model weights using `model.state_dict()` rather than pickling the entire object, ensuring architectural agnostic loading.
* **Inference Check:** Implemented a loading verification step to ensure the reloaded model performs identical inference on test data.

##  Performance & Evaluation
The model serves as a robust baseline for match prediction.
* **Metric:** Accuracy, Precision, Recall, F1-Score.
* **Visuals:**
    * **ROC Curve:** Plotted True Positive vs. False Positive rates.
    * **Confusion Matrix:** Analyzed Type I vs. Type II errors.
* **Baseline Accuracy:** ~51% (Note: This indicates the need for more complex architectures like Neural Networks or XGBoost, which can be easily swapped into this established pipeline).

##  Usage

1.  **Install Dependencies**
    ```bash
    pip install torch torchvision pandas scikit-learn matplotlib
    ```

2.  **Run the Pipeline**
    The notebook executes the full lifecycle:
    * **Preprocessing:** Scales features using `StandardScaler`.
    * **Training:** Runs SGD for 1000 epochs.
    * **Tuning:** Compares Learning Rates.
    * **Analysis:** Plots Feature Importance.

---
**Author:** Bhargav P Y
