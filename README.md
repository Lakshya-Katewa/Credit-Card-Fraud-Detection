# Credit Card Fraud Detection

A machine learning project for detecting fraudulent credit card transactions using classification models, imbalance handling, cross-validation, threshold tuning, and model evaluation.

The project uses **284K+ real-world credit card transactions** and focuses on optimizing fraud detection performance under severe class imbalance, where correctly identifying fraudulent transactions is more important than simply maximizing accuracy.

## Overview

Credit card fraud detection is a highly imbalanced binary classification problem. In a typical transaction dataset, legitimate transactions significantly outnumber fraudulent ones.

This project explores multiple machine learning approaches and compares their performance using metrics that are more suitable for imbalanced classification.

The main goals are:

* Build a leakage-safe fraud detection pipeline
* Handle class imbalance appropriately
* Compare multiple classification algorithms
* Evaluate models using precision, recall, F1-score, ROC-AUC, and PR-AUC
* Tune the classification threshold based on validation performance
* Analyze false positives and false negatives
* Interpret model behavior using feature importance

## Dataset

The project uses a credit card transaction dataset containing **284K+ transactions**.

The target variable represents whether a transaction is fraudulent:

* `0` → Legitimate transaction
* `1` → Fraudulent transaction

Because fraudulent transactions represent only a small fraction of the dataset, standard accuracy is not sufficient for evaluating model performance.

## Machine Learning Models

The following classification models were evaluated:

* Logistic Regression
* Decision Tree
* K-Nearest Neighbors (KNN)
* Random Forest

These models provide a useful comparison between linear, distance-based, tree-based, and ensemble approaches.

## Methodology

### 1. Data Preprocessing

The dataset was first inspected for:

* Missing values
* Class distribution
* Feature distributions
* Duplicate or inconsistent records

Features were prepared for model training while ensuring that information from validation data did not leak into the training process.

### 2. Handling Class Imbalance

Fraudulent transactions are heavily underrepresented compared with legitimate transactions.

Instead of relying on accuracy, the project focuses on metrics that better capture performance on the minority fraud class.

Stratified validation was used to preserve the proportion of fraudulent and legitimate transactions across folds.

### 3. Model Training

Multiple classification algorithms were trained and evaluated under the same validation framework.

This allowed their performance to be compared consistently rather than evaluating each model using different conditions.

### 4. Model Evaluation

The models were evaluated using:

| Metric    | Purpose                                                          |
| --------- | ---------------------------------------------------------------- |
| Precision | Measures how many predicted fraud cases were actually fraudulent |
| Recall    | Measures how many actual fraud cases were detected               |
| F1-Score  | Balances precision and recall                                    |
| ROC-AUC   | Measures overall ranking performance across thresholds           |
| PR-AUC    | Evaluates performance under class imbalance                      |

For fraud detection, **precision and recall are particularly important** because both false positives and false negatives have practical consequences.

### 5. Threshold Tuning

Most binary classification models use a default probability threshold of `0.5` to classify transactions as positive or negative.

Instead of automatically using this threshold, the project evaluated different thresholds on validation data to find a better precision-recall tradeoff.

The selected threshold achieved:

* **87% Precision**
* **74% Recall**

This means the final model was able to maintain relatively high precision while still identifying a substantial portion of fraudulent transactions.

## Results

The models were compared using multiple evaluation metrics rather than relying on a single score.

The final threshold-tuned model achieved:

| Metric    |  Result |
| --------- | ------: |
| Precision | **87%** |
| Recall    | **74%** |

The results demonstrate the importance of threshold selection in imbalanced classification problems.

A higher threshold can reduce false positives but may miss more fraudulent transactions, while a lower threshold can detect more fraud at the cost of generating additional false alerts.

## Error Analysis

Confusion matrices were used to analyze model predictions in terms of:

* True Positives (TP)
* True Negatives (TN)
* False Positives (FP)
* False Negatives (FN)

For fraud detection:

* **False Positive:** A legitimate transaction incorrectly flagged as fraud
* **False Negative:** A fraudulent transaction incorrectly classified as legitimate

The project examines these errors to understand the practical tradeoff between catching more fraud and avoiding unnecessary transaction alerts.

## Feature Importance

Feature importance was analyzed for tree-based models to understand which input variables contributed most to the model's predictions.

This provides additional interpretability beyond simply measuring predictive performance.

## Project Workflow

```text
Raw Transaction Data
        │
        ▼
Data Exploration & Cleaning
        │
        ▼
Train / Validation Split
        │
        ▼
Preprocessing
        │
        ▼
Stratified Cross-Validation
        │
        ▼
Model Training
        │
        ├── Logistic Regression
        ├── Decision Tree
        ├── KNN
        └── Random Forest
        │
        ▼
Model Evaluation
        │
        ├── Precision
        ├── Recall
        ├── F1
        ├── ROC-AUC
        └── PR-AUC
        │
        ▼
Threshold Tuning
        │
        ▼
Final Model
        │
        ▼
Confusion Matrix & Feature Analysis
```

## Tech Stack

* **Python**
* **Pandas** — Data manipulation
* **NumPy** — Numerical computation
* **Scikit-learn** — Machine learning and evaluation
* **Matplotlib / Seaborn** — Data visualization
* **Jupyter Notebook** — Experimentation and analysis

## Repository Structure

```text
credit-card-fraud-detection/
│
├── data/
│   └── README.md
│
├── notebooks/
│   └── fraud_detection.ipynb
│
├── README.md
├── requirements.txt
└── .gitignore
```

> The dataset itself may not be included in the repository due to size or distribution restrictions.

## Installation

Clone the repository:

```bash
git clone <your-repository-url>
cd credit-card-fraud-detection
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it:

### Windows

```bash
venv\Scripts\activate
```

### macOS / Linux

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Running the Project

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Open the notebook inside the `notebooks/` directory and run the cells sequentially.

## Key Takeaways

This project demonstrates practical machine learning concepts including:

* Binary classification
* Imbalanced dataset handling
* Stratified cross-validation
* Leakage prevention
* Model comparison
* Precision-recall analysis
* Classification threshold tuning
* Confusion matrix analysis
* Feature importance
* Model interpretation

The project also highlights an important practical lesson in fraud detection: **the best model is not necessarily the one with the highest accuracy, but the one that provides an appropriate tradeoff between detecting fraud and minimizing false alerts.**

## Future Improvements

Potential improvements include:

* Hyperparameter optimization using GridSearchCV or RandomizedSearchCV
* Testing additional ensemble models
* Calibrating predicted probabilities
* Evaluating performance on a time-based validation split
* Building a real-time fraud prediction API
* Monitoring model performance for data and concept drift

## Author

**Lakshya Katewa**
IIT Kanpur

---

⭐ If you found this project useful, consider giving the repository a star.
