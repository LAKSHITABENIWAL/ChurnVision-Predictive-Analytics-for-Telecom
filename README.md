# ChurnVision: Predictive Analytics for Telecom

## Overview

This project focuses on predicting customer churn for a telecom company using a real-world dataset of over 3,300 customers. The goal is to identify customers likely to leave the service, enabling targeted retention strategies. The workflow covers data cleaning, feature engineering, advanced encoding, class balancing, model selection, and pipeline automation.

---

## Table of Contents

- [Project Objective](#project-objective)
- [Dataset](#dataset)
- [Key Features](#key-features)
- [Workflow](#workflow)
- [Modeling Approach](#modeling-approach)
- [Results](#results)
- [How to Run](#how-to-run)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Acknowledgements](#acknowledgements)
- [STAR Format for Resume](#star-format-for-resume)

---

## Project Objective

**Classification:** Predict whether a customer will churn based on their usage, plan, and demographic data.

---

## Dataset

- **Source:** Public telecom customer churn dataset  
- **Records:** 3,333 customers  
- **Features:** Usage metrics, plan details, demographics, customer service interactions  
- **Target:** `churn` (binary: True/False)

---

## Key Features

- **Automated Data Cleaning:** Handled outliers, corrected data types, and dropped irrelevant columns.
- **Advanced Feature Engineering:** Binned and transformed skewed features, created new ratio-based features, and handled multicollinearity.
- **Encoding:** Applied frequency, target, and direct mapping encodings to categorical variables.
- **Class Balancing:** Used ADASYN to address class imbalance.
- **Model Comparison:** Evaluated multiple classifiers (Logistic Regression, Random Forest, SVM, XGBoost, LightGBM, etc.) using cross-validation.
- **Pipeline Automation:** Built modular preprocessing and modeling pipelines for reproducibility.

---

## Workflow

1. **Data Inspection:** Checked for missing values, duplicates, outliers, and class imbalance.
2. **Preprocessing:** Corrected data types, dropped uninformative columns, and handled outliers.
3. **Feature Engineering:** Applied binning, log transformations, and created charge-per-minute features.
4. **Encoding:** Used frequency, target, and direct mapping for categorical variables.
5. **Scaling:** Applied RobustScaler to numerical features.
6. **Feature Selection:** Removed highly correlated and low-variance features.
7. **Class Balancing:** Used ADASYN for oversampling the minority class.
8. **Model Training:** Compared several classifiers with cross-validation.
9. **Hyperparameter Tuning:** Used RandomizedSearchCV for LightGBM.
10. **Evaluation:** Assessed models using accuracy, precision, recall, F1-score, and ROC AUC.

---

## Modeling Approach

- **Pipelines:** Used scikit-learn pipelines and custom transformers for modular preprocessing.
- **Best Model:** LightGBM with hyperparameter tuning.
- **Feature Importance:** Leveraged Random Forest for feature selection.
- **Class Balancing:** ADASYN for oversampling.

---

## Results

| Model           | Test Accuracy | Test Precision | Test Recall | Test F1 | Test ROC AUC |
|-----------------|--------------|---------------|-------------|---------|--------------|
| LightGBM        | 0.94         | 0.88          | 0.67        | 0.76    | 0.90         |
| XGBoost         | 0.93         | 0.77          | 0.69        | 0.73    | 0.88         |
| RandomForest    | 0.90         | 0.71          | 0.58        | 0.64    | 0.87         |

---

## How to Run

1. **Clone the repository:**
    ```bash
    git clone <your-repo-link>
    cd Customer chum prediction
    ```

2. **Install requirements:**
    ```bash
    pip install -r requirements.txt
    ```

3. **Run the notebook:**
    - Open `main.ipynb` in Jupyter Notebook or VS Code and run all cells.

---

## Project Structure

```
Customer chum prediction/
│
├── main.ipynb
├── requirements.txt
└── README.md
```

---

## Requirements

- Python 3.8+
- pandas
- numpy
- scikit-learn
- xgboost
- lightgbm
- imbalanced-learn
- seaborn
- matplotlib

Install all dependencies with:
```bash
pip install -r requirements.txt
```

---

## Acknowledgements

- Dataset sourced from public telecom churn datasets.
- Inspired by best practices in end-to-end machine learning and data science workflows.

---

## STAR Format for Resume

**Situation:**  
Analyzed a telecom dataset of 3,300+ customers to identify churn risk and improve retention strategies.

**Task:**  
Develop an end-to-end machine learning solution to predict customer churn using advanced feature engineering and model selection.

**Action:**  
- Automated data cleaning, encoding, and class balancing with modular pipelines.
- Engineered features, handled outliers, and evaluated multiple models (LightGBM, XGBoost, Random Forest) with cross-validation and hyperparameter tuning.

**Result:**  
Achieved 94% accuracy, 0.76 F1-score, and 0.90 ROC AUC on test data, enabling targeted retention actions and demonstrating strong skills in Python, scikit-learn, and ML automation.
