<h1 align="center">ML Fundamentals</h1>

This repository contains introductory machine learning materials organized by topic. The examples use synthetic data, public datasets bundled with `scikit-learn`, and small public datasets with embedded fallbacks, so the current notebooks and app can run without external data downloads.

# Table of Contents

- [Topics](#topics)
  - [Topic 1. Basics](#topic-1-basics)
  - [Topic 2. EDA](#topic-2-eda)
  - [Topic 3. Linear Regression](#topic-3-linear-regression)
  - [Topic 4. Time Series](#topic-4-time-series)
  - [Topic 5. Logistic Regression](#topic-5-logistic-regression)
  - [Topic 6. Decision Trees](#topic-6-decision-trees)
  - [Topic 7. KNN - Naive Bayes](#topic-7-knn---naive-bayes)
  - [Topic 8. SVM](#topic-8-svm)
  - [Topic 9. Ensembles](#topic-9-ensembles)
  - [Topic 10. Feature Engineering](#topic-10-feature-engineering)
- [Environment Setup](#environment-setup)
- [Project Structure](#project-structure)
- [Requirements](#requirements)

# Topics

## Topic 1. Basics

`Topic 1. Basics/app.py` is a Streamlit demo that compares a previous model with a new model version in a consumer lending scenario. It focuses on business-facing model metrics such as gross profit, automation rate, approval rate, manual reviews, and default rate among approved loans.

Run it with:

```bash
streamlit run "Topic 1. Basics/app.py"
```

## Topic 2. EDA

`Topic 2. EDA/EDA_v1.ipynb` demonstrates a broader exploratory data analysis workflow with the breast cancer and diabetes datasets from `scikit-learn`. It covers target balance, missing values, feature distributions, boxplots, correlations, scatter matrices, PCA, parallel coordinates, and regression-style feature-target analysis.

`Topic 2. EDA/EDA_v2.ipynb` provides a compact EDA workflow with the Wine dataset from `scikit-learn`. It covers imports, data loading, structure checks, data quality checks, target distribution, feature distributions, class comparisons, correlations, and a two-feature scatter plot.

## Topic 3. Linear Regression

`Topic 3. Linear Regression/linear_regression.ipynb` introduces linear regression with simple datasets from `scikit-learn`. It covers one-feature regression, multiple linear regression, residuals, model evaluation metrics, and visual examples of underfitting and overfitting. The notebook also demonstrates why adding more data usually does not fix an underfit model, but can reduce overfitting in a flexible model.

## Topic 4. Time Series

`Topic 4. Time Series/time_series.ipynb` introduces time series forecasting with classic machine learning using the public monthly Airline Passengers dataset. It covers chronological train/test splits, lag features, shifted rolling-window features, baseline forecasts, Ridge Regression, Random Forest, time-aware cross-validation with `TimeSeriesSplit`, and recursive future forecasting. The notebook can load the public CSV and includes an embedded fallback copy for offline use.

## Topic 5. Logistic Regression

`Topic 5. Logistic Regression/logistic_regression.ipynb` introduces logistic regression with the Breast Cancer Wisconsin Diagnostic dataset from `scikit-learn`. It covers sigmoid probabilities, scaling and coefficients, confusion matrices, why accuracy can be misleading on imbalanced binary classification, and how F1/threshold selection better reflect positive-class performance.

## Topic 6. Decision Trees

`Topic 6. Decision Trees/decision_trees.ipynb` introduces Decision Tree classification with the Breast Cancer Wisconsin Diagnostic dataset from `scikit-learn`. It focuses on how trees work under the hood: threshold rules, entropy, Gini impurity, information gain, a manual split calculation, tree inspection, prediction paths, feature importances, and overfitting control with tree size constraints.

## Topic 7. KNN - Naive Bayes

`Topic 7. KNN - Naive Bayes/knn_naive_bayes.ipynb` introduces K-Nearest Neighbors and Gaussian Naive Bayes classification with the Breast Cancer Wisconsin Diagnostic dataset from `scikit-learn`. It covers distance-based prediction, why scaling matters for KNN, choosing `k` with cross-validation, two-feature KNN decision regions, Gaussian Naive Bayes assumptions, learned class-feature statistics, confusion matrices, and side-by-side model comparison.

## Topic 8. SVM

`Topic 8. SVM/svm.ipynb` introduces Support Vector Machine classification with the Breast Cancer Wisconsin Diagnostic dataset from `scikit-learn`. It focuses on how the regularization parameter `C` changes the learned margin, support vectors, model performance, and two-feature decision areas for linear and RBF kernels.

## Topic 9. Ensembles

`Topic 9. Ensembles/ensembles.ipynb` introduces ensemble learning with Logistic Regression as a linear baseline, a shallow Decision Tree for intuition, and `scikit-learn` compatible LightGBM and XGBoost classifiers. It uses the Forest Cover Type dataset from `scikit-learn` when available, with a bundled Digits fallback if the larger dataset cannot be downloaded. The notebook compares accuracy, balanced accuracy, macro F1, per-class reports, confusion matrices, feature importances, and a simple LightGBM + XGBoost soft-voting ensemble.

This topic has optional extra dependencies pinned in `Topic 9. Ensembles/requirements-ensembles.txt`:

```bash
python -m pip install -r "Topic 9. Ensembles/requirements-ensembles.txt"
```

## Topic 10. Feature Engineering

`Topic 10. Feature Engineering/feature_engineering.ipynb` introduces feature engineering for tabular machine learning with a fully offline synthetic customer churn dataset. It covers numeric imputation and scaling, log transforms for skewed features, one-hot encoding with unknown-category handling, date-derived tenure and recency features, interaction features, TF-IDF text features, a leakage-safe `ColumnTransformer` pipeline, model comparison, confusion matrices, and coefficient-based feature inspection.

# Environment Setup

Create and activate a conda environment, then install the project dependencies from `requirements.txt`:

```bash
conda create -n ml-fundamentals python=3.10.12
conda activate ml-fundamentals
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

To verify the environment:

```bash
python -c "import numpy, pandas, sklearn, matplotlib, streamlit, altair; print('Environment ready')"
```

# Project Structure

```text
.
├── Topic 1. Basics/
│   └── app.py
├── Topic 2. EDA/
│   ├── EDA_v1.ipynb
│   └── EDA_v2.ipynb
├── Topic 3. Linear Regression/
│   └── linear_regression.ipynb
├── Topic 4. Time Series/
│   └── time_series.ipynb
├── Topic 5. Logistic Regression/
│   └── logistic_regression.ipynb
├── Topic 6. Decision Trees/
│   └── decision_trees.ipynb
├── Topic 7. KNN - Naive Bayes/
│   └── knn_naive_bayes.ipynb
├── Topic 8. SVM/
│   └── svm.ipynb
├── Topic 9. Ensembles/
│   ├── ensembles.ipynb
│   └── requirements-ensembles.txt
├── Topic 10. Feature Engineering/
│   └── feature_engineering.ipynb
├── README.md
└── requirements.txt
```

# Requirements

The dependency versions are pinned in `requirements.txt` and include:

- `numpy`
- `pandas`
- `scikit-learn`
- `scipy`
- `matplotlib`
- `streamlit`
- `altair`

Topic 9 additionally uses the optional pinned packages:

- `lightgbm==4.3.0`
- `xgboost==2.0.3`
