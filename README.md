<h1 align="center">ML Fundamentals</h1>

This repository contains introductory machine learning materials organized by topic. The examples use synthetic data and public datasets bundled with `scikit-learn`, so the current notebooks and app do not require external data downloads.

# Table of Contents

- [Topics](#topics)
  - [Topic 1. Basics](#topic-1-basics)
  - [Topic 2. EDA](#topic-2-eda)
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
