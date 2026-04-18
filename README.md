# Ames Housing Price Prediction 🏠

A complete end-to-end Machine Learning project predicting house prices using the Ames Housing dataset with scikit-learn Pipelines.

## Project Overview

This project applies the full ML workflow from Chapter 2 of *Hands-On ML with Scikit-Learn, Keras & TensorFlow* — from raw data exploration to final model evaluation.

## Workflow

1. **Exploratory Data Analysis** — correlations, scatter matrix, feature distributions
2. **Feature Selection** — selected top features using Correlation & ANOVA
3. **Preprocessing Pipeline** — SimpleImputer + StandardScaler via make_pipeline
4. **Modeling** — compared two regressors:
   - `LinearRegression` (baseline)
   - `RandomForestRegressor` + GridSearchCV
5. **Evaluation** — RMSE, MAE, Cross-Validation

## Selected Features

| Feature | Description |
|---|---|
| Overall Qual | Overall material and finish quality |
| Gr Liv Area | Above grade living area (sq ft) |
| Garage Cars | Garage capacity |
| Garage Area | Garage size (sq ft) |
| Total Bsmt SF | Total basement area |
| 1st Flr SF | First floor area |

## Results

| Model | CV RMSE | Test RMSE |
|---|---|---|
| LinearRegression | ~30,468 | ~31,985 |
| RandomForestRegressor | Higher (overfitting) | — |

**Winner: LinearRegression** — more stable and generalizes better on this dataset.

## Key Observations

- RandomForest showed signs of overfitting on training data
- LinearRegression proved more reliable with lower and more consistent CV RMSE
- Feature selection via correlation significantly reduced noise

## Tech Stack

- Python 3.10
- scikit-learn
- pandas
- matplotlib / seaborn

## Dataset

[Ames Housing Dataset](https://www.kaggle.com/datasets/prevek18/ames-housing-dataset) — 2,930 houses with 82 features.

## Part of

[Hands-On ML with Scikit-Learn, Keras & TensorFlow](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/) — Chapter 2 practice project.
