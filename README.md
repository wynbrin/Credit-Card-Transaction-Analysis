**Project Overview**

I created this repository to gain deeper experience applying machine learning to real‑world financial data. The repo contains two complementary projects built from the same credit card transaction dataset:

Time Series Forecasting — predicting daily transaction volume using engineered temporal features and XGBoost

Fraud Detection — classifying fraudulent transactions using XGBoost and feature engineering

Both notebooks demonstrate practical ML workflows, model evaluation, and interpretability techniques relevant to financial services.

**Data Source**

This project uses the Credit Card Transactions Dataset published on Kaggle by Priyam Choksi:

https://www.kaggle.com/datasets/priyamchoksi/credit-card-transactions-dataset/data

The dataset is not included in this repository due to size constraints.

To reproduce the analysis:

  - Download the dataset from Kaggle
  
  - Place the files into the data/ directory
  
  - Run the preprocessing steps in each notebook to generate the required features

**Time Series Forecasting (Time Series.ipynb)**

This notebook focuses on forecasting daily credit card transaction volume. The workflow includes:

**Feature Engineering**

- Lag features: lag_1, lag_7, lag_30

- Rolling volatility: rolling_std_7

- Calendar features: day_of_week, month, is_weekend, is_holiday

**Modeling**

I use the XGBoost Regressor, which handles nonlinear patterns, interactions, and seasonality far better than linear or logistic regression for this type of financial time series.

**Evaluation**

- Train/test split based on time

- Actual vs predicted visualization

- Residual diagnostics

- SHAP interpretability

**Rolling window backtesting**

This notebook demonstrates how engineered temporal features and gradient boosting can produce stable, interpretable forecasts.

**Fraud Detection (XGBoost.ipynb)**

This notebook implements a fraud classification model using XGBoost. It includes:

-Exploratory data analysis

-Feature engineering

- Handling class imbalance

- Model training and evaluation

- -ROC, precision‑recall, and threshold tuning

- SHAP‑based interpretability

The goal is to build a practical fraud detection pipeline that balances recall, precision, and operational constraints.


