# AI Health Risk Prediction & Model Optimization
# Author: Jacob Gomez
# Date: 2026-01-12
# Description: End-to-end machine learning pipeline for predicting health risk scores

# -------------------------------
# Import Libraries
# -------------------------------
import pandas as pd
import numpy as np
import math
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# -------------------------------
# Load Dataset
# -------------------------------
data = pd.read_csv("dataset.csv")

print("\n--- DATASET OVERVIEW ---")
print(data.info())
print("\nMissing Values per Column:")
print(data.isnull().sum())

# -------------------------------
#  Data Preprocessing
# -------------------------------
# Fill missing numeric values with median
num_cols = data.select_dtypes(include=['float64','int64']).columns
data[num_cols] = data[num_cols].fillna(data[num_cols].median())

# Encode categorical variables
cat_cols = data.select_dtypes(include=['object','category']).columns
for col in cat_cols:
    data[col] = data[col].astype('category').cat.codes

print("\n--- PROCESSED DATASET INFO ---")
print(data.info())

# -------------------------------
# Train/Test Split
# -------------------------------
X = data.drop('healthRiskScore', axis=1)
y = data['healthRiskScore']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# -------------------------------
# Baseline Model (Random Forest)
# -------------------------------
baseline_rf = RandomForestRegressor(random_state=42)
baseline_rf.fit(X_train, y_train)
y_pred_baseline = baseline_rf.predict(X_test)

rmse_baseline = math.sqrt(mean_squared_error(y_test, y_pred_baseline))
r2_baseline = r2_score(y_test, y_pred_baseline)

print("\n--- BASELINE MODEL ---")
print(f"Baseline RMSE: {rmse_baseline:.6f}")
print(f"Baseline R²: {r2_baseline:.6f}")

# -------------------------------
# Random Forest Optimization (Hyperparameter Tuning)
# -------------------------------
param_grid = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [None, 10, 20, 30, 50],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2']
}

rf = RandomForestRegressor(random_state=42)

rf_random = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_grid,
    n_iter=50,
    scoring='r2',
    cv=5,
    verbose=1,
    random_state=42,
    n_jobs=-1
)

rf_random.fit(X_train, y_train)

print("\nBest Hyperparameters (Randomized Search):")
print(rf_random.best_params_)

y_pred_opt = rf_random.predict(X_test)
rmse_opt = math.sqrt(mean_squared_error(y_test, y_pred_opt))
r2_opt = r2_score(y_test, y_pred_opt)

print("\n--- OPTIMIZED RANDOM FOREST ---")
print(f"Optimized RMSE: {rmse_opt:.6f}")
print(f"Optimized R²: {r2_opt:.6f}")

# -------------------------------
# Ensemble Model (Gradient Boosting)
# -------------------------------
gb_model = GradientBoostingRegressor(random_state=42)
gb_model.fit(X_train, y_train)
y_pred_gb = gb_model.predict(X_test)

rmse_gb = math.sqrt(mean_squared_error(y_test, y_pred_gb))
r2_gb = r2_score(y_test, y_pred_gb)

print("\n--- GRADIENT BOOSTING MODEL ---")
print(f"GB RMSE: {rmse_gb:.6f}")
print(f"GB R²: {r2_gb:.6f}")

# -------------------------------
# Feature Importance (Optimized RF)
# -------------------------------
importances = rf_random.best_estimator_.feature_importances_
sorted_idx = np.argsort(importances)

plt.figure(figsize=(10,8))
plt.barh(X.columns[sorted_idx], importances[sorted_idx])
plt.xlabel("Importance Score")
plt.title("Feature Importance (Optimized Random Forest)")
plt.tight_layout()
plt.show()

# -------------------------------
# Comparison Table of Models
# -------------------------------
results = pd.DataFrame({
    'Model': ['Baseline RF', 'Optimized RF', 'Gradient Boosting'],
    'RMSE': [rmse_baseline, rmse_opt, rmse_gb],
    'R2': [r2_baseline, r2_opt, r2_gb]
})

print("\n--- MODEL COMPARISON ---")
print(results)

# -------------------------------
# Regularization Models (Ridge & Lasso)
# -------------------------------
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
ridge_pred = ridge.predict(X_test)

lasso = Lasso(alpha=0.001, max_iter=10000)
lasso.fit(X_train, y_train)
lasso_pred = lasso.predict(X_test)

rmse_ridge = math.sqrt(mean_squared_error(y_test, ridge_pred))
r2_ridge = r2_score(y_test, ridge_pred)

rmse_lasso = math.sqrt(mean_squared_error(y_test, lasso_pred))
r2_lasso = r2_score(y_test, lasso_pred)

print("\n--- REGULARIZATION MODELS ---")
print(f"Ridge RMSE: {rmse_ridge:.6f}, R²: {r2_ridge:.6f}")
print(f"Lasso RMSE: {rmse_lasso:.6f}, R²: {r2_lasso:.6f}")

# Append regularization results to comparison table
reg_results = pd.DataFrame({
    'Model': ['Ridge', 'Lasso'],
    'RMSE': [rmse_ridge, rmse_lasso],
    'R2': [r2_ridge, r2_lasso]
})

results = pd.concat([results, reg_results], ignore_index=True)

print("\n--- UPDATED MODEL COMPARISON ---")
print(results)

# Optional: Save results to CSV
results.to_csv("task2_results.csv", index=False)
