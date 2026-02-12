# -*- coding: utf-8 -*-
"""
Created on Mon Dec  8 13:46:12 2025

@author: Amare
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor, StackingRegressor

from xgboost import XGBRegressor
from catboost import CatBoostRegressor

# ---------------------------
# Load CSV
# ---------------------------
file_path = r"C:\Users\Amare\Desktop\worku\datato232.csv"
df = pd.read_csv(file_path, encoding="cp1252")

# Clean column names
df.columns = df.columns.str.strip()

# Rename columns
df = df.rename(columns={
    'Name of bacteria': 'Bacteria species',
    'Type of bacteria': 'Gram stain'
})

# Define features and target
features = [
    'Bacteria species',
    'Type of extract',
    'bacteria concentration',
    'AgNP concentration',
    'Resonance  nm',
    'Size nm',
    'shape',
    'Gram stain',
    'dispersity'
]

target = 'Inhibition size mm'

# Keep only necessary columns and drop rows with missing values
df = df[features + [target]].dropna()

# Split X and y
X = df[features]
y = df[target]

# Identify categorical and numeric columns
categorical_cols = X.select_dtypes(include='object').columns.tolist()
numeric_cols = X.select_dtypes(exclude='object').columns.tolist()

# ---------------------------
# Preprocessor for numeric + categorical encoding
# ---------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("num", "passthrough", numeric_cols)
    ]
)

# ---------------------------
# Train/test split
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------
# Define models
# ---------------------------

# CatBoost can handle categorical directly
cat_model = CatBoostRegressor(
    verbose=0,
    depth=6,
    learning_rate=0.1,
    iterations=400,
    random_state=42,
    cat_features=categorical_cols
)

# XGBoost and GradientBoost need encoded numeric features
xgb_model = Pipeline([
    ("prep", preprocessor),
    ("model", XGBRegressor(
        n_estimators=400,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
        random_state=42
    ))
])

gbr_model = Pipeline([
    ("prep", preprocessor),
    ("model", GradientBoostingRegressor(
        n_estimators=400,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    ))
])

# Stacking model
stack_model = StackingRegressor(
    estimators=[
        ("cat", cat_model),
        ("xgb", xgb_model),
        ("gbr", gbr_model)
    ],
    final_estimator=GradientBoostingRegressor(random_state=42)
)

models = {
    "CatBoost": cat_model,
    "XGBoost": xgb_model,
    "GradientBoosting": gbr_model,
    "Stacking": stack_model
}

# ---------------------------
# Train, predict, evaluate
# ---------------------------
results = []

for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    r2 = r2_score(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    results.append([name, r2, rmse])

    # Scatter Plot: Predicted vs Actual
    plt.figure(figsize=(6,5))
    plt.scatter(y_test, preds, c='black', marker='o')
    min_val = min(y_test.min(), preds.min())
    max_val = max(y_test.max(), preds.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'k--')
    plt.xlabel("Actual Inhibition Size (mm)")
    plt.ylabel("Predicted Inhibition Size (mm)")
    plt.title(f"{name}: Predicted vs Actual")
    plt.grid(True)
    plt.show()

    # Feature importance (except stacking)
    if name != "Stacking":
        try:
            if name == "CatBoost":
                importances = model.get_feature_importance()
            else:
                importances = model.named_steps["model"].feature_importances_
            
            plt.figure(figsize=(7,5))
            plt.barh(features, importances)
            plt.title(f"{name} Feature Importance")
            plt.xlabel("Importance")
            plt.grid(axis="x")
            plt.show()
        except:
            print(f"Feature importance not available for {name}")

# Results table
results_df = pd.DataFrame(results, columns=["Model", "R² Score", "RMSE"])
print("\nFINAL MODEL PERFORMANCE:")
print(results_df.to_string(index=False))
