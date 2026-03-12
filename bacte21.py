# ==============================
# COMPARE DATASETS AND RESULTS
# ==============================

import pandas as pd
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt

# Paths
new_results_dir = r"C:\Users\Amare\Desktop\diz_results_20260311_125516"
old_data_path = r"C:\Users\Amare\Desktop\worku\datato232-81.csv"
new_data_path = r"C:\Users\Amare\Desktop\getachewt\datato232-81.csv"

print("="*70)
print("DATASET COMPARISON")
print("="*70)

# Load both datasets
df_old = pd.read_csv(old_data_path)
df_new = pd.read_csv(new_data_path)

target = 'DIZ'
features = ['Cbac', 'CAgNP', 'Ra']

print("\n📊 DATASET SIZES:")
print(f"  Old dataset: {len(df_old)} rows")
print(f"  New dataset: {len(df_new)} rows")

# Clean and compare statistics
print("\n📈 TARGET STATISTICS (after cleaning):")
for name, df in [("Old", df_old), ("New", df_new)]:
    df_clean = df.copy()
    for col in [target] + features:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    df_clean = df_clean.dropna()
    
    y = df_clean[target].values.astype(float)
    
    print(f"\n  {name} dataset ({len(df_clean)} samples):")
    print(f"    Mean: {y.mean():.2f}")
    print(f"    Std:  {y.std():.2f}")
    print(f"    Min:  {y.min():.2f}")
    print(f"    Max:  {y.max():.2f}")

# Load new results
print("\n" + "="*70)
print("NEW DATASET RESULTS")
print("="*70)

# Load equations
csv_path = os.path.join(new_results_dir, "all_equations.csv")
df_eq = pd.read_csv(csv_path)

# Show top 5
print("\n🏆 TOP 5 EQUATIONS:")
for i in range(min(5, len(df_eq))):
    row = df_eq.iloc[i]
    contains_bacteria = 'x0' in str(row['equation'])
    marker = '✓' if contains_bacteria else '✗'
    print(f"\n{i+1}. Loss: {row['loss']:.6f} | Complexity: {row['complexity']} | Bacteria: {marker}")
    print(f"   {row['equation']}")

# Best equation
best_row = df_eq.iloc[0]
print("\n" + "⭐"*70)
print("🌟 BEST EQUATION")
print("⭐"*70)
print(f"Equation: {best_row['equation']}")
print(f"Loss: {best_row['loss']:.6f}")
print(f"Complexity: {best_row['complexity']}")
print(f"Contains bacteria: {'✓' if 'x0' in str(best_row['equation']) else '✗'}")

# Load model and get predictions
model_path = os.path.join(new_results_dir, "pysr_model.pkl")
scaler_path = os.path.join(new_results_dir, "scaler.pkl")

with open(model_path, 'rb') as f:
    model = pickle.load(f)

with open(scaler_path, 'rb') as f:
    scaler = pickle.load(f)

# Get predictions on new data
df_clean_new = df_new.copy()
for col in [target] + features:
    df_clean_new[col] = pd.to_numeric(df_clean_new[col], errors='coerce')
df_clean_new = df_clean_new.dropna()

X_new = df_clean_new[features].values.astype(float)
y_new = df_clean_new[target].values.astype(float)
X_new_scaled = scaler.transform(X_new)

y_pred = model.predict(X_new_scaled)

from sklearn.metrics import r2_score
r2 = r2_score(y_new, y_pred)
print(f"\n📊 R² Score on new dataset: {r2:.4f}")

# Compare with feature correlations
print("\n" + "="*70)
print("FEATURE CORRELATIONS COMPARISON")
print("="*70)

print("\nFeature correlations with DIZ:")
for name, df in [("Old dataset", df_old), ("New dataset", df_new)]:
    df_clean = df.copy()
    for col in [target] + features:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    df_clean = df_clean.dropna()
    
    X = df_clean[features].values.astype(float)
    y = df_clean[target].values.astype(float)
    
    print(f"\n{name}:")
    for i, feat in enumerate(features):
        corr = np.corrcoef(X[:, i], y)[0, 1]
        print(f"  {feat}: {corr:.4f}")

print("\n" + "="*70)
print("INSIGHTS")
print("="*70)
print("""
The new dataset shows:
1. Much lower R² (0.244 vs 0.689) - different relationship
2. Bacteria not in best equation (ranked 4th)
3. Model relies mainly on CAgNP (x1) and Ra (x2)

Possible reasons:
• Different experimental conditions
• Different measurement techniques
• More noise in new dataset
• Smaller range of values

To improve results on new dataset:
1. Check data quality and outliers
2. Try different PySR parameters
3. Consider collecting more samples
""")