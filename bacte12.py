# ==============================
# Analyze the Best Equation
# ==============================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
import os

# Paths
data_path = r"C:\Users\Amare\Desktop\worku\datato232-81.csv"
output_dir = r"C:\Users\Amare\Desktop\pysr_final_manual_save"

print("="*70)
print("ANALYZING THE BEST EQUATION")
print("="*70)

# Load data
df = pd.read_csv(data_path)
target = 'DIZ'
features = ['Cbac', 'CAgNP', 'Ra']

# Clean data
df_clean = df.copy()
for col in [target] + features:
    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
df_clean = df_clean.dropna()

X = df_clean[features].values.astype(float)
y = df_clean[target].values.astype(float)

# Scale data
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)

# The equation: 13.503544 - ((x2 + ((x1 + 0.3436411) * x0)) / square(x0 + 1.1927102))
def equation_prediction(x0, x1, x2):
    """Calculate DIZ from scaled features"""
    return 13.503544 - ((x2 + ((x1 + 0.3436411) * x0)) / ((x0 + 1.1927102) ** 2))

# Make predictions
y_pred = np.array([equation_prediction(X_scaled[i,0], X_scaled[i,1], X_scaled[i,2]) 
                   for i in range(len(y))])

# Calculate metrics
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
r2 = r2_score(y, y_pred)
mae = mean_absolute_error(y, y_pred)
rmse = np.sqrt(mean_squared_error(y, y_pred))

print(f"\n📊 VERIFIED METRICS:")
print(f"  R² = {r2:.4f} ({r2*100:.1f}%)")
print(f"  MAE = {mae:.4f}")
print(f"  RMSE = {rmse:.4f}")

# Feature importance analysis
print("\n" + "="*70)
print("FEATURE IMPORTANCE ANALYSIS")
print("="*70)

# Test each feature's impact
print("\nEffect of increasing each feature by 1 standard deviation:")
print("-"*50)

baseline = np.array([X_scaled[:,0].mean(), X_scaled[:,1].mean(), X_scaled[:,2].mean()])
baseline_pred = equation_prediction(baseline[0], baseline[1], baseline[2])

for i, feat in enumerate(features):
    test_point = baseline.copy()
    test_point[i] += 1  # Increase by 1 std
    new_pred = equation_prediction(test_point[0], test_point[1], test_point[2])
    change = new_pred - baseline_pred
    print(f"  {feat}: {change:+.4f} change in DIZ")

# Visualize the relationship
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for i, feat in enumerate(features):
    # Create range of values for this feature
    feat_range = np.linspace(X_scaled[:,i].min(), X_scaled[:,i].max(), 100)
    
    # Keep other features at their mean
    predictions = []
    for val in feat_range:
        test_point = X_scaled.mean(axis=0).copy()
        test_point[i] = val
        pred = equation_prediction(test_point[0], test_point[1], test_point[2])
        predictions.append(pred)
    
    axes[i].plot(feat_range, predictions, 'b-', linewidth=2)
    axes[i].set_xlabel(f'{feat} (scaled)', fontsize=12)
    axes[i].set_ylabel('Predicted DIZ', fontsize=12)
    axes[i].set_title(f'Effect of {feat} on DIZ', fontsize=14)
    axes[i].grid(True, alpha=0.3)

plt.suptitle('How Each Feature Affects DIZ Prediction', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

print("\n" + "="*70)
print("INTERPRETATION")
print("="*70)
print("""
The equation shows:

DIZ = 13.5 - [ (Ra + (CAgNP + 0.34) × Cbac) / (Cbac + 1.19)² ]

Key insights:
1. BACTERIA (Cbac) appears in BOTH numerator and denominator
   - This creates a non-linear, saturating effect
   
2. When Cbac is small, the denominator is small → larger DIZ
3. When Cbac is large, the denominator grows quadratically → smaller DIZ

4. CAgNP and Ra work together with Cbac in the numerator
   - Their effect is amplified when Cbac is large

This explains why removing outliers hurt performance earlier:
the relationship is strongly non-linear and depends on extreme values!
""")

print("\n" + "="*70)
print("ANALYSIS COMPLETE")
print("="*70)