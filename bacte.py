# ==============================
# Detailed Analysis of Best DIZ Equation
# ==============================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import RobustScaler
import math

# Paths
results_file = r"C:\Users\Amare\Desktop\pysr_final_diz_results\all_evaluated_equations.csv"
data_path = r"C:\Users\Amare\Desktop\worku\datato232-81.csv"
output_dir = r"C:\Users\Amare\Desktop\diz_best_equation_details"
os.makedirs(output_dir, exist_ok=True)

print("="*70)
print("DETAILED ANALYSIS OF BEST DIZ EQUATION")
print("="*70)

# ------------------------------
# 1. Load the best equation
# ------------------------------
df_results = pd.read_csv(results_file)
best_row = df_results.loc[df_results['r2'].idxmax()]

print(f"\n🌟 BEST EQUATION FOUND:")
print(f"   R² Score: {best_row['r2']:.4f} ({best_row['r2']*100:.1f}% variance explained)")
print(f"   MAE: {best_row['mae']:.4f}")
print(f"   RMSE: {best_row['rmse']:.4f}")
print(f"   Complexity: {best_row['complexity']}")
print(f"   Source: {best_row['source']}")
print(f"\n   Equation: {best_row['equation']}")

# ------------------------------
# 2. Load original data
# ------------------------------
print("\n" + "="*70)
print("LOADING ORIGINAL DATA")
print("="*70)

df_data = pd.read_csv(data_path)
target = 'DIZ'
features = [col for col in df_data.columns if col != target]

# Clean data
df_clean = df_data.copy()
for col in [target] + features:
    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
df_clean = df_clean.dropna()

X = df_clean[features].values.astype(float)
y = df_clean[target].values.astype(float)

# Scale features
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)

print(f"\nData shape: {X.shape}")
print(f"Features: {features}")

# ------------------------------
# 3. Safe evaluator function
# ------------------------------
def evaluate_equation(expr, x_values):
    """
    Safely evaluate the equation
    """
    import math
    import numpy as np
    
    safe_dict = {
        'sin': math.sin, 'cos': math.cos, 'tan': math.tan,
        'exp': math.exp, 'log': math.log, 'log10': math.log10,
        'sqrt': math.sqrt, 'abs': abs,
        'square': lambda x: x**2, 'cube': lambda x: x**3,
        'pow': pow, 'pi': math.pi, 'e': math.e
    }
    
    for i, val in enumerate(x_values):
        safe_dict[f'x{i}'] = val
    
    try:
        expr = expr.replace('^', '**')
        result = eval(expr, {"__builtins__": {}}, safe_dict)
        return float(result) if isinstance(result, (int, float, np.number)) else np.nan
    except:
        return np.nan

# ------------------------------
# 4. Get predictions
# ------------------------------
print("\n" + "="*70)
print("GENERATING PREDICTIONS")
print("="*70)

predictions = []
for i in range(len(y)):
    pred = evaluate_equation(best_row['equation'], X_scaled[i])
    predictions.append(pred)

predictions = np.array(predictions)
valid_mask = ~np.isnan(predictions)
y_valid = y[valid_mask]
pred_valid = predictions[valid_mask]

print(f"\nValid predictions: {sum(valid_mask)}/{len(y)} ({sum(valid_mask)/len(y)*100:.1f}%)")

# ------------------------------
# 5. Comprehensive analysis
# ------------------------------
print("\n" + "="*70)
print("COMPREHENSIVE STATISTICAL ANALYSIS")
print("="*70)

# Calculate detailed metrics
r2 = r2_score(y_valid, pred_valid)
mae = mean_absolute_error(y_valid, pred_valid)
rmse = np.sqrt(mean_squared_error(y_valid, pred_valid))
mape = np.mean(np.abs((y_valid - pred_valid) / (y_valid + 1e-10))) * 100
residuals = y_valid - pred_valid

# Additional metrics
n = len(y_valid)
p = X.shape[1]
adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1) if n > p + 1 else r2
mse = mean_squared_error(y_valid, pred_valid)
rmspe = np.sqrt(np.mean(((y_valid - pred_valid) / (y_valid + 1e-10)) ** 2)) * 100

print(f"\n📊 PERFORMANCE METRICS:")
print(f"   R²:                 {r2:.6f} ({r2*100:.2f}%)")
print(f"   Adjusted R²:        {adj_r2:.6f}")
print(f"   MAE:                {mae:.6f}")
print(f"   RMSE:               {rmse:.6f}")
print(f"   MSE:                {mse:.6f}")
print(f"   MAPE:               {mape:.2f}%")
print(f"   RMSPE:              {rmspe:.2f}%")

print(f"\n📈 RESIDUAL STATISTICS:")
print(f"   Mean Residual:      {np.mean(residuals):.6f}")
print(f"   Std Residual:       {np.std(residuals):.6f}")
print(f"   Min Residual:       {np.min(residuals):.6f}")
print(f"   Max Residual:       {np.max(residuals):.6f}")
print(f"   Residual Range:     {np.max(residuals)-np.min(residuals):.6f}")

# ------------------------------
# 6. Create detailed visualizations
# ------------------------------
print("\n" + "="*70)
print("CREATING DETAILED VISUALIZATIONS")
print("="*70)

fig = plt.figure(figsize=(18, 12))

# 1. Predicted vs Actual (main plot)
ax1 = plt.subplot(2, 3, 1)
ax1.scatter(y_valid, pred_valid, alpha=0.6, c='blue', edgecolors='black', s=70)
ax1.plot([y_valid.min(), y_valid.max()], [y_valid.min(), y_valid.max()], 
         'r--', lw=2.5, label='Perfect Fit')
ax1.set_xlabel('Actual DIZ Values', fontsize=12)
ax1.set_ylabel('Predicted DIZ Values', fontsize=12)
ax1.set_title(f'Predicted vs Actual DIZ\nR² = {r2:.4f}', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. Residuals vs Predicted
ax2 = plt.subplot(2, 3, 2)
ax2.scatter(pred_valid, residuals, alpha=0.6, c='green', edgecolors='black', s=70)
ax2.axhline(0, color='r', linestyle='--', lw=2.5)
ax2.set_xlabel('Predicted DIZ Values', fontsize=12)
ax2.set_ylabel('Residuals', fontsize=12)
ax2.set_title(f'Residual Plot\nMean = {np.mean(residuals):.4f}, Std = {np.std(residuals):.4f}', 
             fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)

# 3. Residuals Distribution
ax3 = plt.subplot(2, 3, 3)
ax3.hist(residuals, bins=25, edgecolor='black', alpha=0.7, color='purple')
ax3.axvline(0, color='r', linestyle='--', lw=2.5)
ax3.axvline(np.mean(residuals), color='orange', linestyle='--', lw=2, label=f'Mean: {np.mean(residuals):.3f}')
ax3.set_xlabel('Residuals', fontsize=12)
ax3.set_ylabel('Frequency', fontsize=12)
ax3.set_title('Residuals Distribution', fontsize=14, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 4. Q-Q Plot
ax4 = plt.subplot(2, 3, 4)
from scipy import stats
stats.probplot(residuals, dist="norm", plot=ax4)
ax4.set_title('Q-Q Plot (Normality Check)', fontsize=14, fontweight='bold')
ax4.grid(True, alpha=0.3)

# 5. Actual vs Predicted (Time Series - first 50)
ax5 = plt.subplot(2, 3, 5)
n_show = min(50, len(y_valid))
indices = np.arange(n_show)
ax5.plot(indices, y_valid[:n_show], 'b-', label='Actual', linewidth=2, marker='o', markersize=4)
ax5.plot(indices, pred_valid[:n_show], 'r--', label='Predicted', linewidth=2, marker='s', markersize=4)
ax5.set_xlabel('Sample Index', fontsize=12)
ax5.set_ylabel('DIZ Value', fontsize=12)
ax5.set_title(f'Actual vs Predicted (First {n_show} Samples)', fontsize=14, fontweight='bold')
ax5.legend()
ax5.grid(True, alpha=0.3)

# 6. Equation and metrics
ax6 = plt.subplot(2, 3, 6)
ax6.axis('off')
eq_text = f"Best Equation:\n\n{best_row['equation']}\n\n"
eq_text += "Performance Metrics:\n"
eq_text += f"R² = {r2:.4f}\n"
eq_text += f"Adj R² = {adj_r2:.4f}\n"
eq_text += f"MAE = {mae:.4f}\n"
eq_text += f"RMSE = {rmse:.4f}\n"
eq_text += f"MAPE = {mape:.2f}%\n\n"
eq_text += f"Valid Samples: {len(y_valid)}/{len(y)}"
ax6.text(0.1, 0.5, eq_text, fontsize=11, verticalalignment='center',
        fontfamily='monospace', 
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))

plt.suptitle(f'DETAILED ANALYSIS: Best Equation for DIZ (R² = {r2:.4f})', 
            fontsize=16, fontweight='bold')
plt.tight_layout()

# Save plot
plot_path = os.path.join(output_dir, "detailed_analysis.png")
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.show()
print(f"\n✅ Detailed plot saved to: {plot_path}")

# ------------------------------
# 7. Save equation in multiple formats
# ------------------------------
print("\n" + "="*70)
print("SAVING EQUATION IN MULTIPLE FORMATS")
print("="*70)

# Python format
py_path = os.path.join(output_dir, "equation_python.py")
with open(py_path, 'w') as f:
    f.write("# Equation for DIZ prediction\n")
    f.write("# Features: x0, x1, x2 correspond to scaled feature values\n\n")
    f.write("import math\n\n")
    f.write("def predict_diz(x0, x1, x2):\n")
    f.write("    \"\"\"\n")
    f.write("    Predict DIZ value from scaled features\n")
    f.write(f"    R² = {r2:.4f}, MAE = {mae:.4f}\n")
    f.write("    \"\"\"\n")
    
    # Convert equation to Python
    py_eq = best_row['equation'].replace('^', '**')
    f.write(f"    return {py_eq}\n")

print(f"✅ Python function saved to: {py_path}")

# Excel format
excel_path = os.path.join(output_dir, "equation_excel.txt")
with open(excel_path, 'w') as f:
    f.write("Excel Formula for DIZ Prediction\n")
    f.write("="*50 + "\n\n")
    f.write("Assuming your scaled features are in cells A1, B1, C1:\n\n")
    
    # Convert to Excel formula
    excel_eq = best_row['equation'].replace('^', '^')
    excel_eq = excel_eq.replace('x0', 'A1').replace('x1', 'B1').replace('x2', 'C1')
    f.write(f"= {excel_eq}\n")

print(f"✅ Excel formula saved to: {excel_path}")

# MATLAB format
matlab_path = os.path.join(output_dir, "equation_matlab.m")
with open(matlab_path, 'w') as f:
    f.write("% MATLAB function for DIZ prediction\n")
    f.write(f"% R² = {r2:.4f}, MAE = {mae:.4f}\n\n")
    f.write("function y = predict_diz(x0, x1, x2)\n")
    matlab_eq = best_row['equation'].replace('^', '.^')
    f.write(f"    y = {matlab_eq};\n")
    f.write("end\n")

print(f"✅ MATLAB function saved to: {matlab_path}")

# ------------------------------
# 8. Final summary
# ------------------------------
print("\n" + "="*70)
print("FINAL SUMMARY")
print("="*70)
print(f"\n🌟 Best Equation for DIZ:")
print(f"   {best_row['equation']}")
print(f"\n📊 Performance:")
print(f"   R² = {r2:.4f} ({r2*100:.1f}% variance explained)")
print(f"   MAE = {mae:.4f}")
print(f"   RMSE = {rmse:.4f}")
print(f"   MAPE = {mape:.2f}%")
print(f"\n📁 All files saved to: {output_dir}")
print("\n" + "="*70)