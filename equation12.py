# ==============================
# Find and Evaluate Best Equation from PySR Results
# ==============================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import re

# Paths
results_dir = r"C:\Users\Amare\Desktop\pysr_optimized"
data_path = r"C:\Users\Amare\Desktop\worku\datato232-81.csv"
output_dir = r"C:\Users\Amare\Desktop\pysr_final_evaluation"
os.makedirs(output_dir, exist_ok=True)

print("="*70)
print("FINDING AND EVALUATING BEST EQUATION")
print("="*70)

# ------------------------------
# 1. Find the best equation from CSV
# ------------------------------
csv_file = os.path.join(results_dir, "all_equations.csv")

if not os.path.exists(csv_file):
    print(f"\n❌ Could not find {csv_file}")
    # Try to find any CSV file
    import glob
    csv_files = glob.glob(os.path.join(results_dir, "*.csv"))
    if csv_files:
        csv_file = csv_files[0]
        print(f"✅ Using alternative file: {csv_file}")
    else:
        print("❌ No CSV files found!")
        exit(1)

print(f"\n📊 Reading equations from: {csv_file}")
df_eq = pd.read_csv(csv_file)
print(f"   Found {len(df_eq)} equations")

# Sort by loss (lower is better)
df_eq = df_eq.sort_values('loss')
print(f"\n🏆 Top 5 equations by loss:")
for i in range(min(5, len(df_eq))):
    row = df_eq.iloc[i]
    print(f"\n  Rank {i+1}:")
    print(f"    Complexity: {row['complexity']}")
    print(f"    Loss: {row['loss']:.6f}")
    if 'score' in row:
        print(f"    Score: {row['score']:.4f}")
    print(f"    Equation: {row['equation']}")

# Best equation
best_row = df_eq.iloc[0]
best_eq_str = best_row['equation']
best_loss = best_row['loss']
best_complexity = best_row['complexity']

print("\n" + "⭐"*70)
print("🌟 BEST EQUATION OVERALL")
print("⭐"*70)
print(f"Complexity: {best_complexity}")
print(f"Loss: {best_loss:.6f}")
print(f"\nEquation: {best_eq_str}")

# Save best equation to file
best_txt = os.path.join(output_dir, "best_equation.txt")
with open(best_txt, 'w') as f:
    f.write("="*70 + "\n")
    f.write("BEST EQUATION FOR DIZ\n")
    f.write("="*70 + "\n\n")
    f.write(f"Complexity: {best_complexity}\n")
    f.write(f"Loss: {best_loss:.6f}\n\n")
    f.write(f"Equation: {best_eq_str}\n")
print(f"\n✅ Best equation saved to: {best_txt}")

# ------------------------------
# 2. Load and prepare data for evaluation
# ------------------------------
print("\n" + "="*70)
print("LOADING DATA FOR EVALUATION")
print("="*70)

df_data = pd.read_csv(data_path)
target = 'DIZ'
features = [col for col in df_data.columns if col != target]

print(f"\nFeatures: {features}")
print(f"Target: {target}")

# Clean data
df_clean = df_data.copy()
for col in [target] + features:
    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

initial_rows = len(df_clean)
df_clean = df_clean.dropna()
print(f"\nData cleaning:")
print(f"  Rows dropped: {initial_rows - len(df_clean)}")
print(f"  Final rows: {len(df_clean)}")

X = df_clean[features].values.astype(float)
y = df_clean[target].values.astype(float)

# Scale features (needed for the equation if it uses scaled inputs)
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)

print(f"\nFinal data shape: X {X.shape}, y {y.shape}")

# ------------------------------
# 3. Parse and evaluate the equation manually
# ------------------------------
print("\n" + "="*70)
print("MANUAL EQUATION EVALUATION")
print("="*70)

# Function to safely evaluate mathematical expressions
def safe_eval(expr, x0, x1, x2, feature_names):
    """
    Safely evaluate a mathematical expression with variables x0, x1, x2
    """
    # Create a safe namespace with math functions
    import math
    safe_dict = {
        'x0': x0, 'x1': x1, 'x2': x2,
        'sin': math.sin, 'cos': math.cos, 'tan': math.tan,
        'exp': math.exp, 'log': math.log, 'log10': math.log10,
        'sqrt': math.sqrt, 'abs': abs,
        'square': lambda x: x**2, 'cube': lambda x: x**3,
        'pow': pow, 'pi': math.pi, 'e': math.e
    }
    
    # Add feature names as variables if they exist
    for i, name in enumerate(feature_names[:3]):
        safe_dict[name] = [x0, x1, x2][i]
    
    try:
        # Replace ^ with ** for Python
        expr = expr.replace('^', '**')
        
        # Handle any special cases
        # This is a simple evaluation - for complex expressions, you might need more parsing
        result = eval(expr, {"__builtins__": {}}, safe_dict)
        return result
    except Exception as e:
        print(f"  Evaluation error: {e}")
        return np.nan

# Try to evaluate the best equation
print(f"\nEvaluating: {best_eq_str}")

# Map features to x0, x1, x2 based on order
feature_order = features[:3]  # Assuming first 3 features are used
print(f"\nFeature mapping: x0={feature_order[0]}, x1={feature_order[1]}, x2={feature_order[2]}")

# Evaluate for all samples
y_pred_manual = []
valid_count = 0
error_count = 0

for i in range(len(y)):
    try:
        # Use scaled or original features based on the equation
        # If the equation was trained on scaled data, use X_scaled
        pred = safe_eval(best_eq_str, X_scaled[i, 0], X_scaled[i, 1], X_scaled[i, 2], feature_order)
        if not np.isnan(pred):
            y_pred_manual.append(pred)
            valid_count += 1
        else:
            y_pred_manual.append(np.nan)
            error_count += 1
    except:
        y_pred_manual.append(np.nan)
        error_count += 1

y_pred_manual = np.array(y_pred_manual)

print(f"\nEvaluation results:")
print(f"  Successful evaluations: {valid_count}/{len(y)}")
print(f"  Failed evaluations: {error_count}")

if valid_count > 0:
    # Remove NaN values for metrics
    valid_mask = ~np.isnan(y_pred_manual)
    y_valid = y[valid_mask]
    y_pred_valid = y_pred_manual[valid_mask]
    
    # Calculate metrics
    r2 = r2_score(y_valid, y_pred_valid)
    mae = mean_absolute_error(y_valid, y_pred_valid)
    rmse = np.sqrt(mean_squared_error(y_valid, y_pred_valid))
    mape = np.mean(np.abs((y_valid - y_pred_valid) / (y_valid + 1e-10))) * 100
    
    print("\n📈 PERFORMANCE METRICS:")
    print(f"  R² Score:  {r2:.6f} ({r2*100:.2f}%)")
    print(f"  MAE:       {mae:.6f}")
    print(f"  RMSE:      {rmse:.6f}")
    print(f"  MAPE:      {mape:.2f}%")
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Predicted vs Actual
    axes[0].scatter(y_valid, y_pred_valid, alpha=0.6, c='blue', edgecolors='black', s=50)
    axes[0].plot([y_valid.min(), y_valid.max()], [y_valid.min(), y_valid.max()], 'r--', lw=2)
    axes[0].set_xlabel('Actual DIZ')
    axes[0].set_ylabel('Predicted DIZ')
    axes[0].set_title(f'Manual Evaluation: Predicted vs Actual (R² = {r2:.4f})')
    axes[0].grid(True, alpha=0.3)
    
    # Residuals
    residuals = y_valid - y_pred_valid
    axes[1].scatter(y_pred_valid, residuals, alpha=0.6, c='green', edgecolors='black', s=50)
    axes[1].axhline(0, color='r', linestyle='--', lw=2)
    axes[1].set_xlabel('Predicted DIZ')
    axes[1].set_ylabel('Residuals')
    axes[1].set_title('Residual Plot')
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle(f'Manual Evaluation of Best Equation\n{best_eq_str[:80]}...')
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, "manual_evaluation.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"\n✅ Plot saved to: {plot_path}")
    
    # Append metrics to best equation file
    with open(best_txt, 'a') as f:
        f.write("\n" + "-"*50 + "\n")
        f.write("MANUAL EVALUATION METRICS\n")
        f.write("-"*50 + "\n")
        f.write(f"R² Score:  {r2:.6f}\n")
        f.write(f"MAE:       {mae:.6f}\n")
        f.write(f"RMSE:      {rmse:.6f}\n")
        f.write(f"MAPE:      {mape:.2f}%\n")
    
    print(f"\n✅ Metrics appended to: {best_txt}")

# ------------------------------
# 4. Alternative: Use PySR's built-in prediction
# ------------------------------
print("\n" + "="*70)
print("ATTEMPTING TO LOAD PySR MODEL")
print("="*70)

try:
    # Try to load the model if it was saved
    model_files = glob.glob(os.path.join(results_dir, "*.pkl"))
    if model_files:
        import joblib
        model = joblib.load(model_files[0])
        print(f"✅ Loaded model from: {model_files[0]}")
        
        # Make predictions
        y_pred_model = model.predict(X_scaled)
        
        # Calculate metrics
        r2_model = r2_score(y, y_pred_model)
        print(f"\n📈 PySR Model R²: {r2_model:.4f}")
    else:
        print("⚠️  No saved model found. The model was not saved during training.")
        print("   To save the model, add 'model.save(\"model.pkl\")' after training.")
except Exception as e:
    print(f"⚠️  Could not load model: {e}")

# ------------------------------
# 5. Summary
# ------------------------------
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"\n✅ Best equation found and saved to: {best_txt}")
print(f"✅ Evaluation plot saved to: {plot_path}")
print(f"\nEquation: {best_eq_str}")
print(f"\nTo use this equation in Excel or other tools:")
print("1. Replace x0, x1, x2 with your feature names:")
for i, feat in enumerate(feature_order[:3]):
    print(f"   x{i} = {feat}")
print("2. Implement the mathematical expression in your tool")
print("3. Make sure to scale your inputs if the equation was trained on scaled data")

print("\n" + "="*70)
print("ANALYSIS COMPLETE")
print("="*70)