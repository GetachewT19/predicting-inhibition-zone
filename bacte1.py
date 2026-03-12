# ==============================
# Final Analysis - Real DIZ Data Only
# ==============================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import RobustScaler

# Paths
results_dir = r"C:\Users\Amare\Desktop\pysr_diagnostic_fixed"
data_path = r"C:\Users\Amare\Desktop\worku\datato232-81.csv"
output_dir = r"C:\Users\Amare\Desktop\pysr_final_diz_results"
os.makedirs(output_dir, exist_ok=True)

print("="*70)
print("FINAL ANALYSIS - REAL DIZ DATA ONLY")
print("="*70)

# ------------------------------
# 1. Load only real data results (exclude synthetic)
# ------------------------------
csv_files = glob.glob(os.path.join(results_dir, "*.csv"))
print(f"\n📊 Found {len(csv_files)} total files")

real_results = []
for csv_file in csv_files:
    filename = os.path.basename(csv_file)
    # Exclude synthetic results
    if 'synthetic' not in filename.lower():
        try:
            df = pd.read_csv(csv_file)
            if 'loss' in df.columns and 'equation' in df.columns and len(df) > 0:
                df['source_file'] = filename
                real_results.append(df)
                print(f"  ✅ {filename}: {len(df)} equations")
        except Exception as e:
            print(f"  ❌ {filename}: Error - {e}")

if not real_results:
    print("\n❌ No valid real data results found!")
    exit(1)

# Combine all real results
combined_df = pd.concat(real_results, ignore_index=True)
print(f"\n📊 Combined real data results: {len(combined_df)} total equations")

# ------------------------------
# 2. Load and prepare original data
# ------------------------------
print("\n" + "="*70)
print("LOADING DIZ DATA")
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
df_clean = df_clean.dropna()

X = df_clean[features].values.astype(float)
y = df_clean[target].values.astype(float)

# Scale features (equations were trained on scaled data)
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)

print(f"\nFinal data shape: X {X.shape}, y {y.shape}")
print(f"Target stats - Mean: {y.mean():.4f}, Std: {y.std():.4f}")

# ------------------------------
# 3. Safe equation evaluator
# ------------------------------
def safe_eval_equation(expr, x_values):
    """
    Safely evaluate a mathematical expression
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
        if isinstance(result, (int, float, np.number)):
            return float(result)
        elif hasattr(result, '__len__') and len(result) == 1:
            return float(result[0])
        else:
            return np.nan
    except Exception as e:
        return np.nan

# ------------------------------
# 4. Evaluate all equations from real data
# ------------------------------
print("\n" + "="*70)
print("EVALUATING EQUATIONS ON DIZ DATA")
print("="*70)

results_list = []
total_eq = len(combined_df)

for idx, row in combined_df.iterrows():
    if idx % 10 == 0:
        print(f"  Progress: {idx}/{total_eq}")
    
    expr = row['equation']
    
    # Evaluate on all samples
    predictions = []
    valid_count = 0
    
    for i in range(len(y)):
        pred = safe_eval_equation(expr, X_scaled[i])
        if not np.isnan(pred) and not np.isinf(pred):
            predictions.append(pred)
            valid_count += 1
        else:
            predictions.append(np.nan)
    
    predictions = np.array(predictions)
    valid_mask = ~np.isnan(predictions)
    
    if valid_count > len(y) * 0.5:  # At least 50% valid predictions
        y_valid = y[valid_mask]
        pred_valid = predictions[valid_mask]
        
        # Calculate metrics
        r2 = r2_score(y_valid, pred_valid)
        mae = mean_absolute_error(y_valid, pred_valid)
        rmse = np.sqrt(mean_squared_error(y_valid, pred_valid))
        
        results_list.append({
            'equation': expr,
            'loss': row['loss'],
            'complexity': row['complexity'],
            'source': row['source_file'],
            'r2': r2,
            'mae': mae,
            'rmse': rmse,
            'valid_pct': valid_count/len(y)*100
        })

# Convert to DataFrame
results_df = pd.DataFrame(results_list)
print(f"\n✅ Successfully evaluated {len(results_df)} equations")

if len(results_df) == 0:
    print("\n❌ No equations could be evaluated successfully!")
    exit(1)

# ------------------------------
# 5. Find best equations
# ------------------------------
print("\n" + "="*70)
print("BEST EQUATIONS BY R² SCORE")
print("="*70)

# Sort by R² (higher is better)
best_by_r2 = results_df.sort_values('r2', ascending=False).head(10)

for i, (idx, row) in enumerate(best_by_r2.iterrows()):
    print(f"\n{i+1}. R² = {row['r2']:.4f}")
    print(f"   MAE = {row['mae']:.4f}, RMSE = {row['rmse']:.4f}")
    print(f"   Complexity: {row['complexity']}, Loss: {row['loss']:.6f}")
    print(f"   Source: {row['source']}")
    print(f"   Valid predictions: {row['valid_pct']:.1f}%")
    print(f"   Equation: {row['equation'][:100]}...")

# Best equation
best_row = best_by_r2.iloc[0]

print("\n" + "⭐"*70)
print("🌟 BEST EQUATION FOR DIZ")
print("⭐"*70)
print(f"R² Score: {best_row['r2']:.4f} ({best_row['r2']*100:.1f}% variance explained)")
print(f"MAE: {best_row['mae']:.4f}")
print(f"RMSE: {best_row['rmse']:.4f}")
print(f"Complexity: {best_row['complexity']}")
print(f"Source: {best_row['source']}")
print(f"\nEquation: {best_row['equation']}")

# Save best equation
best_txt = os.path.join(output_dir, "best_diz_equation.txt")
with open(best_txt, 'w') as f:
    f.write("="*70 + "\n")
    f.write("BEST EQUATION FOR DIZ\n")
    f.write("="*70 + "\n\n")
    f.write(f"R² Score: {best_row['r2']:.6f} ({best_row['r2']*100:.2f}%)\n")
    f.write(f"MAE: {best_row['mae']:.6f}\n")
    f.write(f"RMSE: {best_row['rmse']:.6f}\n")
    f.write(f"Complexity: {best_row['complexity']}\n")
    f.write(f"Source: {best_row['source']}\n\n")
    f.write(f"Equation: {best_row['equation']}\n")

print(f"\n✅ Best equation saved to: {best_txt}")

# Save all evaluated results
results_path = os.path.join(output_dir, "all_evaluated_equations.csv")
results_df.to_csv(results_path, index=False)
print(f"✅ All evaluated equations saved to: {results_path}")

# ------------------------------
# 6. Visualize best equation
# ------------------------------
print("\n" + "="*70)
print("VISUALIZING BEST EQUATION")
print("="*70)

# Get predictions for best equation
predictions = []
for i in range(len(y)):
    pred = safe_eval_equation(best_row['equation'], X_scaled[i])
    predictions.append(pred)

predictions = np.array(predictions)
valid_mask = ~np.isnan(predictions)
y_valid = y[valid_mask]
pred_valid = predictions[valid_mask]

fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. Predicted vs Actual
axes[0, 0].scatter(y_valid, pred_valid, alpha=0.6, c='blue', edgecolors='black', s=60)
axes[0, 0].plot([y_valid.min(), y_valid.max()], [y_valid.min(), y_valid.max()], 
                'r--', lw=2, label='Perfect Fit')
axes[0, 0].set_xlabel('Actual DIZ Values', fontsize=12)
axes[0, 0].set_ylabel('Predicted DIZ Values', fontsize=12)
axes[0, 0].set_title(f'Predicted vs Actual DIZ\nR² = {best_row["r2"]:.4f}', fontsize=14)
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 2. Residuals
residuals = y_valid - pred_valid
axes[0, 1].scatter(pred_valid, residuals, alpha=0.6, c='green', edgecolors='black', s=60)
axes[0, 1].axhline(0, color='r', linestyle='--', lw=2)
axes[0, 1].set_xlabel('Predicted DIZ Values', fontsize=12)
axes[0, 1].set_ylabel('Residuals', fontsize=12)
axes[0, 1].set_title(f'Residual Plot\nMean = {np.mean(residuals):.4f}, Std = {np.std(residuals):.4f}', 
                     fontsize=14)
axes[0, 1].grid(True, alpha=0.3)

# 3. Residuals distribution
axes[1, 0].hist(residuals, bins=20, edgecolor='black', alpha=0.7, color='purple')
axes[1, 0].axvline(0, color='r', linestyle='--', lw=2)
axes[1, 0].set_xlabel('Residuals', fontsize=12)
axes[1, 0].set_ylabel('Frequency', fontsize=12)
axes[1, 0].set_title('Residuals Distribution', fontsize=14)
axes[1, 0].grid(True, alpha=0.3)

# 4. Equation and metrics
axes[1, 1].axis('off')
eq_text = f"Best Equation Found:\n\n{best_row['equation']}\n\n"
eq_text += f"Performance Metrics:\n"
eq_text += f"• R² = {best_row['r2']:.4f}\n"
eq_text += f"• MAE = {best_row['mae']:.4f}\n"
eq_text += f"• RMSE = {best_row['rmse']:.4f}\n"
eq_text += f"• Valid Predictions: {best_row['valid_pct']:.1f}%\n\n"
eq_text += f"Equation Complexity: {best_row['complexity']}"
axes[1, 1].text(0.1, 0.5, eq_text, fontsize=11, verticalalignment='center',
               fontfamily='monospace', 
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))

plt.suptitle('PySR Results: Best Equation for DIZ', fontsize=16, fontweight='bold')
plt.tight_layout()

# Save plot
plot_path = os.path.join(output_dir, "best_diz_equation.png")
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.show()
print(f"\n✅ Plot saved to: {plot_path}")

# ------------------------------
# 7. Summary statistics
# ------------------------------
print("\n" + "="*70)
print("FINAL SUMMARY")
print("="*70)
print(f"\n📊 Total equations analyzed: {len(results_df)}")
print(f"\n📈 Best R² achieved: {best_row['r2']:.4f}")
print(f"\n📉 R² distribution:")
print(f"   Mean R²: {results_df['r2'].mean():.4f}")
print(f"   Median R²: {results_df['r2'].median():.4f}")
print(f"   Std R²: {results_df['r2'].std():.4f}")
print(f"   Max R²: {results_df['r2'].max():.4f}")
print(f"   Min R²: {results_df['r2'].min():.4f}")

# Count equations with positive R²
positive_r2 = sum(results_df['r2'] > 0)
print(f"\n✅ Equations with positive R²: {positive_r2}/{len(results_df)} ({positive_r2/len(results_df)*100:.1f}%)")

# Save summary
summary_path = os.path.join(output_dir, "summary.txt")
with open(summary_path, 'w') as f:
    f.write("="*70 + "\n")
    f.write("FINAL PySR RESULTS FOR DIZ\n")
    f.write("="*70 + "\n\n")
    f.write(f"Total equations evaluated: {len(results_df)}\n")
    f.write(f"Equations with positive R²: {positive_r2}\n\n")
    f.write("BEST EQUATION:\n")
    f.write("-"*50 + "\n")
    f.write(f"R²: {best_row['r2']:.6f}\n")
    f.write(f"MAE: {best_row['mae']:.6f}\n")
    f.write(f"RMSE: {best_row['rmse']:.6f}\n")
    f.write(f"Complexity: {best_row['complexity']}\n")
    f.write(f"Source: {best_row['source']}\n\n")
    f.write(f"Equation:\n{best_row['equation']}\n")

print(f"\n✅ Summary saved to: {summary_path}")
print(f"\n✅ All results saved to: {output_dir}")
print("\n" + "="*70)
print("ANALYSIS COMPLETE")
print("="*70)