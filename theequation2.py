# ==============================
# Comprehensive Analysis of All PySR Results
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
output_dir = r"C:\Users\Amare\Desktop\pysr_final_results"
os.makedirs(output_dir, exist_ok=True)

print("="*70)
print("COMPREHENSIVE ANALYSIS OF ALL PySR RESULTS")
print("="*70)

# ------------------------------
# 1. Load all result files
# ------------------------------
csv_files = glob.glob(os.path.join(results_dir, "*.csv"))
print(f"\n📊 Found {len(csv_files)} result files:")

all_results = []
for csv_file in csv_files:
    filename = os.path.basename(csv_file)
    try:
        df = pd.read_csv(csv_file)
        if 'loss' in df.columns and 'equation' in df.columns:
            # Add filename as column
            df['source_file'] = filename
            all_results.append(df)
            print(f"  ✅ {filename}: {len(df)} equations")
        else:
            print(f"  ⚠ {filename}: No equation data")
    except Exception as e:
        print(f"  ❌ {filename}: Error reading - {e}")

if not all_results:
    print("\n❌ No valid equation files found!")
    exit(1)

# Combine all results
combined_df = pd.concat(all_results, ignore_index=True)
print(f"\n📊 Combined results: {len(combined_df)} total equations")

# ------------------------------
# 2. Find best equations overall
# ------------------------------
print("\n" + "="*70)
print("BEST EQUATIONS OVERALL (by loss)")
print("="*70)

# Sort by loss
combined_sorted = combined_df.sort_values('loss').head(20)

for i, (idx, row) in enumerate(combined_sorted.iterrows()):
    print(f"\n{i+1}. Loss: {row['loss']:.6f}")
    print(f"   Complexity: {row['complexity']}")
    print(f"   Source: {row['source_file']}")
    print(f"   Equation: {row['equation']}")

# Save top 100 equations
top100 = combined_df.sort_values('loss').head(100)
top100_path = os.path.join(output_dir, "top_100_equations.csv")
top100.to_csv(top100_path, index=False)
print(f"\n✅ Top 100 equations saved to: {top100_path}")

# Best equation
best_row = combined_sorted.iloc[0]
best_eq = best_row['equation']
best_loss = best_row['loss']
best_complexity = best_row['complexity']
best_source = best_row['source_file']

print("\n" + "⭐"*70)
print("🌟 BEST EQUATION OVERALL")
print("⭐"*70)
print(f"Source file: {best_source}")
print(f"Complexity: {best_complexity}")
print(f"Loss: {best_loss:.6f}")
print(f"\nEquation: {best_eq}")

# Save best equation
best_txt = os.path.join(output_dir, "best_equation.txt")
with open(best_txt, 'w') as f:
    f.write("="*70 + "\n")
    f.write("BEST EQUATION FOR DIZ\n")
    f.write("="*70 + "\n\n")
    f.write(f"Source: {best_source}\n")
    f.write(f"Complexity: {best_complexity}\n")
    f.write(f"Loss: {best_loss:.6f}\n\n")
    f.write(f"Equation: {best_eq}\n")
print(f"\n✅ Best equation saved to: {best_txt}")

# ------------------------------
# 3. Load and prepare original data for evaluation
# ------------------------------
print("\n" + "="*70)
print("LOADING ORIGINAL DATA FOR EVALUATION")
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

# ------------------------------
# 4. Evaluate top equations
# ------------------------------
print("\n" + "="*70)
print("EVALUATING TOP 10 EQUATIONS")
print("="*70)

def safe_eval_equation(expr, x_values):
    """
    Safely evaluate a mathematical expression with variables x0, x1, x2,...
    """
    import math
    import numpy as np
    
    # Create safe namespace
    safe_dict = {
        'sin': math.sin, 'cos': math.cos, 'tan': math.tan,
        'exp': math.exp, 'log': math.log, 'log10': math.log10,
        'sqrt': math.sqrt, 'abs': abs,
        'square': lambda x: x**2, 'cube': lambda x: x**3,
        'pow': pow, 'pi': math.pi, 'e': math.e,
        'np': np
    }
    
    # Add variables x0, x1, x2,...
    for i, val in enumerate(x_values):
        safe_dict[f'x{i}'] = val
    
    try:
        # Replace ^ with **
        expr = expr.replace('^', '**')
        result = eval(expr, {"__builtins__": {}}, safe_dict)
        return float(result) if isinstance(result, (int, float)) else np.nan
    except:
        return np.nan

# Evaluate top 10 equations
top10 = combined_sorted.head(10)
results = []

for idx, row in top10.iterrows():
    expr = row['equation']
    print(f"\nEvaluating: {expr[:80]}...")
    
    predictions = []
    valid_count = 0
    
    for i in range(len(y)):
        pred = safe_eval_equation(expr, X_scaled[i])
        if not np.isnan(pred):
            predictions.append(pred)
            valid_count += 1
        else:
            predictions.append(np.nan)
    
    predictions = np.array(predictions)
    valid_mask = ~np.isnan(predictions)
    
    if valid_count > 10:  # Need at least some valid predictions
        y_valid = y[valid_mask]
        pred_valid = predictions[valid_mask]
        
        r2 = r2_score(y_valid, pred_valid)
        mae = mean_absolute_error(y_valid, pred_valid)
        rmse = np.sqrt(mean_squared_error(y_valid, pred_valid))
        
        results.append({
            'equation': expr,
            'loss': row['loss'],
            'complexity': row['complexity'],
            'r2': r2,
            'mae': mae,
            'rmse': rmse,
            'valid_pct': valid_count/len(y)*100
        })
        
        print(f"  ✅ R² = {r2:.4f}, MAE = {mae:.4f}, Valid: {valid_count}/{len(y)}")
    else:
        print(f"  ❌ Too few valid predictions ({valid_count}/{len(y)})")

# ------------------------------
# 5. Best equation by R²
# ------------------------------
if results:
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('r2', ascending=False)
    
    print("\n" + "="*70)
    print("BEST EQUATION BY R² SCORE")
    print("="*70)
    
    best_r2_row = results_df.iloc[0]
    print(f"\nR² Score: {best_r2_row['r2']:.4f}")
    print(f"MAE: {best_r2_row['mae']:.4f}")
    print(f"RMSE: {best_r2_row['rmse']:.4f}")
    print(f"Valid predictions: {best_r2_row['valid_pct']:.1f}%")
    print(f"\nEquation: {best_r2_row['equation']}")
    
    # Save best R² equation
    best_r2_txt = os.path.join(output_dir, "best_r2_equation.txt")
    with open(best_r2_txt, 'w') as f:
        f.write("="*70 + "\n")
        f.write("BEST EQUATION BY R² SCORE\n")
        f.write("="*70 + "\n\n")
        f.write(f"R² Score: {best_r2_row['r2']:.6f}\n")
        f.write(f"MAE: {best_r2_row['mae']:.6f}\n")
        f.write(f"RMSE: {best_r2_row['rmse']:.6f}\n")
        f.write(f"Valid predictions: {best_r2_row['valid_pct']:.1f}%\n\n")
        f.write(f"Original loss: {best_r2_row['loss']:.6f}\n")
        f.write(f"Complexity: {best_r2_row['complexity']}\n\n")
        f.write(f"Equation: {best_r2_row['equation']}\n")
    
    print(f"\n✅ Best R² equation saved to: {best_r2_txt}")
    
    # ------------------------------
    # 6. Visualization of best equation
    # ------------------------------
    print("\n" + "="*70)
    print("VISUALIZING BEST EQUATION")
    print("="*70)
    
    # Get predictions for best equation
    best_expr = best_r2_row['equation']
    predictions = []
    for i in range(len(y)):
        pred = safe_eval_equation(best_expr, X_scaled[i])
        predictions.append(pred)
    
    predictions = np.array(predictions)
    valid_mask = ~np.isnan(predictions)
    y_valid = y[valid_mask]
    pred_valid = predictions[valid_mask]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. Predicted vs Actual
    axes[0, 0].scatter(y_valid, pred_valid, alpha=0.6, c='blue', edgecolors='black', s=50)
    axes[0, 0].plot([y_valid.min(), y_valid.max()], [y_valid.min(), y_valid.max()], 'r--', lw=2)
    axes[0, 0].set_xlabel('Actual DIZ')
    axes[0, 0].set_ylabel('Predicted DIZ')
    axes[0, 0].set_title(f'Predicted vs Actual (R² = {best_r2_row["r2"]:.4f})')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Residuals
    residuals = y_valid - pred_valid
    axes[0, 1].scatter(pred_valid, residuals, alpha=0.6, c='green', edgecolors='black', s=50)
    axes[0, 1].axhline(0, color='r', linestyle='--', lw=2)
    axes[0, 1].set_xlabel('Predicted DIZ')
    axes[0, 1].set_ylabel('Residuals')
    axes[0, 1].set_title(f'Residual Plot (σ = {np.std(residuals):.4f})')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Residuals distribution
    axes[1, 0].hist(residuals, bins=20, edgecolor='black', alpha=0.7, color='purple')
    axes[1, 0].axvline(0, color='r', linestyle='--', lw=2)
    axes[1, 0].set_xlabel('Residuals')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Residuals Distribution')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Equation text
    axes[1, 1].axis('off')
    eq_text = f"Best Equation:\n\n{best_expr}\n\n"
    eq_text += f"R² = {best_r2_row['r2']:.4f}\n"
    eq_text += f"MAE = {best_r2_row['mae']:.4f}\n"
    eq_text += f"RMSE = {best_r2_row['rmse']:.4f}"
    axes[1, 1].text(0.1, 0.5, eq_text, fontsize=12, verticalalignment='center',
                   fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.suptitle('Best PySR Equation Performance', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, "best_equation_analysis.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"\n✅ Plot saved to: {plot_path}")
    
    # ------------------------------
    # 7. Summary report
    # ------------------------------
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    print(f"\n📊 Total equations analyzed: {len(combined_df)}")
    print(f"\n🌟 Best equation by loss:")
    print(f"   Loss: {best_loss:.6f}")
    print(f"   Complexity: {best_complexity}")
    print(f"   Source: {best_source}")
    print(f"\n📈 Best equation by R² score:")
    print(f"   R²: {best_r2_row['r2']:.4f}")
    print(f"   MAE: {best_r2_row['mae']:.4f}")
    print(f"   RMSE: {best_r2_row['rmse']:.4f}")
    print(f"\n✅ All results saved to: {output_dir}")
    
    # Create summary file
    summary_path = os.path.join(output_dir, "summary.txt")
    with open(summary_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("PySR RESULTS SUMMARY\n")
        f.write("="*70 + "\n\n")
        f.write(f"Total equations found: {len(combined_df)}\n")
        f.write(f"Result files analyzed: {len(csv_files)}\n\n")
        f.write("BEST EQUATION BY LOSS:\n")
        f.write("-"*50 + "\n")
        f.write(f"Loss: {best_loss:.6f}\n")
        f.write(f"Complexity: {best_complexity}\n")
        f.write(f"Source: {best_source}\n")
        f.write(f"Equation: {best_eq}\n\n")
        f.write("BEST EQUATION BY R²:\n")
        f.write("-"*50 + "\n")
        f.write(f"R²: {best_r2_row['r2']:.6f}\n")
        f.write(f"MAE: {best_r2_row['mae']:.6f}\n")
        f.write(f"RMSE: {best_r2_row['rmse']:.6f}\n")
        f.write(f"Valid predictions: {best_r2_row['valid_pct']:.1f}%\n")
        f.write(f"Equation: {best_r2_row['equation']}\n")
    
    print(f"\n✅ Summary saved to: {summary_path}")

else:
    print("\n❌ No equations could be evaluated successfully!")

print("\n" + "="*70)
print("ANALYSIS COMPLETE")
print("="*70)