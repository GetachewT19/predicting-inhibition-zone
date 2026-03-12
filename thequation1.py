# ==============================
# Optimized PySR Script for DIZ
# ==============================

import pandas as pd
import numpy as np
from pysr import PySRRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import warnings
import os
import time

warnings.filterwarnings('ignore')

# ------------------------------
# Configuration
# ------------------------------
data_path = r"C:\Users\Amare\Desktop\worku\datato232-81.csv"
output_dir = r"C:\Users\Amare\Desktop\pysr_optimized"
os.makedirs(output_dir, exist_ok=True)

# File paths
csv_path = os.path.join(output_dir, "all_equations.csv")
txt_path = os.path.join(output_dir, "best_equation.txt")
plot_path = os.path.join(output_dir, "analysis.png")

print("="*70)
print("OPTIMIZED PySR FOR DIZ")
print("="*70)

# ------------------------------
# Load and clean data
# ------------------------------
df = pd.read_csv(data_path)
print(f"\n📊 Dataset shape: {df.shape}")

target = 'DIZ'
features = [col for col in df.columns if col != target]

print(f"✅ Target: {target}")
print(f"✅ Features: {features}")

# Clean data
df_clean = df.copy()
for col in [target] + features:
    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

initial_rows = len(df_clean)
df_clean = df_clean.dropna()
print(f"\nData cleaning:")
print(f"  Rows dropped: {initial_rows - len(df_clean)}")
print(f"  Final rows: {len(df_clean)}")

X = df_clean[features].values.astype(float)
y = df_clean[target].values.astype(float)

# ------------------------------
# Feature correlations
# ------------------------------
print("\n" + "-"*50)
print("FEATURE CORRELATIONS WITH DIZ")
print("-"*50)

correlations = []
for i, feat in enumerate(features):
    corr = np.corrcoef(X[:, i], y)[0, 1]
    correlations.append((feat, corr))
    print(f"  {feat:15}: {corr:8.4f}")

# Sort by absolute correlation
correlations.sort(key=lambda x: abs(x[1]), reverse=True)
print(f"\n🎯 Top 3 features:")
for feat, corr in correlations[:3]:
    print(f"  {feat}: {corr:.4f} (|corr| = {abs(corr):.4f})")

# ------------------------------
# Scale features
# ------------------------------
print("\n" + "-"*50)
print("FEATURE SCALING")
print("-"*50)

# Use RobustScaler for better handling of outliers
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)
print("✅ Features scaled with RobustScaler")

# ------------------------------
# PySR Configuration
# ------------------------------
print("\n" + "="*70)
print("PYSR CONFIGURATION")
print("="*70)

# Based on diagnostic, use configurations that worked
model = PySRRegressor(
    niterations=1000,           # Increased iterations
    populations=8,              # Multiple populations
    population_size=100,        # Good population size
    procs=4,                    # Use 4 processes
    maxsize=15,                  # Good starting complexity
    parsimony=0.01,              # Penalty for complexity
    binary_operators=["+", "-", "*", "/", "^"],
    unary_operators=["sqrt", "square", "exp", "log", "sin", "cos"],
    loss="L2DistLoss()",
    denoise=True,
    progress=True,
    update_verbosity=1,
    random_state=42,
    timeout_in_seconds=3600,     # 1 hour timeout
    temp_equation_file=True,
    warm_start=False
)

print(f"\nConfiguration:")
print(f"  Iterations: {model.niterations}")
print(f"  Populations: {model.populations}")
print(f"  Population size: {model.population_size}")
print(f"  Max complexity: {model.maxsize}")
print(f"  Binary ops: {model.binary_operators}")
print(f"  Unary ops: {model.unary_operators}")

# ------------------------------
# Fit model
# ------------------------------
print("\n" + "="*70)
print("TRAINING PySR")
print("="*70)
print("⏳ This may take 30-60 minutes...")
print("-"*50)

start_time = time.time()

try:
    model.fit(X_scaled, y)
    elapsed = time.time() - start_time
    print(f"\n✅ Training completed in {elapsed/60:.1f} minutes!")
except Exception as e:
    print(f"\n❌ Training failed: {e}")
    exit(1)

# ------------------------------
# Results
# ------------------------------
print("\n" + "="*70)
print("RESULTS")
print("="*70)

if hasattr(model, 'equations_') and model.equations_ is not None:
    equations = model.equations_
    print(f"\n📊 Found {len(equations)} equations")
    
    # Save all equations
    equations.to_csv(csv_path, index=False)
    print(f"✅ All equations saved to: {csv_path}")
    
    # Top 10 equations
    print("\n🏆 TOP 10 EQUATIONS:")
    top10 = equations.sort_values('loss').head(10)
    for i, (idx, row) in enumerate(top10.iterrows()):
        print(f"\n  Rank {i+1}:")
        print(f"    Complexity: {row['complexity']}")
        print(f"    Loss: {row['loss']:.6f}")
        print(f"    Equation: {row['equation'][:80]}...")
    
    # Best equation
    best_idx = equations['loss'].idxmin()
    best_eq = equations.loc[best_idx, 'equation']
    best_loss = equations.loc[best_idx, 'loss']
    best_complexity = equations.loc[best_idx, 'complexity']
    
    print("\n" + "⭐"*70)
    print("🌟 BEST EQUATION")
    print("⭐"*70)
    print(f"Complexity: {best_complexity}")
    print(f"Loss: {best_loss:.6f}")
    print(f"\nEquation: {best_eq}")
    
    # Try to get sympy format
    try:
        sympy_eq = model.sympy()
        print(f"\nSymPy format: {sympy_eq}")
    except:
        pass
    
    # ------------------------------
    # Predictions and metrics
    # ------------------------------
    y_pred = model.predict(X_scaled)
    
    # Calculate metrics
    r2 = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mape = np.mean(np.abs((y - y_pred) / (y + 1e-10))) * 100
    
    # Adjusted R²
    n = len(y)
    p = X.shape[1]
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1) if n > p + 1 else r2
    
    print("\n" + "="*70)
    print("PERFORMANCE METRICS")
    print("="*70)
    print(f"R² Score:      {r2:.6f} ({r2*100:.2f}%)")
    print(f"Adjusted R²:   {adj_r2:.6f}")
    print(f"MAE:           {mae:.6f}")
    print(f"RMSE:          {rmse:.6f}")
    print(f"MAPE:          {mape:.2f}%")
    
    # Save best equation with metrics
    with open(txt_path, "w") as f:
        f.write("="*70 + "\n")
        f.write("BEST EQUATION FOR DIZ\n")
        f.write("="*70 + "\n\n")
        f.write(f"Equation: {best_eq}\n\n")
        f.write(f"Loss: {best_loss:.6f}\n")
        f.write(f"Complexity: {best_complexity}\n\n")
        f.write("-"*50 + "\n")
        f.write("PERFORMANCE METRICS\n")
        f.write("-"*50 + "\n")
        f.write(f"R² Score:  {r2:.6f}\n")
        f.write(f"Adjusted R²: {adj_r2:.6f}\n")
        f.write(f"MAE:       {mae:.6f}\n")
        f.write(f"RMSE:      {rmse:.6f}\n")
        f.write(f"MAPE:      {mape:.2f}%\n\n")
        
        try:
            f.write(f"SymPy format: {model.sympy()}\n")
        except:
            pass
    
    print(f"\n✅ Best equation saved to: {txt_path}")
    
    # ------------------------------
    # Visualization
    # ------------------------------
    residuals = y - y_pred
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. Predicted vs Actual
    axes[0, 0].scatter(y, y_pred, alpha=0.6, c='blue', edgecolors='black', s=60)
    axes[0, 0].plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
    axes[0, 0].set_xlabel('Actual DIZ')
    axes[0, 0].set_ylabel('Predicted DIZ')
    axes[0, 0].set_title(f'Predicted vs Actual (R² = {r2:.4f})')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Residuals
    axes[0, 1].scatter(y_pred, residuals, alpha=0.6, c='green', edgecolors='black', s=60)
    axes[0, 1].axhline(0, color='r', linestyle='--', lw=2)
    axes[0, 1].set_xlabel('Predicted DIZ')
    axes[0, 1].set_ylabel('Residuals')
    axes[0, 1].set_title('Residual Plot')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Residuals distribution
    axes[1, 0].hist(residuals, bins=20, edgecolor='black', alpha=0.7, color='purple')
    axes[1, 0].axvline(0, color='r', linestyle='--', lw=2)
    axes[1, 0].set_xlabel('Residuals')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Residuals Distribution')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Complexity vs Loss
    axes[1, 1].scatter(equations['complexity'], equations['loss'], alpha=0.6, c='orange', s=60)
    axes[1, 1].set_xlabel('Complexity')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].set_title('Complexity vs Loss')
    axes[1, 1].set_yscale('log')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].scatter(best_complexity, best_loss, c='red', s=200, marker='*')
    
    plt.suptitle(f'PySR Results: DIZ Prediction\nBest Equation Complexity: {best_complexity}', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"\n✅ Plot saved to: {plot_path}")
    
else:
    print("\n❌ No equations found!")

print("\n" + "="*70)
print("✅ ANALYSIS COMPLETE")
print("="*70)
print(f"\nAll outputs saved to: {output_dir}")
print(f"  - All equations: {csv_path}")
print(f"  - Best equation: {txt_path}")
print(f"  - Plot: {plot_path}")