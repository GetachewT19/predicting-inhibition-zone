# ==============================
# PySR Symbolic Regression - All Features
# Target: DIZ
# ==============================

import pandas as pd
import numpy as np
from pysr import PySRRegressor
from sklearn.preprocessing import StandardScaler
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
output_dir = r"C:\Users\Amare\Desktop\pysr_all_features"
os.makedirs(output_dir, exist_ok=True)

# File paths for results
csv_path = os.path.join(output_dir, "all_equations.csv")
txt_path = os.path.join(output_dir, "best_equation.txt")
plot_path = os.path.join(output_dir, "analysis_plots.png")
summary_path = os.path.join(output_dir, "summary.txt")

# ------------------------------
# Load and prepare data
# ------------------------------
print("="*70)
print("PySR SYMBOLIC REGRESSION - ALL FEATURES")
print("="*70)

# Load data
df = pd.read_csv(data_path)
print(f"\n📊 Dataset shape: {df.shape}")
print(f"📋 Columns: {df.columns.tolist()}")

# Identify target
target = 'DIZ'
if target not in df.columns:
    print(f"\n❌ ERROR: Target column '{target}' not found!")
    print(f"Available columns: {df.columns.tolist()}")
    exit(1)

# All other columns are features
features = [col for col in df.columns if col != target]
print(f"\n✅ Target: {target}")
print(f"✅ Features ({len(features)}): {features}")

# ------------------------------
# Data cleaning
# ------------------------------
print("\n" + "-"*70)
print("DATA CLEANING")
print("-"*70)

# Work with a copy
df_clean = df.copy()

# Convert all columns to numeric, coercing errors to NaN
non_numeric_counts = {}
for col in [target] + features:
    original_non_numeric = df_clean[col].isna().sum()
    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    new_non_numeric = df_clean[col].isna().sum() - original_non_numeric
    
    if new_non_numeric > 0:
        non_numeric_counts[col] = new_non_numeric
        # Show examples of non-numeric values
        mask = df[col].apply(lambda x: not isinstance(x, (int, float)) or pd.isna(x))
        examples = df[col][mask].unique()[:3]
        print(f"  ⚠ Column '{col}': {new_non_numeric} non-numeric values")
        if len(examples) > 0:
            print(f"     Examples: {examples}")

# Drop rows with any NaN values
initial_rows = len(df_clean)
df_clean = df_clean.dropna()
rows_dropped = initial_rows - len(df_clean)

print(f"\nCleaning results:")
print(f"  Initial rows: {initial_rows}")
print(f"  Rows dropped: {rows_dropped}")
print(f"  Final rows: {len(df_clean)}")

if len(df_clean) == 0:
    print("\n❌ ERROR: No data left after cleaning!")
    exit(1)

# ------------------------------
# Prepare features and target
# ------------------------------
X = df_clean[features].values.astype(float)
y = df_clean[target].values.astype(float)

print(f"\nFinal data shape: X {X.shape}, y {y.shape}")

# ------------------------------
# Exploratory Data Analysis
# ------------------------------
print("\n" + "-"*70)
print("EXPLORATORY DATA ANALYSIS")
print("-"*70)

# Target statistics
print(f"\n📈 Target '{target}' statistics:")
print(f"  Mean:     {y.mean():.4f}")
print(f"  Std:      {y.std():.4f}")
print(f"  Min:      {y.min():.4f}")
print(f"  25%:      {np.percentile(y, 25):.4f}")
print(f"  50%:      {np.percentile(y, 50):.4f}")
print(f"  75%:      {np.percentile(y, 75):.4f}")
print(f"  Max:      {y.max():.4f}")

# Feature statistics and correlations with target
print(f"\n📊 Feature correlations with target:")
correlations = []
for i, feat in enumerate(features):
    corr = np.corrcoef(X[:, i], y)[0, 1]
    correlations.append((feat, corr))
    print(f"  {feat:15}: {corr:8.4f}")

# Sort by absolute correlation
correlations.sort(key=lambda x: abs(x[1]), reverse=True)
print(f"\n🔍 Top 5 features by absolute correlation:")
for feat, corr in correlations[:5]:
    print(f"  {feat:15}: {abs(corr):8.4f} (raw: {corr:8.4f})")

# ------------------------------
# Scale features
# ------------------------------
print("\n" + "-"*70)
print("FEATURE SCALING")
print("-"*70)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"Features scaled to mean≈0, std≈1")

# ------------------------------
# PySR Model Configuration - FIXED VERSION
# ------------------------------
print("\n" + "="*70)
print("PYSR MODEL CONFIGURATION")
print("="*70)
print(f"Target: {target}")
print(f"Features: {len(features)} variables")
print(f"Samples: {len(y)}")
print(f"Iterations: 1000")
print("="*70)

# FIXED: Removed equation_file and output_directory parameters
# Using only valid parameters for your PySR version
model = PySRRegressor(
    niterations=1000,
    populations=8,
    population_size=100,
    procs=4,
    multithreading=True,
    maxsize=25,
    parsimony=0.01,
    binary_operators=["+", "-", "*", "/", "^"],
    unary_operators=["exp", "log", "sqrt", "square", "cube", "sin", "cos"],
    loss="L2DistLoss()",
    denoise=True,
    progress=True,
    update_verbosity=1,
    warm_start=False,
    random_state=42,
    timeout_in_seconds=3600,  # 1 hour timeout
    temp_equation_file=True,  # This will save equations temporarily
    # Removed: equation_file, output_directory
)

# ------------------------------
# Fit model
# ------------------------------
print("\n🚀 Starting PySR search...")
print("This may take 30-60 minutes depending on complexity.")
print("Progress will be shown below:")
print("-"*70)

start_time = time.time()

try:
    model.fit(X_scaled, y)
    elapsed_time = time.time() - start_time
    print(f"\n✅ PySR completed in {elapsed_time/60:.1f} minutes!")
except Exception as e:
    print(f"\n❌ PySR failed: {e}")
    
    # Try with simpler configuration
    print("\n🔄 Trying with simpler configuration...")
    model = PySRRegressor(
        niterations=500,
        populations=5,
        population_size=80,
        procs=4,
        maxsize=15,
        parsimony=0.02,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["exp", "log", "sqrt"],
        loss="L2DistLoss()",
        progress=True,
        random_state=42,
        temp_equation_file=True
    )
    model.fit(X_scaled, y)
    print("✅ Simplified model completed!")

# ------------------------------
# Results
# ------------------------------
print("\n" + "="*70)
print("RESULTS")
print("="*70)

if hasattr(model, 'equations_') and model.equations_ is not None:
    equations = model.equations_
    print(f"\n📊 Found {len(equations)} equations")
    
    # Display top 10 equations
    print("\n🏆 TOP 10 EQUATIONS (lowest loss):")
    equations_sorted = equations.sort_values('loss').head(10)
    for i, (idx, row) in enumerate(equations_sorted.iterrows()):
        print(f"\n  Rank {i+1}:")
        print(f"    Complexity: {row['complexity']}")
        print(f"    Loss: {row['loss']:.6f}")
        if 'score' in row:
            print(f"    Score: {row['score']:.4f}")
        print(f"    Equation: {row['equation']}")
    
    # Best equation
    best_idx = equations['loss'].idxmin()
    best_eq = equations.loc[best_idx, 'equation']
    best_loss = equations.loc[best_idx, 'loss']
    best_complexity = equations.loc[best_idx, 'complexity']
    
    print("\n" + "⭐"*70)
    print("🌟 BEST EQUATION")
    print("⭐"*70)
    print(f"Equation: {best_eq}")
    print(f"Loss: {best_loss:.6f}")
    print(f"Complexity: {best_complexity}")
    
    # Save all equations to CSV
    equations.to_csv(csv_path, index=False)
    print(f"\n✅ All equations saved: {csv_path}")
    
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
    print(f"R² Score:      {r2:.6f} ({r2*100:.2f}% variance explained)")
    print(f"Adjusted R²:   {adj_r2:.6f}")
    print(f"MAE:           {mae:.6f}")
    print(f"RMSE:          {rmse:.6f}")
    print(f"MAPE:          {mape:.2f}%")
    print(f"RMSE/Mean:     {rmse/y.mean():.3f}")
    
    # Save best equation and metrics
    with open(txt_path, "w") as f:
        f.write("="*70 + "\n")
        f.write("BEST SYMBOLIC EQUATION\n")
        f.write("="*70 + "\n\n")
        f.write(f"Target: {target}\n")
        f.write(f"Features: {len(features)} variables\n")
        f.write(f"Samples: {len(y)}\n\n")
        f.write(f"Equation: {best_eq}\n")
        f.write(f"Loss: {best_loss:.6f}\n")
        f.write(f"Complexity: {best_complexity}\n\n")
        
        try:
            f.write(f"SymPy format: {model.sympy()}\n\n")
        except:
            pass
        
        f.write("-"*50 + "\n")
        f.write("PERFORMANCE METRICS\n")
        f.write("-"*50 + "\n")
        f.write(f"R² Score:      {r2:.6f} ({r2*100:.2f}%)\n")
        f.write(f"Adjusted R²:   {adj_r2:.6f}\n")
        f.write(f"MAE:           {mae:.6f}\n")
        f.write(f"RMSE:          {rmse:.6f}\n")
        f.write(f"MAPE:          {mape:.2f}%\n")
        f.write(f"RMSE/Mean:     {rmse/y.mean():.3f}\n")
    
    print(f"\n✅ Best equation saved: {txt_path}")
    
    # ------------------------------
    # Visualization
    # ------------------------------
    residuals = y - y_pred
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Predicted vs Actual
    axes[0, 0].scatter(y, y_pred, alpha=0.6, c='blue', edgecolors='black', s=50)
    axes[0, 0].plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2, label='Perfect Fit')
    axes[0, 0].set_xlabel('Actual DIZ')
    axes[0, 0].set_ylabel('Predicted DIZ')
    axes[0, 0].set_title(f'Predicted vs Actual (R² = {r2:.4f})')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Residuals
    axes[0, 1].scatter(y_pred, residuals, alpha=0.6, c='green', edgecolors='black', s=50)
    axes[0, 1].axhline(0, color='r', linestyle='--', lw=2)
    axes[0, 1].set_xlabel('Predicted DIZ')
    axes[0, 1].set_ylabel('Residuals')
    axes[0, 1].set_title('Residual Plot')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Residuals distribution
    axes[0, 2].hist(residuals, bins=20, edgecolor='black', alpha=0.7, color='purple')
    axes[0, 2].axvline(0, color='r', linestyle='--', lw=2)
    axes[0, 2].set_xlabel('Residuals')
    axes[0, 2].set_ylabel('Frequency')
    axes[0, 2].set_title('Residuals Distribution')
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Complexity vs Loss
    axes[1, 0].scatter(equations['complexity'], equations['loss'], alpha=0.6, c='orange', s=50)
    axes[1, 0].set_xlabel('Complexity')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].set_title('Complexity vs Loss')
    axes[1, 0].set_yscale('log')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].scatter(best_complexity, best_loss, c='red', s=200, marker='*', 
                       label=f'Best (C={best_complexity})')
    axes[1, 0].legend()
    
    # 5. Q-Q plot
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title('Q-Q Plot (Normality Check)')
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. Feature importance (correlation-based)
    feat_importance = [abs(c) for _, c in correlations[:10]]  # Top 10
    feat_names = [f for f, _ in correlations[:10]]
    axes[1, 2].barh(range(len(feat_importance)), feat_importance, color='skyblue', edgecolor='black')
    axes[1, 2].set_yticks(range(len(feat_importance)))
    axes[1, 2].set_yticklabels(feat_names)
    axes[1, 2].set_xlabel('Absolute Correlation with DIZ')
    axes[1, 2].set_title('Top 10 Feature Correlations')
    axes[1, 2].invert_yaxis()
    axes[1, 2].grid(True, alpha=0.3, axis='x')
    
    plt.suptitle(f'PySR Results: DIZ with {len(features)} Features\nBest Eq Complexity: {best_complexity}', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"\n✅ Plots saved: {plot_path}")
    
    # ------------------------------
    # Summary
    # ------------------------------
    with open(summary_path, "w") as f:
        f.write("="*70 + "\n")
        f.write("PYSR SYMBOLIC REGRESSION SUMMARY\n")
        f.write("="*70 + "\n\n")
        f.write(f"Dataset: {data_path}\n")
        f.write(f"Target: {target}\n")
        f.write(f"Features: {len(features)}\n")
        f.write(f"Samples after cleaning: {len(y)}\n\n")
        f.write("TOP 5 FEATURES BY CORRELATION:\n")
        for feat, corr in correlations[:5]:
            f.write(f"  {feat}: {corr:.4f}\n")
        f.write("\n" + "="*70 + "\n")
        f.write("BEST EQUATION\n")
        f.write("="*70 + "\n")
        f.write(f"{best_eq}\n\n")
        f.write(f"Loss: {best_loss:.6f}\n")
        f.write(f"Complexity: {best_complexity}\n\n")
        f.write("PERFORMANCE METRICS:\n")
        f.write(f"  R²: {r2:.4f}\n")
        f.write(f"  Adjusted R²: {adj_r2:.4f}\n")
        f.write(f"  MAE: {mae:.4f}\n")
        f.write(f"  RMSE: {rmse:.4f}\n")
        f.write(f"  MAPE: {mape:.2f}%\n")
    
    print(f"\n✅ Summary saved: {summary_path}")
    
else:
    print("\n❌ No equations found!")

print("\n" + "="*70)
print("✅ ANALYSIS COMPLETE")
print("="*70)
print(f"\nAll outputs saved to: {output_dir}")
print(f"  - All equations: {csv_path}")
print(f"  - Best equation: {txt_path}")
print(f"  - Plots: {plot_path}")
print(f"  - Summary: {summary_path}")
print("\n" + "="*70)