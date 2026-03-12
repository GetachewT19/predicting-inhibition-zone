# ==============================
# Continue PySR on Clean Data
# ==============================

import pandas as pd
import numpy as np
from pysr import PySRRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import warnings
import os
import traceback

warnings.filterwarnings('ignore')

# ------------------------------
# Configuration
# ------------------------------
data_path = r"C:\Users\Amare\Desktop\worku\datato232-81.csv"
output_dir = r"C:\Users\Amare\Desktop\pysr_cleaned_data"
os.makedirs(output_dir, exist_ok=True)

print("="*70)
print("CONTINUING PySR ON CLEAN DATA")
print("="*70)

# ------------------------------
# Reload and clean data
# ------------------------------
df = pd.read_csv(data_path)
target = 'DIZ'
features = ['Cbac', 'CAgNP', 'Ra']

# Basic cleaning
df_clean = df.copy()
for col in [target] + features:
    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
df_clean = df_clean.dropna()

X = df_clean[features].values.astype(float)
y = df_clean[target].values.astype(float)

# Outlier detection (same as before)
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

# Statistical outliers
z_scores = np.abs(stats.zscore(y))
z_score_outliers = np.where(z_scores > 3)[0]

Q1, Q3 = np.percentile(y, 25), np.percentile(y, 75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
iqr_outliers = np.where((y < lower_bound) | (y > upper_bound))[0]

median_y = np.median(y)
mad_y = np.median(np.abs(y - median_y))
if mad_y > 0:
    modified_z_scores = 0.6745 * (y - median_y) / mad_y
    mod_z_score_outliers = np.where(np.abs(modified_z_scores) > 3.5)[0]
else:
    mod_z_score_outliers = []

# Multivariate outliers
Xy = np.column_stack([X, y])
iso_forest = IsolationForest(contamination=0.1, random_state=42)
iso_labels = iso_forest.fit_predict(Xy)
iso_outliers = np.where(iso_labels == -1)[0]

lof = LocalOutlierFactor(contamination=0.1)
lof_labels = lof.fit_predict(Xy)
lof_outliers = np.where(lof_labels == -1)[0]

# Combine outliers
outlier_mask = np.zeros(len(y), dtype=bool)
outlier_mask[z_score_outliers] = True
outlier_mask[iqr_outliers] = True
outlier_mask[mod_z_score_outliers] = True
outlier_mask[iso_outliers] = True
outlier_mask[lof_outliers] = True

# Create clean dataset
X_clean = X[~outlier_mask]
y_clean = y[~outlier_mask]

print(f"\nClean data shape: {X_clean.shape}")
print(f"Target statistics after cleaning:")
print(f"  Mean: {y_clean.mean():.4f}")
print(f"  Std: {y_clean.std():.4f}")
print(f"  Min: {y_clean.min():.4f}")
print(f"  Max: {y_clean.max():.4f}")

# Scale data
scaler = RobustScaler()
X_clean_scaled = scaler.fit_transform(X_clean)

# ------------------------------
# Run PySR with simpler configuration
# ------------------------------
print("\n" + "="*70)
print("RUNNING PySR ON CLEAN DATA (Simplified)")
print("="*70)

# Use a simpler configuration that's more likely to work
model = PySRRegressor(
    niterations=500,  # Fewer iterations
    populations=5,    # Fewer populations
    population_size=50,
    maxsize=15,
    parsimony=0.01,
    binary_operators=["+", "-", "*", "/"],
    unary_operators=["sqrt", "square"],
    procs=2,
    loss="L2DistLoss()",
    progress=True,
    random_state=42,
    temp_equation_file=True,
    warm_start=False,
    timeout_in_seconds=1800  # 30 minutes timeout
)

try:
    print("\nFitting model...")
    model.fit(X_clean_scaled, y_clean)
    print("✅ Model fitting completed!")
    
    if hasattr(model, 'equations_') and model.equations_ is not None:
        equations = model.equations_
        print(f"\n✅ Found {len(equations)} equations")
        
        # Save equations
        csv_path = os.path.join(output_dir, "clean_data_equations.csv")
        equations.to_csv(csv_path, index=False)
        print(f"✅ Equations saved to: {csv_path}")
        
        # Display top equations
        print("\n🏆 TOP 10 EQUATIONS:")
        top10 = equations.sort_values('loss').head(10)
        for i, (idx, row) in enumerate(top10.iterrows()):
            print(f"\n  {i+1}. Loss: {row['loss']:.6f}")
            print(f"     Complexity: {row['complexity']}")
            print(f"     Equation: {row['equation']}")
        
        # Best equation
        best_idx = equations['loss'].idxmin()
        best_eq = equations.loc[best_idx, 'equation']
        best_loss = equations.loc[best_idx, 'loss']
        best_complexity = equations.loc[best_idx, 'complexity']
        
        print("\n" + "⭐"*70)
        print("🌟 BEST EQUATION ON CLEAN DATA")
        print("⭐"*70)
        print(f"Loss: {best_loss:.6f}")
        print(f"Complexity: {best_complexity}")
        print(f"Equation: {best_eq}")
        print(f"Contains bacteria: {'x0' in best_eq}")
        
        # Make predictions
        y_pred = model.predict(X_clean_scaled)
        r2 = r2_score(y_clean, y_pred)
        mae = mean_absolute_error(y_clean, y_pred)
        rmse = np.sqrt(mean_squared_error(y_clean, y_pred))
        
        print(f"\n📈 Performance Metrics:")
        print(f"  R² = {r2:.4f}")
        print(f"  MAE = {mae:.4f}")
        print(f"  RMSE = {rmse:.4f}")
        
        # Save best equation
        txt_path = os.path.join(output_dir, "clean_data_best_equation.txt")
        with open(txt_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("BEST EQUATION ON CLEAN DATA\n")
            f.write("="*70 + "\n\n")
            f.write(f"Samples after cleaning: {len(y_clean)}\n")
            f.write(f"Outliers removed: {len(y) - len(y_clean)}\n\n")
            f.write(f"R² Score: {r2:.6f}\n")
            f.write(f"MAE: {mae:.6f}\n")
            f.write(f"RMSE: {rmse:.6f}\n")
            f.write(f"Loss: {best_loss:.6f}\n")
            f.write(f"Complexity: {best_complexity}\n\n")
            f.write(f"Equation: {best_eq}\n")
        
        print(f"\n✅ Best equation saved to: {txt_path}")
        
        # Visualization
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Predicted vs Actual
        axes[0, 0].scatter(y_clean, y_pred, alpha=0.6, c='blue', edgecolors='black', s=60)
        axes[0, 0].plot([y_clean.min(), y_clean.max()], [y_clean.min(), y_clean.max()], 
                        'r--', lw=2)
        axes[0, 0].set_xlabel('Actual DIZ')
        axes[0, 0].set_ylabel('Predicted DIZ')
        axes[0, 0].set_title(f'Predicted vs Actual (R² = {r2:.4f})')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Residuals
        residuals = y_clean - y_pred
        axes[0, 1].scatter(y_pred, residuals, alpha=0.6, c='green', edgecolors='black', s=60)
        axes[0, 1].axhline(0, color='r', linestyle='--', lw=2)
        axes[0, 1].set_xlabel('Predicted DIZ')
        axes[0, 1].set_ylabel('Residuals')
        axes[0, 1].set_title(f'Residuals (σ = {np.std(residuals):.4f})')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Complexity vs Loss
        axes[1, 0].scatter(equations['complexity'], equations['loss'], alpha=0.6, c='orange', s=60)
        axes[1, 0].set_xlabel('Complexity')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].set_title('Complexity vs Loss')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].scatter(best_complexity, best_loss, c='red', s=200, marker='*')
        
        # Equation text
        axes[1, 1].axis('off')
        eq_text = f"Best Equation:\n\n{best_eq}\n\n"
        eq_text += f"R² = {r2:.4f}\n"
        eq_text += f"MAE = {mae:.4f}\n"
        eq_text += f"RMSE = {rmse:.4f}\n"
        eq_text += f"Samples: {len(y_clean)}"
        axes[1, 1].text(0.1, 0.5, eq_text, fontsize=11, verticalalignment='center',
                       fontfamily='monospace', 
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))
        
        plt.suptitle('PySR Results on Clean Data', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        plot_path = os.path.join(output_dir, "clean_data_results.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"\n✅ Plot saved to: {plot_path}")
        
    else:
        print("❌ No equations found in model")
        
except Exception as e:
    print(f"\n❌ Error during PySR fitting: {e}")
    traceback.print_exc()
    
    # Try even simpler configuration
    print("\n" + "="*70)
    print("ATTEMPTING ULTRA-SIMPLE CONFIGURATION")
    print("="*70)
    
    model_simple = PySRRegressor(
        niterations=200,
        populations=3,
        population_size=30,
        maxsize=10,
        binary_operators=["+", "*"],
        unary_operators=["square"],
        procs=1,
        loss="L2DistLoss()",
        random_state=42
    )
    
    try:
        model_simple.fit(X_clean_scaled, y_clean)
        
        if hasattr(model_simple, 'equations_') and model_simple.equations_ is not None:
            equations = model_simple.equations_
            print(f"\n✅ Found {len(equations)} equations with simple config")
            
            # Save results
            csv_path = os.path.join(output_dir, "clean_data_simple_equations.csv")
            equations.to_csv(csv_path, index=False)
            print(f"✅ Simple equations saved to: {csv_path}")
            
            # Display best
            best_idx = equations['loss'].idxmin()
            print(f"\nBest equation (simple):")
            print(f"  {equations.loc[best_idx, 'equation']}")
            
    except Exception as e2:
        print(f"❌ Even simple config failed: {e2}")

print("\n" + "="*70)
print("ANALYSIS COMPLETE")
print("="*70)