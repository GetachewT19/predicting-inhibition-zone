# ==============================
# PySR with Outlier Removal for Better DIZ Prediction
# ==============================

import pandas as pd
import numpy as np
from pysr import PySRRegressor
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
import matplotlib.pyplot as plt
import warnings
import os

warnings.filterwarnings('ignore')

# ------------------------------
# Configuration
# ------------------------------
data_path = r"C:\Users\Amare\Desktop\worku\datato232-81.csv"
output_dir = r"C:\Users\Amare\Desktop\pysr_cleaned_data"
os.makedirs(output_dir, exist_ok=True)

print("="*70)
print("PySR WITH ADVANCED OUTLIER REMOVAL")
print("="*70)

# ------------------------------
# Load data
# ------------------------------
df = pd.read_csv(data_path)
target = 'DIZ'
features = ['Cbac', 'CAgNP', 'Ra']

print(f"\n📊 Original dataset shape: {df.shape}")
print(f"Features: {features}")
print(f"Target: {target}")

# ------------------------------
# Step 1: Basic cleaning (non-numeric values)
# ------------------------------
print("\n" + "-"*70)
print("STEP 1: BASIC CLEANING")
print("-"*70)

df_clean = df.copy()
for col in [target] + features:
    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

initial_rows = len(df_clean)
df_clean = df_clean.dropna()
rows_dropped_nan = initial_rows - len(df_clean)

print(f"Rows dropped due to non-numeric values: {rows_dropped_nan}")
print(f"Rows after basic cleaning: {len(df_clean)}")

X = df_clean[features].values.astype(float)
y = df_clean[target].values.astype(float)

# ------------------------------
# Step 2: Statistical outlier detection
# ------------------------------
print("\n" + "-"*70)
print("STEP 2: STATISTICAL OUTLIER DETECTION")
print("-"*70)

# Method 1: Z-score method
from scipy import stats
z_scores = np.abs(stats.zscore(y))
z_score_threshold = 3
z_score_outliers = np.where(z_scores > z_score_threshold)[0]
print(f"Z-score outliers (>{z_score_threshold}σ): {len(z_score_outliers)}")

# Method 2: IQR method
Q1 = np.percentile(y, 25)
Q3 = np.percentile(y, 75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
iqr_outliers = np.where((y < lower_bound) | (y > upper_bound))[0]
print(f"IQR outliers (1.5*IQR): {len(iqr_outliers)}")

# Method 3: Modified Z-score (more robust for small datasets)
median_y = np.median(y)
mad_y = np.median(np.abs(y - median_y))  # Median Absolute Deviation
modified_z_scores = 0.6745 * (y - median_y) / mad_y if mad_y > 0 else np.zeros_like(y)
mod_z_score_threshold = 3.5
mod_z_score_outliers = np.where(np.abs(modified_z_scores) > mod_z_score_threshold)[0]
print(f"Modified Z-score outliers: {len(mod_z_score_outliers)}")

# ------------------------------
# Step 3: Multivariate outlier detection
# ------------------------------
print("\n" + "-"*70)
print("STEP 3: MULTIVARIATE OUTLIER DETECTION")
print("-"*70)

# Combine features and target for multivariate analysis
Xy = np.column_stack([X, y])

# Method 4: Isolation Forest
iso_forest = IsolationForest(contamination=0.1, random_state=42)
iso_labels = iso_forest.fit_predict(Xy)
iso_outliers = np.where(iso_labels == -1)[0]
print(f"Isolation Forest outliers (10% contamination): {len(iso_outliers)}")

# Method 5: Local Outlier Factor
lof = LocalOutlierFactor(contamination=0.1, novelty=False)
lof_labels = lof.fit_predict(Xy)
lof_outliers = np.where(lof_labels == -1)[0]
print(f"Local Outlier Factor outliers: {len(lof_outliers)}")

# ------------------------------
# Step 4: Create clean dataset by removing outliers
# ------------------------------
print("\n" + "-"*70)
print("STEP 4: CREATING CLEAN DATASET")
print("-"*70)

# Create mask for outliers (union of all methods)
outlier_mask = np.zeros(len(y), dtype=bool)

# Add outliers from each method (you can adjust which methods to use)
outlier_mask[z_score_outliers] = True
outlier_mask[iqr_outliers] = True
outlier_mask[mod_z_score_outliers] = True
outlier_mask[iso_outliers] = True
outlier_mask[lof_outliers] = True

# Remove duplicates and count
total_outliers = np.sum(outlier_mask)
print(f"Total unique outliers detected: {total_outliers}")

# Create clean dataset
X_clean = X[~outlier_mask]
y_clean = y[~outlier_mask]

print(f"\nOriginal data: {len(y)} samples")
print(f"Clean data: {len(y_clean)} samples")
print(f"Removed: {len(y) - len(y_clean)} samples ({((len(y) - len(y_clean))/len(y)*100):.1f}%)")

# ------------------------------
# Step 5: Visualize outlier removal
# ------------------------------
print("\n" + "-"*70)
print("STEP 5: VISUALIZING OUTLIER REMOVAL")
print("-"*70)

fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. Target distribution before/after
axes[0, 0].hist(y, bins=30, alpha=0.5, label='Original', color='red', edgecolor='black')
axes[0, 0].hist(y_clean, bins=30, alpha=0.5, label='Clean', color='green', edgecolor='black')
axes[0, 0].set_xlabel('DIZ Value')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].set_title('Target Distribution: Before vs After Outlier Removal')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 2. Box plot comparison
axes[0, 1].boxplot([y, y_clean], labels=['Original', 'Clean'])
axes[0, 1].set_ylabel('DIZ Value')
axes[0, 1].set_title('Box Plot Comparison')
axes[0, 1].grid(True, alpha=0.3)

# 3. Scatter plot of first two features with outliers highlighted
axes[1, 0].scatter(X[~outlier_mask, 0], X[~outlier_mask, 1], 
                   c='blue', alpha=0.5, label='Inliers', s=30)
axes[1, 0].scatter(X[outlier_mask, 0], X[outlier_mask, 1], 
                   c='red', alpha=0.7, label='Outliers', s=50, marker='x')
axes[1, 0].set_xlabel(features[0])
axes[1, 0].set_ylabel(features[1])
axes[1, 0].set_title('Outliers in Feature Space')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# 4. Target vs Feature 1 with outliers
axes[1, 1].scatter(X[~outlier_mask, 0], y[~outlier_mask], 
                   c='blue', alpha=0.5, label='Inliers', s=30)
axes[1, 1].scatter(X[outlier_mask, 0], y[outlier_mask], 
                   c='red', alpha=0.7, label='Outliers', s=50, marker='x')
axes[1, 1].set_xlabel(features[0])
axes[1, 1].set_ylabel(target)
axes[1, 1].set_title(f'{target} vs {features[0]}')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.suptitle('Outlier Detection Results', fontsize=14, fontweight='bold')
plt.tight_layout()
plot_path = os.path.join(output_dir, "outlier_removal.png")
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.show()
print(f"✅ Outlier visualization saved to: {plot_path}")

# ------------------------------
# Step 6: Scale clean data
# ------------------------------
print("\n" + "-"*70)
print("STEP 6: SCALING CLEAN DATA")
print("-"*70)

scaler = RobustScaler()
X_clean_scaled = scaler.fit_transform(X_clean)

print(f"Clean data shape: X {X_clean_scaled.shape}, y {y_clean.shape}")
print(f"\nTarget statistics after cleaning:")
print(f"  Mean: {y_clean.mean():.4f}")
print(f"  Std: {y_clean.std():.4f}")
print(f"  Min: {y_clean.min():.4f}")
print(f"  Max: {y_clean.max():.4f}")

# ------------------------------
# Step 7: Run PySR on clean data
# ------------------------------
print("\n" + "="*70)
print("STEP 7: RUNNING PySR ON CLEAN DATA")
print("="*70)

model = PySRRegressor(
    niterations=1500,
    populations=15,
    population_size=120,
    maxsize=20,
    parsimony=0.001,
    binary_operators=["+", "-", "*", "/", "^"],
    unary_operators=["exp", "log", "sqrt", "square", "cube"],
    procs=4,
    multithreading=True,
    loss="L2DistLoss()",
    denoise=True,
    progress=True,
    random_state=42,
    timeout_in_seconds=3600,
    temp_equation_file=True,
    variable_names=features
)

try:
    model.fit(X_clean_scaled, y_clean)
    
    if hasattr(model, 'equations_') and model.equations_ is not None:
        equations = model.equations_
        print(f"\n✅ Found {len(equations)} equations")
        
        # Filter equations that use bacteria (x0)
        bacteria_eqs = equations[equations['equation'].str.contains('x0', na=False)]
        print(f"\n🧫 Equations containing bacteria: {len(bacteria_eqs)}")
        
        # Best overall equation
        best_idx = equations['loss'].idxmin()
        best_eq = equations.loc[best_idx, 'equation']
        best_loss = equations.loc[best_idx, 'loss']
        best_complexity = equations.loc[best_idx, 'complexity']
        
        print("\n" + "⭐"*70)
        print("🌟 BEST EQUATION ON CLEAN DATA")
        print("⭐"*70)
        print(f"Complexity: {best_complexity}")
        print(f"Loss: {best_loss:.6f}")
        print(f"Equation: {best_eq}")
        print(f"Uses bacteria: {'x0' in best_eq}")
        
        # Make predictions
        y_pred = model.predict(X_clean_scaled)
        r2 = r2_score(y_clean, y_pred)
        mae = mean_absolute_error(y_clean, y_pred)
        rmse = np.sqrt(mean_squared_error(y_clean, y_pred))
        
        print(f"\n📈 Performance on clean data:")
        print(f"   R² = {r2:.4f}")
        print(f"   MAE = {mae:.4f}")
        print(f"   RMSE = {rmse:.4f}")
        
        # Save results
        csv_path = os.path.join(output_dir, "clean_data_equations.csv")
        equations.to_csv(csv_path, index=False)
        
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
        
        print(f"\n✅ Results saved to: {output_dir}")
        
        # Visualize results
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Predicted vs Actual
        axes[0, 0].scatter(y_clean, y_pred, alpha=0.6, c='blue', edgecolors='black', s=60)
        axes[0, 0].plot([y_clean.min(), y_clean.max()], [y_clean.min(), y_clean.max()], 
                        'r--', lw=2, label='Perfect Fit')
        axes[0, 0].set_xlabel('Actual DIZ')
        axes[0, 0].set_ylabel('Predicted DIZ')
        axes[0, 0].set_title(f'Clean Data: R² = {r2:.4f}')
        axes[0, 0].legend()
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
        print(f"✅ Results plot saved to: {plot_path}")
        
except Exception as e:
    print(f"\n❌ PySR failed: {e}")

print("\n" + "="*70)
print("ANALYSIS COMPLETE")
print("="*70)
print(f"\nAll outputs saved to: {output_dir}")