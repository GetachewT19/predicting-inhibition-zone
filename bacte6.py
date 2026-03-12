# ==============================
# FINAL PySR RUN - CORRECTED VERSION
# ==============================

import pandas as pd
import numpy as np
from pysr import PySRRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import warnings
import os
import pickle
import datetime

warnings.filterwarnings('ignore')

# ------------------------------
# Configuration
# ------------------------------
data_path = r"C:\Users\Amare\Desktop\worku\datato232-81.csv"
output_dir = r"C:\Users\Amare\Desktop\pysr_final_corrected"
os.makedirs(output_dir, exist_ok=True)

print("="*70)
print("FINAL PySR RUN - CORRECTED VERSION")
print("="*70)
print(f"Start time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ------------------------------
# Load and prepare ALL data
# ------------------------------
df = pd.read_csv(data_path)
target = 'DIZ'
features = ['Cbac', 'CAgNP', 'Ra']

print(f"\n📊 Original dataset shape: {df.shape}")

# Basic cleaning only (remove non-numeric)
df_clean = df.copy()
for col in [target] + features:
    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

initial_rows = len(df_clean)
df_clean = df_clean.dropna()
print(f"Rows dropped (non-numeric only): {initial_rows - len(df_clean)}")
print(f"Final dataset: {len(df_clean)} samples (KEEPING ALL DATA)")

X = df_clean[features].values.astype(float)
y = df_clean[target].values.astype(float)

print(f"\n📈 Target statistics (ALL data):")
print(f"  Mean: {y.mean():.2f}")
print(f"  Std: {y.std():.2f}")
print(f"  Min: {y.min():.2f}")
print(f"  Max: {y.max():.2f}")

# ------------------------------
# Feature correlations
# ------------------------------
print("\n" + "-"*70)
print("FEATURE CORRELATIONS WITH TARGET")
print("-"*70)

for i, feat in enumerate(features):
    corr = np.corrcoef(X[:, i], y)[0, 1]
    print(f"  {feat}: {corr:.4f}")

# ------------------------------
# Scale features
# ------------------------------
print("\n" + "-"*70)
print("SCALING WITH ROBUSTSCALER")
print("-"*70)

scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)

# Save the scaler for later use
scaler_path = os.path.join(output_dir, "scaler.pkl")
with open(scaler_path, 'wb') as f:
    pickle.dump(scaler, f)
print(f"✅ Scaler saved to: {scaler_path}")

# ------------------------------
# PySR Configuration - CORRECTED (no equation_file)
# ------------------------------
print("\n" + "="*70)
print("PySR CONFIGURATION")
print("="*70)

# Create model with valid parameters only
model = PySRRegressor(
    niterations=1000,
    populations=10,
    population_size=100,
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
    timeout_in_seconds=3600,  # 1 hour timeout
    temp_equation_file=True,   # This will create temporary equation files
    variable_names=features,
    update_verbosity=1,
    warm_start=False
)

print(f"\nConfiguration:")
print(f"  Iterations: {model.niterations}")
print(f"  Populations: {model.populations}")
print(f"  Population size: {model.population_size}")
print(f"  Max complexity: {model.maxsize}")
print(f"  Parsimony: {model.parsimony}")
print(f"  Temp equation file: Enabled")

# ------------------------------
# Fit model
# ------------------------------
print("\n" + "="*70)
print("TRAINING PySR")
print("="*70)
print("⏳ This will take about 1 hour...")
print("-"*50)

try:
    model.fit(X_scaled, y)
    print("\n✅ PySR training completed!")
    
    # Save the entire model
    model_path = os.path.join(output_dir, "pysr_model.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"✅ Model saved to: {model_path}")
    
    # Check if equations exist
    if hasattr(model, 'equations_') and model.equations_ is not None:
        equations = model.equations_
        print(f"\n📊 Found {len(equations)} equations in memory")
        
        # Save equations to CSV
        csv_path = os.path.join(output_dir, "all_equations.csv")
        equations.to_csv(csv_path, index=False)
        print(f"✅ Equations saved to: {csv_path}")
        
        # Display top 10 equations
        print("\n🏆 TOP 10 EQUATIONS:")
        top10 = equations.sort_values('loss').head(10)
        for i, (idx, row) in enumerate(top10.iterrows()):
            contains_bacteria = 'x0' in str(row['equation'])
            marker = '✓' if contains_bacteria else '✗'
            print(f"\n  {i+1}. Loss: {row['loss']:.6f} | Complexity: {row['complexity']} | Bacteria: {marker}")
            print(f"     Equation: {row['equation'][:100]}...")
        
        # Best equation
        best_idx = equations['loss'].idxmin()
        best_eq = equations.loc[best_idx, 'equation']
        best_loss = equations.loc[best_idx, 'loss']
        best_complexity = equations.loc[best_idx, 'complexity']
        
        print("\n" + "⭐"*70)
        print("🌟 BEST EQUATION")
        print("⭐"*70)
        print(f"Loss: {best_loss:.6f}")
        print(f"Complexity: {best_complexity}")
        print(f"Contains bacteria: {'✓' if 'x0' in best_eq else '✗'}")
        print(f"\nEquation: {best_eq}")
        
        # Make predictions
        y_pred = model.predict(X_scaled)
        
        # Calculate metrics
        r2 = r2_score(y, y_pred)
        mae = mean_absolute_error(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        mape = np.mean(np.abs((y - y_pred) / (y + 1e-10))) * 100
        
        print(f"\n📈 PERFORMANCE METRICS:")
        print(f"  R² = {r2:.4f} ({r2*100:.1f}% variance explained)")
        print(f"  MAE = {mae:.4f}")
        print(f"  RMSE = {rmse:.4f}")
        print(f"  MAPE = {mape:.2f}%")
        
        # Save best equation and metrics to text file
        txt_path = os.path.join(output_dir, "best_equation.txt")
        with open(txt_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("BEST EQUATION FOR DIZ\n")
            f.write("="*70 + "\n\n")
            f.write(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Samples: {len(y)}\n")
            f.write(f"Features: {features}\n\n")
            f.write(f"R² Score: {r2:.6f}\n")
            f.write(f"MAE: {mae:.6f}\n")
            f.write(f"RMSE: {rmse:.6f}\n")
            f.write(f"MAPE: {mape:.2f}%\n")
            f.write(f"Loss: {best_loss:.6f}\n")
            f.write(f"Complexity: {best_complexity}\n\n")
            f.write(f"Equation: {best_eq}\n")
        
        print(f"\n✅ Best equation saved to: {txt_path}")
        
        # Check for bacteria equations
        bacteria_eqs = equations[equations['equation'].str.contains('x0', na=False)]
        if len(bacteria_eqs) > 0:
            print(f"\n🧫 Found {len(bacteria_eqs)} equations with bacteria")
            bact_path = os.path.join(output_dir, "bacteria_equations.csv")
            bacteria_eqs.to_csv(bact_path, index=False)
            print(f"✅ Bacteria equations saved to: {bact_path}")
            
            # Show best bacteria equation
            best_bact = bacteria_eqs.sort_values('loss').iloc[0]
            print(f"\nBest bacteria equation:")
            print(f"  Loss: {best_bact['loss']:.6f}")
            print(f"  Complexity: {best_bact['complexity']}")
            print(f"  Equation: {best_bact['equation']}")
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Predicted vs Actual
        axes[0, 0].scatter(y, y_pred, alpha=0.6, c='blue', edgecolors='black', s=50)
        axes[0, 0].plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2, label='Perfect Fit')
        axes[0, 0].set_xlabel('Actual DIZ')
        axes[0, 0].set_ylabel('Predicted DIZ')
        axes[0, 0].set_title(f'Predicted vs Actual (R² = {r2:.4f})')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Residuals
        residuals = y - y_pred
        axes[0, 1].scatter(y_pred, residuals, alpha=0.6, c='green', edgecolors='black', s=50)
        axes[0, 1].axhline(0, color='r', linestyle='--', lw=2)
        axes[0, 1].set_xlabel('Predicted DIZ')
        axes[0, 1].set_ylabel('Residuals')
        axes[0, 1].set_title(f'Residuals (σ = {np.std(residuals):.4f})')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Complexity vs Loss
        axes[1, 0].scatter(equations['complexity'], equations['loss'], alpha=0.6, c='orange', s=50)
        axes[1, 0].set_xlabel('Complexity')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].set_title('Complexity vs Loss')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].scatter(best_complexity, best_loss, c='red', s=200, marker='*', 
                          label=f'Best (C={best_complexity})')
        axes[1, 0].legend()
        
        # Equation text
        axes[1, 1].axis('off')
        eq_text = f"Best Equation:\n\n"
        # Split long equation into multiple lines
        eq_lines = [best_eq[i:i+50] for i in range(0, len(best_eq), 50)]
        for line in eq_lines:
            eq_text += line + "\n"
        eq_text += f"\nR² = {r2:.4f}\n"
        eq_text += f"MAE = {mae:.4f}\n"
        eq_text += f"RMSE = {rmse:.4f}\n"
        eq_text += f"Contains bacteria: {'Yes' if 'x0' in best_eq else 'No'}"
        axes[1, 1].text(0.1, 0.5, eq_text, fontsize=10, verticalalignment='center',
                       fontfamily='monospace', 
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))
        
        plt.suptitle('PySR Final Results - All Data', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        plot_path = os.path.join(output_dir, "final_results.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"\n✅ Plot saved to: {plot_path}")
        
        # Create a simple text summary
        summary_path = os.path.join(output_dir, "summary.txt")
        with open(summary_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("PySR RESULTS SUMMARY\n")
            f.write("="*70 + "\n\n")
            f.write(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total equations found: {len(equations)}\n")
            f.write(f"Equations with bacteria: {len(bacteria_eqs) if len(bacteria_eqs) > 0 else 0}\n\n")
            f.write(f"Best R²: {r2:.4f}\n")
            f.write(f"Best equation: {best_eq}\n")
        
        print(f"\n✅ Summary saved to: {summary_path}")
        
    else:
        print("❌ No equations found in model")
        
except Exception as e:
    print(f"\n❌ Error during PySR: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)
print("ANALYSIS COMPLETE")
print("="*70)
print(f"End time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"\n✅ All outputs saved to: {output_dir}")
print("\nFiles created:")
print(f"  - {os.path.join(output_dir, 'all_equations.csv')}")
print(f"  - {os.path.join(output_dir, 'best_equation.txt')}")
print(f"  - {os.path.join(output_dir, 'final_results.png')}")
print(f"  - {os.path.join(output_dir, 'pysr_model.pkl')}")
print(f"  - {os.path.join(output_dir, 'scaler.pkl')}")
if os.path.exists(os.path.join(output_dir, 'bacteria_equations.csv')):
    print(f"  - {os.path.join(output_dir, 'bacteria_equations.csv')}")
print(f"  - {os.path.join(output_dir, 'summary.txt')}")