# ==============================
# FINAL ATTEMPT - Manual Equation Saving
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
output_dir = r"C:\Users\Amare\Desktop\pysr_final_manual_save"
os.makedirs(output_dir, exist_ok=True)

print("="*70)
print("FINAL ATTEMPT - MANUAL EQUATION SAVING")
print("="*70)

# ------------------------------
# Load and prepare ALL data
# ------------------------------
df = pd.read_csv(data_path)
target = 'DIZ'
features = ['Cbac', 'CAgNP', 'Ra']

print(f"\n📊 Original dataset shape: {df.shape}")

# Basic cleaning
df_clean = df.copy()
for col in [target] + features:
    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

initial_rows = len(df_clean)
df_clean = df_clean.dropna()
print(f"Rows dropped (non-numeric only): {initial_rows - len(df_clean)}")
print(f"Final dataset: {len(df_clean)} samples")

X = df_clean[features].values.astype(float)
y = df_clean[target].values.astype(float)

print(f"\n📈 Target statistics:")
print(f"  Mean: {y.mean():.2f}")
print(f"  Std: {y.std():.2f}")
print(f"  Min: {y.min():.2f}")
print(f"  Max: {y.max():.2f}")

# ------------------------------
# Feature correlations
# ------------------------------
print("\n" + "-"*70)
print("FEATURE CORRELATIONS")
print("-"*70)

correlations = []
for i, feat in enumerate(features):
    corr = np.corrcoef(X[:, i], y)[0, 1]
    correlations.append((feat, corr))
    print(f"  {feat}: {corr:.4f}")

# ------------------------------
# Scale features
# ------------------------------
print("\n" + "-"*70)
print("SCALING FEATURES")
print("-"*70)

scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)

# Save scaler
scaler_path = os.path.join(output_dir, "scaler.pkl")
with open(scaler_path, 'wb') as f:
    pickle.dump(scaler, f)
print(f"✅ Scaler saved to: {scaler_path}")

# ------------------------------
# PySR with simpler configuration
# ------------------------------
print("\n" + "="*70)
print("RUNNING PySR")
print("="*70)

# Use a simpler configuration that's more likely to work
model = PySRRegressor(
    niterations=500,  # Fewer iterations
    populations=5,
    population_size=80,
    maxsize=15,
    parsimony=0.01,
    binary_operators=["+", "-", "*", "/"],
    unary_operators=["sqrt", "square"],
    procs=2,
    loss="L2DistLoss()",
    progress=True,
    random_state=42,
    temp_equation_file=True,
    update_verbosity=1
)

print(f"\nConfiguration:")
print(f"  Iterations: {model.niterations}")
print(f"  Populations: {model.populations}")
print(f"  Max complexity: {model.maxsize}")

# ------------------------------
# Fit model and save results manually
# ------------------------------
print("\n" + "="*70)
print("TRAINING PySR")
print("="*70)
print("⏳ This will take about 30 minutes...")
print("-"*50)

try:
    model.fit(X_scaled, y)
    print("\n✅ PySR training completed!")
    
    # Check if equations exist
    if hasattr(model, 'equations_') and model.equations_ is not None:
        equations = model.equations_
        print(f"\n📊 Found {len(equations)} equations")
        
        # MANUALLY save equations to CSV
        csv_path = os.path.join(output_dir, "all_equations.csv")
        equations.to_csv(csv_path, index=False)
        print(f"✅ Equations manually saved to: {csv_path}")
        
        # Verify file was created
        if os.path.exists(csv_path):
            print(f"   File size: {os.path.getsize(csv_path)} bytes")
        
        # Save model
        model_path = os.path.join(output_dir, "pysr_model.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"✅ Model saved to: {model_path}")
        
        # Display top equations
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
        
        print(f"\n📈 PERFORMANCE METRICS:")
        print(f"  R² = {r2:.4f} ({r2*100:.1f}%)")
        print(f"  MAE = {mae:.4f}")
        print(f"  RMSE = {rmse:.4f}")
        
        # Save best equation to text file
        txt_path = os.path.join(output_dir, "best_equation.txt")
        with open(txt_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("BEST EQUATION FOR DIZ\n")
            f.write("="*70 + "\n\n")
            f.write(f"Date: {datetime.datetime.now()}\n")
            f.write(f"Features: {features}\n")
            f.write(f"Samples: {len(y)}\n\n")
            f.write(f"R² Score: {r2:.6f}\n")
            f.write(f"MAE: {mae:.6f}\n")
            f.write(f"RMSE: {rmse:.6f}\n")
            f.write(f"Loss: {best_loss:.6f}\n")
            f.write(f"Complexity: {best_complexity}\n\n")
            f.write(f"Equation: {best_eq}\n")
        
        print(f"✅ Best equation saved to: {txt_path}")
        
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
            print(f"  Equation: {best_bact['equation']}")
        
        # Create and save plot
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Predicted vs Actual
        axes[0].scatter(y, y_pred, alpha=0.6, c='blue', edgecolors='black', s=50)
        axes[0].plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
        axes[0].set_xlabel('Actual DIZ')
        axes[0].set_ylabel('Predicted DIZ')
        axes[0].set_title(f'Predicted vs Actual (R² = {r2:.4f})')
        axes[0].grid(True, alpha=0.3)
        
        # Residuals
        residuals = y - y_pred
        axes[1].scatter(y_pred, residuals, alpha=0.6, c='green', edgecolors='black', s=50)
        axes[1].axhline(0, color='r', linestyle='--', lw=2)
        axes[1].set_xlabel('Predicted DIZ')
        axes[1].set_ylabel('Residuals')
        axes[1].set_title(f'Residuals (σ = {np.std(residuals):.4f})')
        axes[1].grid(True, alpha=0.3)
        
        plt.suptitle('PySR Results', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        plot_path = os.path.join(output_dir, "results.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"✅ Plot saved to: {plot_path}")
        
        # Create summary file
        summary_path = os.path.join(output_dir, "summary.txt")
        with open(summary_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("PySR RESULTS SUMMARY\n")
            f.write("="*70 + "\n\n")
            f.write(f"Date: {datetime.datetime.now()}\n")
            f.write(f"Total equations: {len(equations)}\n")
            f.write(f"Equations with bacteria: {len(bacteria_eqs) if len(bacteria_eqs) > 0 else 0}\n\n")
            f.write(f"Best R²: {r2:.4f}\n")
            f.write(f"Best equation: {best_eq}\n")
        
        print(f"✅ Summary saved to: {summary_path}")
        
        # Final verification
        print("\n" + "="*70)
        print("VERIFYING SAVED FILES")
        print("="*70)
        
        expected_files = [
            ("all_equations.csv", csv_path),
            ("best_equation.txt", txt_path),
            ("results.png", plot_path),
            ("pysr_model.pkl", model_path),
            ("scaler.pkl", scaler_path),
            ("summary.txt", summary_path)
        ]
        
        if len(bacteria_eqs) > 0:
            expected_files.append(("bacteria_equations.csv", bact_path))
        
        all_good = True
        for name, path in expected_files:
            if os.path.exists(path):
                size = os.path.getsize(path)
                print(f"  ✅ {name}: {size:,} bytes")
            else:
                print(f"  ❌ {name}: NOT FOUND")
                all_good = False
        
        if all_good:
            print("\n✅ ALL FILES SUCCESSFULLY SAVED!")
        else:
            print("\n⚠️ Some files are missing")
        
    else:
        print("❌ No equations found in model")
        
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)
print("ANALYSIS COMPLETE")
print("="*70)
print(f"\nCheck the output directory: {output_dir}")