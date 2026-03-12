# ==============================
# Verify Test File and Run Full Analysis
# ==============================

import pandas as pd
import numpy as np
import os

print("="*70)
print("VERIFYING TEST FILE")
print("="*70)

# Check if test file was saved
test_csv = r"C:\Users\Amare\Desktop\test_equations.csv"
if os.path.exists(test_csv):
    print(f"\n✅ Test file found: {test_csv}")
    
    # Load and display test equations
    df_test = pd.read_csv(test_csv)
    print(f"\nTest equations found: {len(df_test)}")
    print(df_test[['complexity', 'loss', 'equation']].to_string())
    
    print("\n✅ File saving is working!")
else:
    print(f"\n❌ Test file not found: {test_csv}")

print("\n" + "="*70)
print("NOW RUNNING FULL ANALYSIS WITH PROPER SAVING")
print("="*70)

# Now run the full analysis with the working approach
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
output_dir = r"C:\Users\Amare\Desktop\pysr_final_working"
os.makedirs(output_dir, exist_ok=True)

print(f"\n📁 Output directory: {output_dir}")

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

# Save the scaler
scaler_path = os.path.join(output_dir, "scaler.pkl")
with open(scaler_path, 'wb') as f:
    pickle.dump(scaler, f)
print(f"✅ Scaler saved to: {scaler_path}")

# ------------------------------
# PySR Configuration
# ------------------------------
print("\n" + "="*70)
print("PySR CONFIGURATION")
print("="*70)

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
    timeout_in_seconds=3600,
    temp_equation_file=True,
    variable_names=features,
    update_verbosity=1
)

print(f"\nConfiguration:")
print(f"  Iterations: {model.niterations}")
print(f"  Populations: {model.populations}")
print(f"  Max complexity: {model.maxsize}")

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
    
    # Save the model
    model_path = os.path.join(output_dir, "pysr_model.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"✅ Model saved to: {model_path}")
    
    # Check for equations
    if hasattr(model, 'equations_') and model.equations_ is not None:
        equations = model.equations_
        print(f"\n📊 Found {len(equations)} equations")
        
        # Save equations to CSV
        csv_path = os.path.join(output_dir, "all_equations.csv")
        equations.to_csv(csv_path, index=False)
        print(f"✅ Equations saved to: {csv_path}")
        
        # Display top 10
        print("\n🏆 TOP 10 EQUATIONS:")
        top10 = equations.sort_values('loss').head(10)
        for i, (idx, row) in enumerate(top10.iterrows()):
            contains_bacteria = 'x0' in str(row['equation'])
            marker = '✓' if contains_bacteria else '✗'
            print(f"\n  {i+1}. Loss: {row['loss']:.6f} | Complexity: {row['complexity']} | Bacteria: {marker}")
            print(f"     Equation: {row['equation']}")
        
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
        print(f"  R² = {r2:.4f} ({r2*100:.1f}% variance explained)")
        print(f"  MAE = {mae:.4f}")
        print(f"  RMSE = {rmse:.4f}")
        
        # Save best equation
        txt_path = os.path.join(output_dir, "best_equation.txt")
        with open(txt_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("BEST EQUATION FOR DIZ\n")
            f.write("="*70 + "\n\n")
            f.write(f"R² Score: {r2:.6f}\n")
            f.write(f"MAE: {mae:.6f}\n")
            f.write(f"RMSE: {rmse:.6f}\n")
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
            
            # Show best bacteria equation
            best_bact = bacteria_eqs.sort_values('loss').iloc[0]
            print(f"\nBest bacteria equation:")
            print(f"  Loss: {best_bact['loss']:.6f}")
            print(f"  Equation: {best_bact['equation']}")
        
        # Create plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Predicted vs Actual
        ax1.scatter(y, y_pred, alpha=0.6, c='blue', edgecolors='black', s=50)
        ax1.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
        ax1.set_xlabel('Actual DIZ')
        ax1.set_ylabel('Predicted DIZ')
        ax1.set_title(f'Predicted vs Actual (R² = {r2:.4f})')
        ax1.grid(True, alpha=0.3)
        
        # Residuals
        residuals = y - y_pred
        ax2.scatter(y_pred, residuals, alpha=0.6, c='green', edgecolors='black', s=50)
        ax2.axhline(0, color='r', linestyle='--', lw=2)
        ax2.set_xlabel('Predicted DIZ')
        ax2.set_ylabel('Residuals')
        ax2.set_title(f'Residuals (σ = {np.std(residuals):.4f})')
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle('PySR Results - All Data', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        plot_path = os.path.join(output_dir, "results.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"\n✅ Plot saved to: {plot_path}")
        
    else:
        print("❌ No equations found")
        
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)
print("ANALYSIS COMPLETE")
print("="*70)
print(f"\n✅ All outputs saved to: {output_dir}")