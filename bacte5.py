# ==============================
# FINAL OPTIMIZED PySR - USING ALL DATA
# ==============================

import pandas as pd
import numpy as np
from pysr import PySRRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import warnings
import os

warnings.filterwarnings('ignore')

# ------------------------------
# Configuration
# ------------------------------
data_path = r"C:\Users\Amare\Desktop\worku\datato232-81.csv"
output_dir = r"C:\Users\Amare\Desktop\pysr_final_all_data"
os.makedirs(output_dir, exist_ok=True)

print("="*70)
print("FINAL OPTIMIZED PySR - USING ALL DATA")
print("="*70)

# ------------------------------
# Load and prepare ALL data (no outlier removal)
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
# Feature correlations (ALL data)
# ------------------------------
print("\n" + "-"*70)
print("FEATURE CORRELATIONS WITH TARGET (ALL DATA)")
print("-"*70)

for i, feat in enumerate(features):
    corr = np.corrcoef(X[:, i], y)[0, 1]
    print(f"  {feat}: {corr:.4f}")

# Note: Bacteria has weak correlation overall, but strong in outliers!
print("\n⚠️  Note: Cbac shows weak overall correlation but")
print("   strong correlation (-0.58) in high-value outliers!")

# ------------------------------
# Scale features (RobustScaler handles outliers)
# ------------------------------
print("\n" + "-"*70)
print("SCALING WITH ROBUSTSCALER")
print("-"*70)

scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)

print("✅ Features scaled (robust to outliers)")

# ------------------------------
# Optimized PySR Configuration
# ------------------------------
print("\n" + "="*70)
print("OPTIMIZED PySR CONFIGURATION")
print("="*70)

model = PySRRegressor(
    # Increased iterations for thorough search
    niterations=2000,
    populations=20,
    population_size=150,
    
    # Allow more complexity to capture outlier patterns
    maxsize=30,
    parsimony=0.0005,  # Very low parsimony to allow complex expressions
    
    # Rich operator set
    binary_operators=["+", "-", "*", "/", "^"],
    unary_operators=["exp", "log", "sqrt", "square", "cube", "abs"],
    
    # Use all available cores
    procs=4,
    multithreading=True,
    
    loss="L2DistLoss()",
    denoise=True,
    progress=True,
    random_state=42,
    timeout_in_seconds=7200,  # 2 hours timeout
    
    temp_equation_file=True,
    warm_start=False,
    
    # Feature names for better readability
    variable_names=features
)

print(f"\nConfiguration:")
print(f"  Iterations: {model.niterations}")
print(f"  Populations: {model.populations}")
print(f"  Population size: {model.population_size}")
print(f"  Max complexity: {model.maxsize}")
print(f"  Parsimony: {model.parsimony}")

# ------------------------------
# Fit model
# ------------------------------
print("\n" + "="*70)
print("TRAINING PySR ON ALL DATA")
print("="*70)
print("⏳ This will take 1-2 hours...")
print("-"*50)

try:
    model.fit(X_scaled, y)
    print("\n✅ PySR training completed!")
    
    if hasattr(model, 'equations_') and model.equations_ is not None:
        equations = model.equations_
        print(f"\n📊 Found {len(equations)} equations")
        
        # Save all equations
        csv_path = os.path.join(output_dir, "all_equations.csv")
        equations.to_csv(csv_path, index=False)
        print(f"✅ All equations saved to: {csv_path}")
        
        # Display top 20 equations
        print("\n🏆 TOP 20 EQUATIONS:")
        top20 = equations.sort_values('loss').head(20)
        for i, (idx, row) in enumerate(top20.iterrows()):
            print(f"\n  {i+1}. Loss: {row['loss']:.6f}")
            print(f"     Complexity: {row['complexity']}")
            print(f"     Contains bacteria: {'✓' if 'x0' in row['equation'] else '✗'}")
            print(f"     Equation: {row['equation'][:100]}...")
        
        # Best equation
        best_idx = equations['loss'].idxmin()
        best_eq = equations.loc[best_idx, 'equation']
        best_loss = equations.loc[best_idx, 'loss']
        best_complexity = equations.loc[best_idx, 'complexity']
        
        print("\n" + "⭐"*70)
        print("🌟 BEST EQUATION (ALL DATA)")
        print("⭐"*70)
        print(f"Loss: {best_loss:.6f}")
        print(f"Complexity: {best_complexity}")
        print(f"Contains bacteria: {'✓' if 'x0' in best_eq else '✗'}")
        print(f"\nEquation: {best_eq}")
        
        # Get predictions
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
        
        # Save best equation
        txt_path = os.path.join(output_dir, "best_equation.txt")
        with open(txt_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("BEST EQUATION (ALL DATA)\n")
            f.write("="*70 + "\n\n")
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
        
        # Create visualization
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Predicted vs Actual
        axes[0, 0].scatter(y, y_pred, alpha=0.6, c='blue', edgecolors='black', s=50)
        axes[0, 0].plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
        axes[0, 0].set_xlabel('Actual DIZ')
        axes[0, 0].set_ylabel('Predicted DIZ')
        axes[0, 0].set_title(f'Predicted vs Actual (R² = {r2:.4f})')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Residuals
        residuals = y - y_pred
        axes[0, 1].scatter(y_pred, residuals, alpha=0.6, c='green', edgecolors='black', s=50)
        axes[0, 1].axhline(0, color='r', linestyle='--', lw=2)
        axes[0, 1].set_xlabel('Predicted DIZ')
        axes[0, 1].set_ylabel('Residuals')
        axes[0, 1].set_title(f'Residuals (σ = {np.std(residuals):.4f})')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Highlight outliers in residuals
        # Find high-value points
        high_value = y > 30
        axes[0, 2].scatter(y_pred[~high_value], residuals[~high_value], 
                          alpha=0.5, c='blue', label='Normal', s=30)
        axes[0, 2].scatter(y_pred[high_value], residuals[high_value], 
                          alpha=0.8, c='red', label='High DIZ (>30)', s=80, marker='x')
        axes[0, 2].axhline(0, color='r', linestyle='--', lw=2)
        axes[0, 2].set_xlabel('Predicted DIZ')
        axes[0, 2].set_ylabel('Residuals')
        axes[0, 2].set_title('Residuals by Value Group')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Complexity vs Loss
        axes[1, 0].scatter(equations['complexity'], equations['loss'], alpha=0.6, c='orange', s=50)
        axes[1, 0].set_xlabel('Complexity')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].set_title('Complexity vs Loss')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].scatter(best_complexity, best_loss, c='red', s=200, marker='*')
        
        # 5. Feature importance (Random Forest)
        from sklearn.ensemble import RandomForestRegressor
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X_scaled, y)
        importances = rf.feature_importances_
        axes[1, 1].bar(features, importances, color='skyblue', edgecolor='black')
        axes[1, 1].set_ylabel('Importance')
        axes[1, 1].set_title('Random Forest Feature Importance')
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        # 6. Equation and notes
        axes[1, 2].axis('off')
        notes = f"Best Equation:\n\n{best_eq[:150]}...\n\n"
        notes += f"R² = {r2:.4f}\n"
        notes += f"MAE = {mae:.4f}\n"
        notes += f"Contains bacteria: {'Yes' if 'x0' in best_eq else 'No'}\n\n"
        notes += "Note: Outliers (high DIZ) are\ncritical for good predictions!"
        axes[1, 2].text(0.1, 0.5, notes, fontsize=11, verticalalignment='center',
                       fontfamily='monospace', 
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))
        
        plt.suptitle(f'PySR Results on ALL Data (with outliers)', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        plot_path = os.path.join(output_dir, "final_results.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"\n✅ Plot saved to: {plot_path}")
        
        # Check if bacteria is used
        if 'x0' in best_eq:
            print("\n✅ Best equation INCLUDES bacteria concentration!")
        else:
            print("\n⚠️  Best equation does NOT include bacteria.")
            print("   Checking top 10 for bacteria-inclusive equations...")
            
            bacteria_eqs = equations[equations['equation'].str.contains('x0', na=False)]
            if len(bacteria_eqs) > 0:
                print(f"\nFound {len(bacteria_eqs)} equations with bacteria:")
                top_bacteria = bacteria_eqs.sort_values('loss').head(5)
                for i, (idx, row) in enumerate(top_bacteria.iterrows()):
                    print(f"\n  {i+1}. Loss: {row['loss']:.6f}")
                    print(f"     Equation: {row['equation'][:100]}...")
                
                # Save bacteria equations
                bact_path = os.path.join(output_dir, "bacteria_equations.csv")
                bacteria_eqs.to_csv(bact_path, index=False)
                print(f"\n✅ Bacteria equations saved to: {bact_path}")
        
    else:
        print("❌ No equations found")
        
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)
print("ANALYSIS COMPLETE")
print("="*70)
print(f"\nAll outputs saved to: {output_dir}")