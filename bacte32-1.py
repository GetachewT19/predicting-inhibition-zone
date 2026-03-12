# ==============================
# ULTIMATE DIZ OPTIMIZATION - ADVANCED STRATEGIES
# ==============================

import pandas as pd
import numpy as np
from pysr import PySRRegressor
from sklearn.preprocessing import RobustScaler, PowerTransformer, PolynomialFeatures
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score
import warnings
import os
import pickle
import time

warnings.filterwarnings('ignore')

# Configuration
data_path = r"C:\Users\Amare\Desktop\getachewt\datato232-81.csv"
output_dir = r"C:\Users\Amare\Desktop\diz_ultimate"
os.makedirs(output_dir, exist_ok=True)

print("="*70)
print("ULTIMATE DIZ OPTIMIZATION")
print(f"Current best: 0.7749")
print(f"Theoretical max: 0.9173")
print("="*70)

# Load data
df = pd.read_csv(data_path)
target = 'DIZ'
features = ['Cbac', 'CAgNP', 'Ra']

# Clean data
df_clean = df.copy()
for col in [target] + features:
    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
df_clean = df_clean.dropna()

X = df_clean[features].values.astype(float)
y = df_clean[target].values.astype(float)

print(f"\n📊 Dataset: {len(y)} samples")

# ==============================
# STRATEGY 1: ENSEMBLE FEATURE ENGINEERING
# ==============================
print("\n" + "="*70)
print("STRATEGY 1: ENSEMBLE FEATURE ENGINEERING")
print("="*70)

# Create multiple feature sets
feature_sets = []

# Original features
feature_sets.append(('original', X))

# Polynomial features (degree 2)
poly2 = PolynomialFeatures(degree=2, include_bias=False)
X_poly2 = poly2.fit_transform(X)
feature_sets.append(('poly2', X_poly2))

# Polynomial features (degree 3) - only interactions
poly3 = PolynomialFeatures(degree=3, include_bias=False, interaction_only=True)
X_poly3 = poly3.fit_transform(X)
feature_sets.append(('poly3', X_poly3))

# Ratio features
X_ratios = np.column_stack([
    X[:, 0] / (X[:, 1] + 1e-10),  # Cbac/CAgNP
    X[:, 1] / (X[:, 2] + 1e-10),  # CAgNP/Ra
    X[:, 0] / (X[:, 2] + 1e-10),  # Cbac/Ra
])
feature_sets.append(('ratios', X_ratios))

print(f"Created {len(feature_sets)} feature sets")

# ==============================
# STRATEGY 2: MULTIPLE TARGET TRANSFORMATIONS
# ==============================
print("\n" + "="*70)
print("STRATEGY 2: MULTIPLE TARGET TRANSFORMATIONS")
print("="*70)

target_transforms = {
    'original': y,
    'log': np.log1p(y - y.min() + 1),
    'sqrt': np.sqrt(y - y.min() + 1),
    'inverse': 1 / (y + 1e-10),
    'square': y**2,
}

# Add Box-Cox if positive
if (y > 0).all():
    pt = PowerTransformer(method='box-cox')
    y_boxcox = pt.fit_transform(y.reshape(-1, 1)).flatten()
    target_transforms['boxcox'] = y_boxcox

print(f"Created {len(target_transforms)} target transformations")

# ==============================
# STRATEGY 3: CUSTOM WEIGHTING FOR BACTERIA
# ==============================
print("\n" + "="*70)
print("STRATEGY 3: CUSTOM WEIGHTING FOR BACTERIA")
print("="*70)

# Create weighted samples to emphasize bacteria importance
from sklearn.utils import class_weight

# Create weights based on bacteria concentration
bacteria_weights = np.abs(X[:, 0]) / np.abs(X[:, 0]).max()
sample_weights = 1 + bacteria_weights  # Higher weight for high bacteria samples

print(f"Created sample weights: min={sample_weights.min():.2f}, max={sample_weights.max():.2f}")

# ==============================
# STRATEGY 4: OPTIMIZED PySR WITH CUSTOM SETTINGS
# ==============================
print("\n" + "="*70)
print("STRATEGY 4: OPTIMIZED PySR WITH CUSTOM SETTINGS")
print("="*70)

best_r2 = 0.7749
best_equation = None
best_model = None

# Try each feature set with best transformation
for feat_name, X_feat in feature_sets:
    print(f"\n{'='*50}")
    print(f"FEATURE SET: {feat_name}")
    print(f"{'='*50}")
    
    # Scale features
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X_feat)
    
    for trans_name, y_trans in target_transforms.items():
        print(f"\n  📊 Transformation: {trans_name}")
        
        # Ultra-aggressive PySR configuration
        model = PySRRegressor(
            niterations=2000,
            populations=25,
            population_size=300,
            maxsize=40,
            parsimony=0.0001,
            binary_operators=["+", "-", "*", "/", "^"],
            unary_operators=["exp", "log", "sqrt", "square", "cube", "sin", "cos", "tanh", "abs"],
            procs=8,
            multithreading=True,
            loss="L2DistLoss()",
            denoise=True,
            progress=True,
            random_state=42,
            timeout_in_seconds=7200,
            temp_equation_file=True,
            turbo=True,
            should_optimize_constants=True,
            optimizer_algorithm="BFGS",
            optimizer_nrestarts=30,
            warm_start=False
        )
        
        try:
            # Fit with sample weights if available
            model.fit(X_scaled, y_trans)
            
            if hasattr(model, 'equations_') and model.equations_ is not None:
                equations = model.equations_
                
                # Get best equation
                best_idx = equations['loss'].idxmin()
                best_eq_candidate = equations.loc[best_idx, 'equation']
                
                # Make predictions
                y_pred_trans = model.predict(X_scaled)
                
                # Inverse transform
                if trans_name == 'log':
                    y_pred = np.expm1(y_pred_trans) + y.min() - 1
                elif trans_name == 'sqrt':
                    y_pred = y_pred_trans ** 2 + y.min() - 1
                elif trans_name == 'inverse':
                    y_pred = 1 / (y_pred_trans + 1e-10)
                elif trans_name == 'square':
                    y_pred = np.sqrt(np.abs(y_pred_trans))
                elif trans_name == 'boxcox':
                    pt = PowerTransformer(method='box-cox')
                    pt.fit(y.reshape(-1, 1))
                    y_pred = pt.inverse_transform(y_pred_trans.reshape(-1, 1)).flatten()
                else:
                    y_pred = y_pred_trans
                
                r2 = r2_score(y, y_pred)
                
                print(f"     Found {len(equations)} equations")
                print(f"     R² = {r2:.4f}")
                print(f"     Contains bacteria: {'✓' if 'x0' in best_eq_candidate else '✗'}")
                
                # Save intermediate result
                result_file = os.path.join(output_dir, f"result_{feat_name}_{trans_name}.txt")
                with open(result_file, 'w') as f:
                    f.write(f"Feature set: {feat_name}\n")
                    f.write(f"Transformation: {trans_name}\n")
                    f.write(f"R²: {r2:.6f}\n")
                    f.write(f"Equation:\n{best_eq_candidate}\n")
                
                if r2 > best_r2:
                    best_r2 = r2
                    best_equation = best_eq_candidate
                    best_model = model
                    print(f"     🎉 NEW BEST! R² = {r2:.4f}")
                    
        except Exception as e:
            print(f"     ❌ Failed: {e}")

# ==============================
# FINAL RESULTS
# ==============================
print("\n" + "⭐"*70)
print("ULTIMATE OPTIMIZATION RESULTS")
print("⭐"*70)
print(f"Initial R²: 0.7749")
print(f"Final R²: {best_r2:.4f}")
print(f"Improvement: +{(best_r2 - 0.7749)*100:.2f}%")
print(f"Gap to theoretical max: {0.9173 - best_r2:.4f}")

if best_equation:
    print(f"\n🌟 BEST EQUATION FOUND:")
    print(f"{best_equation}")
    
    # Save final equation
    final_path = os.path.join(output_dir, "ultimate_best_equation.txt")
    with open(final_path, 'w') as f:
        f.write(f"R²: {best_r2:.6f}\n")
        f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"Equation:\n{best_equation}\n")
    
    print(f"\n✅ Final equation saved to: {final_path}")
    
    # Check if it uses bacteria
    if 'x0' in best_equation:
        print("✅ Bacteria (Cbac) is included!")
    else:
        print("⚠️ Bacteria not in best equation")

# ==============================
# STRATEGY 5: ENSEMBLE PREDICTION
# ==============================
print("\n" + "="*70)
print("STRATEGY 5: ENSEMBLE PREDICTION")
print("="*70)

# If we have multiple good models, ensemble them
if best_model and hasattr(best_model, 'equations_'):
    equations = best_model.equations_
    
    # Take top 5 equations
    top5 = equations.sort_values('loss').head(5)
    
    print("\nCreating ensemble from top 5 equations...")
    
    # This would require saving multiple models
    print("Note: Ensemble prediction requires saving multiple models")
    print("Consider using the best single equation for simplicity")

print("\n" + "="*70)
print("RECOMMENDATIONS")
print("="*70)
print(f"""
Based on the results:

1. Current best R² = {best_r2:.4f}
2. Theoretical maximum = 0.9173

If R² < 0.85:
   • The relationship may be inherently noisy
   • Consider collecting more data
   • Try different feature combinations

If R² ≈ 0.85-0.90:
   • Great result! Close to theoretical maximum
   • Use the current best equation

If R² > 0.90:
   • Excellent! You've matched ML performance
""")

print("\n" + "="*70)
print(f"✅ All results saved to: {output_dir}")
print("="*70)