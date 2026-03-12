# ==============================
# ULTIMATE DIZ OPTIMIZATION - FIXED VERSION
# ==============================

import pandas as pd
import numpy as np
from pysr import PySRRegressor
from sklearn.preprocessing import RobustScaler, PowerTransformer, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import warnings
import os
import time
import gc

warnings.filterwarnings('ignore')

# Configuration
data_path = r"C:\Users\Amare\Desktop\getachewt\datato232-81.csv"
output_dir = r"C:\Users\Amare\Desktop\diz_ultimate"
os.makedirs(output_dir, exist_ok=True)

print("="*70)
print("ULTIMATE DIZ OPTIMIZATION - FIXED VERSION")
print(f"Starting from: 0.7749")
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
print(f"Features: {features}")

# ==============================
# SIMPLE BUT EFFECTIVE FEATURE ENGINEERING
# ==============================
print("\n" + "="*70)
print("FEATURE ENGINEERING")
print("="*70)

# Create enhanced feature set
X_enhanced = X.copy()
feature_names = features.copy()

# Add interaction terms
X_enhanced = np.column_stack([
    X_enhanced,
    X[:, 0] * X[:, 1],  # Cbac * CAgNP
    X[:, 0] * X[:, 2],  # Cbac * Ra
    X[:, 1] * X[:, 2],  # CAgNP * Ra
])
feature_names.extend(['Cbac*CAgNP', 'Cbac*Ra', 'CAgNP*Ra'])

# Add ratio terms (with small epsilon to avoid division by zero)
eps = 1e-10
X_enhanced = np.column_stack([
    X_enhanced,
    X[:, 0] / (X[:, 1] + eps),  # Cbac/CAgNP
    X[:, 1] / (X[:, 2] + eps),  # CAgNP/Ra
    X[:, 0] / (X[:, 2] + eps),  # Cbac/Ra
])
feature_names.extend(['Cbac/CAgNP', 'CAgNP/Ra', 'Cbac/Ra'])

# Add squared terms
X_enhanced = np.column_stack([
    X_enhanced,
    X[:, 0]**2,  # Cbac²
    X[:, 1]**2,  # CAgNP²
    X[:, 2]**2,  # Ra²
])
feature_names.extend(['Cbac²', 'CAgNP²', 'Ra²'])

print(f"Original features: 3")
print(f"Enhanced features: {X_enhanced.shape[1]}")

# Scale features
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X_enhanced)

# ==============================
# OPTIMIZED PySR CONFIGURATION
# ==============================
print("\n" + "="*70)
print("OPTIMIZED PySR SEARCH")
print("="*70)

best_r2 = 0.7749
best_equation = None
best_model = None

# Try different configurations
configs = [
    {
        'name': 'Balanced Search',
        'niterations': 1000,
        'populations': 15,
        'population_size': 50,
        'maxsize': 30,
        'parsimony': 0.001,
    },
    {
        'name': 'Complex Search',
        'niterations': 2000,
        'populations': 20,
        'population_size': 100,
        'maxsize': 40,
        'parsimony': 0.0001,
    }
]

for config in configs:
    print(f"\n{'='*50}")
    print(f"CONFIG: {config['name']}")
    print(f"{'='*50}")
    
    try:
        # Create model with current config
        model = PySRRegressor(
            niterations=config['niterations'],
            populations=config['populations'],
            population_size=config['population_size'],
            maxsize=config['maxsize'],
            parsimony=config['parsimony'],
            binary_operators=["+", "-", "*", "/", "^"],
            unary_operators=[
                "exp", "log", "sqrt", "square", "cube",
                "sin", "cos", "tanh", "abs"
            ],
            procs=4,  # Reduced to avoid memory issues
            multithreading=True,
            loss="L2DistLoss()",
            denoise=True,
            progress=True,
            random_state=42,
            timeout_in_seconds=3600,  # 1 hour timeout
            temp_equation_file=True,
            turbo=True,
            should_optimize_constants=True,
            optimizer_algorithm="BFGS",
            optimizer_nrestarts=10,
            warm_start=False,
            batching=True,
            batch_size=50
        )
        
        # Fit the model
        print("  Training... (this may take a while)")
        model.fit(X_scaled, y)
        
        if hasattr(model, 'equations_') and model.equations_ is not None:
            equations = model.equations_
            print(f"  Found {len(equations)} equations")
            
            # Get best equation
            best_idx = equations['loss'].idxmin()
            best_eq_candidate = equations.loc[best_idx, 'equation']
            
            # Make predictions
            y_pred = model.predict(X_scaled)
            r2 = r2_score(y, y_pred)
            
            # Check if bacteria is included
            has_bacteria = 'x0' in best_eq_candidate
            
            print(f"  R² = {r2:.4f}")
            print(f"  Contains bacteria: {'✓' if has_bacteria else '✗'}")
            print(f"  Equation: {best_eq_candidate[:100]}...")
            
            # Save result
            result_file = os.path.join(output_dir, f"result_{config['name'].replace(' ', '_')}.txt")
            with open(result_file, 'w') as f:
                f.write(f"Configuration: {config['name']}\n")
                f.write(f"R²: {r2:.6f}\n")
                f.write(f"Contains bacteria: {has_bacteria}\n")
                f.write(f"Equation:\n{best_eq_candidate}\n")
            
            if r2 > best_r2:
                best_r2 = r2
                best_equation = best_eq_candidate
                best_model = model
                print(f"  🎉 NEW BEST! R² = {r2:.4f}")
        
        # Clean up to prevent memory issues
        del model
        gc.collect()
        
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        continue

# ==============================
# FINAL RESULTS
# ==============================
print("\n" + "⭐"*70)
print("FINAL OPTIMIZATION RESULTS")
print("⭐"*70)
print(f"Starting R²: 0.7749")
print(f"Final R²: {best_r2:.4f}")
print(f"Improvement: +{(best_r2 - 0.7749)*100:.2f}%")
print(f"Gap to theoretical max: {0.9173 - best_r2:.4f}")

if best_equation:
    print(f"\n🌟 BEST EQUATION:")
    print(f"{best_equation}")
    
    # Check bacteria inclusion
    if 'x0' in best_equation:
        print("\n✅ Bacteria (Cbac) is included!")
    else:
        print("\n⚠️ WARNING: Bacteria NOT in best equation!")
    
    # Save final equation
    final_path = os.path.join(output_dir, "best_equation_final.txt")
    with open(final_path, 'w') as f:
        f.write(f"R² Score: {best_r2:.6f}\n")
        f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"Features used (x0=Cbac, x1=CAgNP, x2=Ra):\n")
        f.write(f"{best_equation}\n\n")
        f.write(f"Feature mapping:\n")
        for i, name in enumerate(feature_names):
            f.write(f"x{i} = {name}\n")
    
    print(f"\n✅ Final equation saved to: {final_path}")

# ==============================
# RECOMMENDATIONS
# ==============================
print("\n" + "="*70)
print("RECOMMENDATIONS")
print("="*70)

if best_r2 < 0.85:
    print("""
📊 R² < 0.85 - Room for Improvement:
   • The relationship may be inherently noisy
   • Consider collecting more data points
   • Try different feature combinations
   • Focus on bacteria interactions
    """)
elif best_r2 < 0.90:
    print("""
📊 R² ≈ 0.85-0.90 - Good Result:
   • Close to theoretical maximum!
   • Use the current best equation
   • Validate on test data
   • Consider publishing results
    """)
else:
    print("""
📊 R² > 0.90 - Excellent Result:
   • You've matched ML performance!
   • Equation is highly reliable
   • Ready for practical applications
    """)

# Show bacteria importance
print("\n" + "="*70)
print("BACTERIA IMPORTANCE ANALYSIS")
print("="*70)
print("""
To verify bacteria importance:
1. Check if 'x0' appears in the equation
2. If not, bacteria may not be critical
3. Consider domain expertise
""")

print("\n" + "="*70)
print(f"✅ All results saved to: {output_dir}")
print("="*70)