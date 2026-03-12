# ==============================
# Fixed PySR Diagnostic - Corrected maxsize
# ==============================

import pandas as pd
import numpy as np
from pysr import PySRRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import r2_score
import warnings
import os
import sys

warnings.filterwarnings('ignore')

# ------------------------------
# Configuration
# ------------------------------
data_path = r"C:\Users\Amare\Desktop\worku\datato232-81.csv"
output_dir = r"C:\Users\Amare\Desktop\pysr_diagnostic_fixed"
os.makedirs(output_dir, exist_ok=True)

print("="*70)
print("FIXED PySR DIAGNOSTIC - CORRECTED MAXSIZE")
print("="*70)

# ------------------------------
# Load data
# ------------------------------
df = pd.read_csv(data_path)
print(f"\n📊 Dataset shape: {df.shape}")
print(f"📋 Columns: {df.columns.tolist()}")

target = 'DIZ'
if target not in df.columns:
    print(f"\n❌ ERROR: Target column '{target}' not found!")
    exit(1)

features = [col for col in df.columns if col != target]
print(f"\n✅ Target: {target}")
print(f"✅ Features ({len(features)}): {features}")

# ------------------------------
# Clean data
# ------------------------------
print("\n" + "-"*70)
print("DATA CLEANING")
print("-"*70)

df_clean = df.copy()

# Convert to numeric
for col in [target] + features:
    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

# Drop NaNs
initial_rows = len(df_clean)
df_clean = df_clean.dropna()
print(f"Rows dropped: {initial_rows - len(df_clean)}")
print(f"Final rows: {len(df_clean)}")

if len(df_clean) < 10:
    print("\n❌ ERROR: Insufficient data!")
    exit(1)

X = df_clean[features].values.astype(float)
y = df_clean[target].values.astype(float)

# ------------------------------
# Test 1: Simple synthetic data (verify PySR works)
# ------------------------------
print("\n" + "="*70)
print("TEST 1: PySR WITH SIMPLE SYNTHETIC DATA")
print("="*70)

# Create simple synthetic relationship
np.random.seed(42)
n_samples = 100
X_synth = np.random.randn(n_samples, 1)
y_synth = 2 * X_synth[:, 0] + 0.1 * np.random.randn(n_samples)

print(f"Synthetic data shape: X {X_synth.shape}, y {y_synth.shape}")

# FIXED: Increased maxsize to 7 (minimum required)
model_synth = PySRRegressor(
    niterations=100,
    populations=3,
    population_size=50,
    maxsize=7,  # Minimum required by PySR
    binary_operators=["+", "*"],
    unary_operators=[],
    loss="L2DistLoss()",
    progress=True,
    random_state=42,
    temp_equation_file=True,
    procs=1
)

try:
    print("\nFitting PySR on synthetic data...")
    model_synth.fit(X_synth, y_synth)
    
    if hasattr(model_synth, 'equations_') and model_synth.equations_ is not None:
        equations = model_synth.equations_
        print(f"✅ Found {len(equations)} equations")
        if len(equations) > 0:
            print("\nTop equations:")
            print(equations[['complexity', 'loss', 'equation']].head())
            
            # Save results
            csv_path = os.path.join(output_dir, "synthetic_results.csv")
            equations.to_csv(csv_path, index=False)
            print(f"✅ Results saved to: {csv_path}")
    else:
        print("❌ No equations found for synthetic data!")
        
except Exception as e:
    print(f"❌ PySR failed on synthetic data: {e}")

# ------------------------------
# Test 2: Single best feature - CORRECTED
# ------------------------------
print("\n" + "="*70)
print("TEST 2: PySR WITH SINGLE BEST FEATURE")
print("="*70)

# Find feature with highest absolute correlation
correlations = []
for i, feat in enumerate(features):
    corr = np.corrcoef(X[:, i], y)[0, 1]
    correlations.append((feat, abs(corr), i))

best_feat = max(correlations, key=lambda x: x[1])
print(f"\nBest feature: {best_feat[0]} (|corr| = {best_feat[1]:.4f})")

# Use only that feature
X_best = X[:, best_feat[2]].reshape(-1, 1)

# Try different scalers with CORRECTED maxsize
for scaler_name, scaler in [('StandardScaler', StandardScaler()), 
                           ('RobustScaler', RobustScaler())]:
    print(f"\n{scaler_name} scaling:")
    X_scaled = scaler.fit_transform(X_best)
    
    # FIXED: All configs now use maxsize >= 7
    configs = [
        {
            'name': 'Minimal',
            'niterations': 200,
            'binary': ["+", "-", "*"],
            'unary': [],
            'maxsize': 7
        },
        {
            'name': 'With sqrt',
            'niterations': 300,
            'binary': ["+", "-", "*", "/"],
            'unary': ["sqrt"],
            'maxsize': 8
        },
        {
            'name': 'With square',
            'niterations': 300,
            'binary': ["+", "-", "*", "/"],
            'unary': ["square"],
            'maxsize': 8
        },
        {
            'name': 'Full',
            'niterations': 400,
            'binary': ["+", "-", "*", "/", "^"],
            'unary': ["sqrt", "square", "exp", "log"],
            'maxsize': 10
        }
    ]
    
    for config in configs:
        print(f"\n  Trying {config['name']} (maxsize={config['maxsize']})...")
        
        model = PySRRegressor(
            niterations=config['niterations'],
            populations=3,
            population_size=50,
            maxsize=config['maxsize'],
            binary_operators=config['binary'],
            unary_operators=config['unary'],
            procs=1,
            loss="L2DistLoss()",
            random_state=42,
            temp_equation_file=True,
            progress=False  # Reduce output
        )
        
        try:
            model.fit(X_scaled, y)
            
            if hasattr(model, 'equations_') and model.equations_ is not None:
                if len(model.equations_) > 0:
                    print(f"    ✅ Found {len(model.equations_)} equations")
                    best_idx = model.equations_['loss'].idxmin()
                    print(f"    Best: {model.equations_.loc[best_idx, 'equation']}")
                    
                    # Save results
                    filename = f"single_feat_{scaler_name}_{config['name']}.csv"
                    csv_path = os.path.join(output_dir, filename)
                    model.equations_.to_csv(csv_path, index=False)
                else:
                    print(f"    ⚠ No equations found")
            else:
                print(f"    ⚠ No equations attribute")
                
        except Exception as e:
            print(f"    ❌ Failed: {e}")

# ------------------------------
# Test 3: All features with CORRECTED maxsize
# ------------------------------
print("\n" + "="*70)
print("TEST 3: PySR WITH ALL FEATURES")
print("="*70)

# Use RobustScaler
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)

# Multiple configurations for all features
all_configs = [
    {
        'name': 'Simple',
        'niterations': 300,
        'binary': ["+", "-", "*"],
        'unary': [],
        'maxsize': 7,
        'populations': 3
    },
    {
        'name': 'Medium',
        'niterations': 500,
        'binary': ["+", "-", "*", "/"],
        'unary': ["sqrt", "square"],
        'maxsize': 10,
        'populations': 5
    },
    {
        'name': 'Complex',
        'niterations': 800,
        'binary': ["+", "-", "*", "/", "^"],
        'unary': ["sqrt", "square", "exp", "log"],
        'maxsize': 15,
        'populations': 8
    }
]

for config in all_configs:
    print(f"\nTrying {config['name']} configuration...")
    print(f"  maxsize={config['maxsize']}, populations={config['populations']}")
    
    model_all = PySRRegressor(
        niterations=config['niterations'],
        populations=config['populations'],
        population_size=80,
        maxsize=config['maxsize'],
        binary_operators=config['binary'],
        unary_operators=config['unary'],
        procs=2,
        loss="L2DistLoss()",
        random_state=42,
        temp_equation_file=True,
        progress=True,
        update_verbosity=1
    )
    
    try:
        model_all.fit(X_scaled, y)
        
        if hasattr(model_all, 'equations_') and model_all.equations_ is not None:
            equations = model_all.equations_
            if len(equations) > 0:
                print(f"  ✅ Found {len(equations)} equations")
                
                # Show top 5
                print("\n  Top 5 equations:")
                top5 = equations.sort_values('loss').head(5)
                for i, (idx, row) in enumerate(top5.iterrows()):
                    print(f"    {i+1}. C={row['complexity']}, Loss={row['loss']:.4f}")
                    print(f"       {row['equation'][:80]}...")
                
                # Calculate R²
                y_pred = model_all.predict(X_scaled)
                r2 = r2_score(y, y_pred)
                print(f"\n  📈 R² Score: {r2:.4f}")
                
                # Save results
                filename = f"all_features_{config['name']}.csv"
                csv_path = os.path.join(output_dir, filename)
                equations.to_csv(csv_path, index=False)
                print(f"  ✅ Results saved to: {csv_path}")
            else:
                print(f"  ⚠ No equations found")
        else:
            print(f"  ⚠ No equations attribute")
            
    except Exception as e:
        print(f"  ❌ PySR failed: {e}")

# ------------------------------
# Test 4: With normalized target
# ------------------------------
print("\n" + "="*70)
print("TEST 4: PySR WITH NORMALIZED TARGET")
print("="*70)

# Normalize target
y_normalized = (y - y.mean()) / y.std()

model_norm = PySRRegressor(
    niterations=400,
    populations=4,
    population_size=70,
    maxsize=8,
    binary_operators=["+", "-", "*", "/"],
    unary_operators=["sqrt", "square"],
    procs=2,
    loss="L2DistLoss()",
    random_state=42,
    temp_equation_file=True,
    progress=True
)

try:
    print("\nFitting PySR with normalized target...")
    model_norm.fit(X_scaled, y_normalized)
    
    if hasattr(model_norm, 'equations_') and model_norm.equations_ is not None:
        equations = model_norm.equations_
        if len(equations) > 0:
            print(f"✅ Found {len(equations)} equations")
            print("\nTop equations:")
            print(equations[['complexity', 'loss', 'equation']].head())
            
            # Save results
            csv_path = os.path.join(output_dir, "normalized_target.csv")
            equations.to_csv(csv_path, index=False)
            print(f"✅ Results saved to: {csv_path}")
            
            # Transform predictions back
            y_pred_norm = model_norm.predict(X_scaled)
            y_pred = y_pred_norm * y.std() + y.mean()
            r2 = r2_score(y, y_pred)
            print(f"\n📈 R² Score (original scale): {r2:.4f}")
        else:
            print("❌ No equations found!")
    else:
        print("❌ No equations attribute!")
        
except Exception as e:
    print(f"❌ PySR failed: {e}")

# ------------------------------
# Summary
# ------------------------------
print("\n" + "="*70)
print("DIAGNOSTIC SUMMARY")
print("="*70)
print(f"\n✅ All tests completed with corrected maxsize (minimum 7)")
print(f"\nAll outputs saved to: {output_dir}")
print("\nCheck the CSV files for results:")
print(f"  - synthetic_results.csv")
print(f"  - single_feat_*.csv")
print(f"  - all_features_*.csv")
print(f"  - normalized_target.csv")
print("\n" + "="*70)