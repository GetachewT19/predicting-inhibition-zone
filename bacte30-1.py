# ==============================
# COMPLETE DIZ ANALYSIS - START TO FINISH
# ==============================
# INTENSIVE DIZ OPTIMIZATION - TARGET R² > 0.85
# ==============================

import pandas as pd
import numpy as np
from pysr import PySRRegressor
from sklearn.preprocessing import RobustScaler, PowerTransformer, PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import warnings
import os
import pickle
import time
from datetime import datetime

warnings.filterwarnings('ignore')

# Configuration
data_path = r"C:\Users\Amare\Desktop\getachewt\datato232-81.csv"
output_dir = r"C:\Users\Amare\Desktop\getachewt_intensive"
os.makedirs(output_dir, exist_ok=True)

# Create subdirectories
models_dir = os.path.join(output_dir, "models")
plots_dir = os.path.join(output_dir, "plots")
results_dir = os.path.join(output_dir, "results")
os.makedirs(models_dir, exist_ok=True)
os.makedirs(plots_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)

print("="*70)
print("INTENSIVE DIZ OPTIMIZATION")
print(f"Current best: 0.7663")
print(f"Target: > 0.85")
print(f"Theoretical max (from ML): 0.9173")
print("="*70)
print(f"Start time: {datetime.now()}")

# Load and prepare data
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
# STRATEGY 1: FEATURE ENGINEERING
# ==============================
print("\n" + "="*70)
print("STRATEGY 1: FEATURE ENGINEERING")
print("="*70)

# Create polynomial features (interactions)
poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
X_poly = poly.fit_transform(X)
feature_names = poly.get_feature_names_out(features)
print(f"Created {X_poly.shape[1]} interaction features: {feature_names}")

# Combine with original features
X_enhanced = np.hstack([X, X_poly])
enhanced_names = list(features) + list(feature_names)
print(f"Enhanced feature set: {len(enhanced_names)} features")

# Scale enhanced features
scaler_enhanced = RobustScaler()
X_enhanced_scaled = scaler_enhanced.fit_transform(X_enhanced)

# Save enhanced scaler
scaler_path = os.path.join(models_dir, "scaler_enhanced.pkl")
with open(scaler_path, 'wb') as f:
    pickle.dump(scaler_enhanced, f)
print(f"✅ Enhanced scaler saved")

# ==============================
# STRATEGY 2: TARGET TRANSFORMATIONS
# ==============================
print("\n" + "="*70)
print("STRATEGY 2: TARGET TRANSFORMATIONS")
print("="*70)

transformations = {
    'original': y,
    'log': np.log1p(y - y.min() + 1),
    'sqrt': np.sqrt(y - y.min() + 1),
}

# Box-Cox requires positive data
if (y > 0).all():
    pt_box = PowerTransformer(method='box-cox')
    y_boxcox = pt_box.fit_transform(y.reshape(-1, 1)).flatten()
    transformations['boxcox'] = y_boxcox

# Yeo-Johnson works for any data
pt_yj = PowerTransformer(method='yeo-johnson')
y_yeojohnson = pt_yj.fit_transform(y.reshape(-1, 1)).flatten()
transformations['yeojohnson'] = y_yeojohnson

# Test each transformation with Extra Trees
print("\nTesting transformations with Extra Trees:")
best_trans_r2 = 0
best_trans_name = 'original'
best_y_trans = y
best_pt = None

for name, y_trans in transformations.items():
    et = ExtraTreesRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    scores = cross_val_score(et, X_enhanced_scaled, y_trans, cv=5, scoring='r2')
    mean_r2 = scores.mean()
    print(f"  {name}: {mean_r2:.4f} (+/- {scores.std()*2:.4f})")
    
    if mean_r2 > best_trans_r2:
        best_trans_r2 = mean_r2
        best_trans_name = name
        best_y_trans = y_trans
        if name == 'boxcox':
            best_pt = pt_box
        elif name == 'yeojohnson':
            best_pt = pt_yj

print(f"\n✅ Best transformation: {best_trans_name} (R² = {best_trans_r2:.4f})")

# ==============================
# STRATEGY 3: MULTIPLE PySR CONFIGURATIONS
# ==============================
print("\n" + "="*70)
print("STRATEGY 3: AGGRESSIVE PySR SEARCH")
print("="*70)

# Aggressive configurations
configs = [
    {
        'name': 'Aggressive_1',
        'niterations': 3000,
        'populations': 20,
        'population_size': 300,
        'maxsize': 35,
        'parsimony': 0.0005,
        'binary': ["+", "-", "*", "/", "^"],
        'unary': ["exp", "log", "sqrt", "square", "cube"],
    },
    {
        'name': 'Aggressive_2',
        'niterations': 4000,
        'populations': 25,
        'population_size': 400,
        'maxsize': 40,
        'parsimony': 0.0001,
        'binary': ["+", "-", "*", "/", "^"],
        'unary': ["exp", "log", "sqrt", "square", "cube", "sin", "cos"],
    },
    {
        'name': 'Ultimate',
        'niterations': 5000,
        'populations': 30,
        'population_size': 500,
        'maxsize': 45,
        'parsimony': 0.00005,
        'binary': ["+", "-", "*", "/", "^"],
        'unary': ["exp", "log", "sqrt", "square", "cube", "sin", "cos", "tanh", "abs"],
    }
]

best_r2 = 0.7663  # Current best
best_equation = None
best_model = None
best_config_name = ""
all_results = []

for config in configs:
    print(f"\n🚀 Running {config['name']}...")
    print(f"   Iterations: {config['niterations']}, Maxsize: {config['maxsize']}")
    print(f"   This will take ~{config['niterations']/1000:.1f} hours...")
    
    model = PySRRegressor(
        niterations=config['niterations'],
        populations=config['populations'],
        population_size=config['population_size'],
        maxsize=config['maxsize'],
        parsimony=config['parsimony'],
        binary_operators=config['binary'],
        unary_operators=config['unary'],
        procs=8,  # Use all available cores
        multithreading=True,
        loss="L2DistLoss()",
        denoise=True,
        progress=True,
        random_state=42,
        timeout_in_seconds=28800,  # 8 hours max
        temp_equation_file=True,
        warm_start=False,
        turbo=True,
        should_optimize_constants=True,
        optimizer_algorithm="BFGS",
        optimizer_nrestarts=20,
        variable_names=enhanced_names[:10]  # Use first 10 feature names
    )
    
    try:
        start_time = time.time()
        model.fit(X_enhanced_scaled, best_y_trans)
        elapsed = time.time() - start_time
        
        if hasattr(model, 'equations_') and model.equations_ is not None:
            equations = model.equations_
            
            # Save raw equations
            eq_path = os.path.join(results_dir, f"equations_{config['name']}.csv")
            equations.to_csv(eq_path, index=False)
            
            # Get best equation from this run
            best_idx = equations['loss'].idxmin()
            best_eq_candidate = equations.loc[best_idx, 'equation']
            
            # Make predictions (need to inverse transform)
            y_pred_trans = model.predict(X_enhanced_scaled)
            
            # Inverse transform
            if best_trans_name == 'log':
                y_pred = np.expm1(y_pred_trans) + y.min() - 1
            elif best_trans_name == 'sqrt':
                y_pred = y_pred_trans ** 2 + y.min() - 1
            elif best_trans_name == 'boxcox':
                y_pred = best_pt.inverse_transform(y_pred_trans.reshape(-1, 1)).flatten()
            elif best_trans_name == 'yeojohnson':
                y_pred = best_pt.inverse_transform(y_pred_trans.reshape(-1, 1)).flatten()
            else:
                y_pred = y_pred_trans
            
            r2 = r2_score(y, y_pred)
            
            print(f"\n   ✅ Time: {elapsed/60:.1f} minutes")
            print(f"   📊 Equations found: {len(equations)}")
            print(f"   📈 R²: {r2:.4f}")
            print(f"   🔬 Contains bacteria: {'✓' if 'x0' in best_eq_candidate else '✗'}")
            print(f"   📝 Best equation: {best_eq_candidate[:150]}...")
            
            # Save this config's results
            config_results = {
                'config': config['name'],
                'r2': r2,
                'equation': best_eq_candidate,
                'time_min': elapsed/60,
                'n_equations': len(equations)
            }
            all_results.append(config_results)
            
            # Save model
            model_path = os.path.join(models_dir, f"model_{config['name']}.pkl")
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            
            if r2 > best_r2:
                best_r2 = r2
                best_equation = best_eq_candidate
                best_model = model
                best_config_name = config['name']
                print(f"   🎉 NEW BEST! R² = {r2:.4f}")
                
                # Save best so far
                best_path = os.path.join(results_dir, "current_best.txt")
                with open(best_path, 'w') as f:
                    f.write(f"R²: {r2:.6f}\n")
                    f.write(f"Configuration: {config['name']}\n")
                    f.write(f"Transformation: {best_trans_name}\n")
                    f.write(f"Time: {datetime.now()}\n")
                    f.write(f"Equation:\n{best_eq_candidate}\n")
            
    except Exception as e:
        print(f"   ❌ Failed: {e}")

# ==============================
# FINAL RESULTS
# ==============================
print("\n" + "⭐"*70)
print("FINAL OPTIMIZATION RESULTS")
print("⭐"*70)
print(f"Initial R²: 0.7663")
print(f"Final R²: {best_r2:.4f}")
print(f"Improvement: +{(best_r2 - 0.7663)*100:.2f}%")
print(f"Gap to theoretical max (0.9173): {0.9173 - best_r2:.4f}")

if best_equation:
    print(f"\n🌟 BEST EQUATION FOUND:")
    print(f"{best_equation}")
    print(f"\nConfiguration: {best_config_name}")
    print(f"Transformation: {best_trans_name}")
    
    # Check for bacteria
    if 'x0' in best_equation:
        print("✅ Bacteria (Cbac) is included!")
    else:
        print("⚠️ Bacteria not in best equation")

# Save all results
results_df = pd.DataFrame(all_results)
results_df.to_csv(os.path.join(results_dir, "optimization_summary.csv"), index=False)

# Create comparison plot
if len(all_results) > 0:
    plt.figure(figsize=(10, 6))
    
    configs_list = [r['config'] for r in all_results]
    r2_list = [r['r2'] for r in all_results]
    
    colors = ['green' if r2 > 0.8 else 'orange' if r2 > 0.77 else 'red' for r2 in r2_list]
    bars = plt.bar(configs_list, r2_list, color=colors, edgecolor='black')
    
    # Add target line
    plt.axhline(y=0.85, color='blue', linestyle='--', label='Target (0.85)')
    plt.axhline(y=0.9173, color='purple', linestyle='--', label='ML Max (0.9173)')
    
    plt.ylabel('R² Score')
    plt.title('PySR Configuration Performance')
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, r2 in zip(bars, r2_list):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{r2:.4f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plot_path = os.path.join(plots_dir, "optimization_results.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"✅ Optimization plot saved to: {plot_path}")

# Final summary
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"\n📁 All files saved to: {output_dir}")
print(f"\n📊 Best R² achieved: {best_r2:.4f}")
print(f"   Improvement: +{(best_r2 - 0.7663)*100:.2f}%")
print(f"\n⏱️  End time: {datetime.now()}")

# Save final results
final_path = os.path.join(results_dir, "final_optimization_results.txt")
with open(final_path, 'w') as f:
    f.write("="*70 + "\n")
    f.write("INTENSIVE OPTIMIZATION RESULTS\n")
    f.write("="*70 + "\n\n")
    f.write(f"Initial R²: 0.7663\n")
    f.write(f"Final R²: {best_r2:.6f}\n")
    f.write(f"Improvement: {(best_r2 - 0.7663)*100:.2f}%\n")
    f.write(f"Theoretical max: 0.9173\n")
    f.write(f"Gap to max: {0.9173 - best_r2:.4f}\n\n")
    f.write(f"Best Configuration: {best_config_name}\n")
    f.write(f"Transformation: {best_trans_name}\n\n")
    f.write(f"Best Equation:\n{best_equation}\n")