# ==============================
# DIZ EQUATION ANALYZER AND OPTIMIZER
# ==============================

import pandas as pd
import numpy as np
from pysr import PySRRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score
import warnings
import os
import re

warnings.filterwarnings('ignore')

# Configuration
data_path = r"C:\Users\Amare\Desktop\getachewt\datato232-81.csv"
output_dir = r"C:\Users\Amare\Desktop\diz_ultimate"
os.makedirs(output_dir, exist_ok=True)

print("="*70)
print("DIZ EQUATION ANALYZER AND OPTIMIZER")
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
# RECOVER AND ANALYZE FOUND EQUATIONS
# ==============================

print("\n" + "="*70)
print("ANALYZING RECOVERED EQUATIONS")
print("="*70)

# Read the recovered equations file
recovered_file = r"C:\Users\Amare\Desktop\found_equations_091257.csv"
if os.path.exists(recovered_file):
    recovered_df = pd.read_csv(recovered_file)
    print(f"\nFound {len(recovered_df)} equations in recovery file")
    
    # Test each equation
    results = []
    
    for idx, row in recovered_df.iterrows():
        equation_str = row['Equation']
        complexity = row['Complexity'] if 'Complexity' in row else 'N/A'
        
        # Try to evaluate the equation
        try:
            # Create a safe evaluation environment
            safe_dict = {
                'x0': X[:, 0], 'x1': X[:, 1], 'x2': X[:, 2],
                'sin': np.sin, 'cos': np.cos, 'tan': np.tan,
                'exp': np.exp, 'log': np.log, 'sqrt': np.sqrt,
                'abs': np.abs, 'square': np.square, 'cube': lambda x: x**3,
                'tanh': np.tanh
            }
            
            # Clean equation string
            eq_clean = equation_str.replace('^', '**')
            
            # Evaluate
            y_pred = eval(eq_clean, {"__builtins__": {}}, safe_dict)
            
            # Handle inf/nan
            y_pred = np.nan_to_num(y_pred, nan=np.nanmean(y), posinf=np.nanmax(y), neginf=np.nanmin(y))
            
            # Calculate R²
            r2 = r2_score(y, y_pred)
            
            # Check if bacteria is included
            has_bacteria = 'x0' in equation_str
            
            results.append({
                'Equation': equation_str[:100] + '...',
                'Complexity': complexity,
                'R2': r2,
                'Has_Bacteria': has_bacteria
            })
            
            print(f"\nEquation {idx+1}:")
            print(f"  R² = {r2:.4f}")
            print(f"  Contains bacteria: {'✓' if has_bacteria else '✗'}")
            
        except Exception as e:
            print(f"\nEquation {idx+1}: Failed to evaluate - {e}")
    
    # Sort results
    results_df = pd.DataFrame(results)
    if not results_df.empty:
        results_df = results_df.sort_values('R2', ascending=False)
        
        print("\n" + "="*70)
        print("TOP 5 RECOVERED EQUATIONS")
        print("="*70)
        print(results_df.head(5).to_string(index=False))
        
        # Save analyzed results
        analyzed_file = os.path.join(output_dir, "analyzed_equations.csv")
        results_df.to_csv(analyzed_file, index=False)
        print(f"\n✅ Analyzed equations saved to: {analyzed_file}")

# ==============================
# FOCUSED OPTIMIZATION
# ==============================

print("\n" + "="*70)
print("FOCUSED OPTIMIZATION")
print("="*70)

best_r2 = 0.7749
best_equation = None

# Create enhanced features
X_enhanced = np.column_stack([
    X,  # Original
    X[:, 0] * X[:, 1],  # Cbac * CAgNP
    X[:, 0] * X[:, 2],  # Cbac * Ra
    X[:, 1] * X[:, 2],  # CAgNP * Ra
    X[:, 0] / (X[:, 1] + 1e-10),  # Cbac/CAgNP
    X[:, 1] / (X[:, 2] + 1e-10),  # CAgNP/Ra
    X[:, 0] ** 2,  # Cbac²
])

feature_names = ['Cbac', 'CAgNP', 'Ra', 
                 'Cbac*CAgNP', 'Cbac*Ra', 'CAgNP*Ra',
                 'Cbac/CAgNP', 'CAgNP/Ra', 'Cbac²']

# Scale features
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X_enhanced)

# Try multiple configurations in sequence
configs = [
    {
        'name': 'Simple',
        'niterations': 200,
        'populations': 5,
        'population_size': 50,
        'maxsize': 15,
        'binary_operators': ["+", "*"],
        'unary_operators': ["square"],
    },
    {
        'name': 'Medium',
        'niterations': 500,
        'populations': 10,
        'population_size': 100,
        'maxsize': 25,
        'binary_operators': ["+", "-", "*", "/"],
        'unary_operators': ["square", "sqrt", "exp", "log"],
    },
    {
        'name': 'Complex',
        'niterations': 1000,
        'populations': 15,
        'population_size': 150,
        'maxsize': 35,
        'binary_operators': ["+", "-", "*", "/", "^"],
        'unary_operators': ["exp", "log", "sqrt", "square", "sin", "cos", "tanh"],
    }
]

for config in configs:
    print(f"\n{'='*50}")
    print(f"RUNNING: {config['name']} Configuration")
    print(f"{'='*50}")
    
    try:
        model = PySRRegressor(
            niterations=config['niterations'],
            populations=config['populations'],
            population_size=config['population_size'],
            maxsize=config['maxsize'],
            parsimony=0.001,
            binary_operators=config['binary_operators'],
            unary_operators=config['unary_operators'],
            procs=2,  # Conservative
            multithreading=True,
            loss="L2DistLoss()",
            denoise=True,
            progress=True,
            random_state=42,
            timeout_in_seconds=1800,  # 30 minutes
            temp_equation_file=True,
            should_optimize_constants=True,
            optimizer_nrestarts=5,
            batching=True,
            batch_size=50
        )
        
        print(f"Training...")
        model.fit(X_scaled, y)
        
        if hasattr(model, 'equations_') and model.equations_ is not None:
            equations = model.equations_
            print(f"Found {len(equations)} equations")
            
            # Get best equation
            best_idx = equations['loss'].idxmin()
            best_eq = equations.loc[best_idx, 'equation']
            
            # Make predictions
            y_pred = model.predict(X_scaled)
            r2 = r2_score(y, y_pred)
            
            # Check bacteria
            has_bacteria = 'x0' in best_eq
            
            print(f"R² = {r2:.4f}")
            print(f"Contains bacteria: {'✓' if has_bacteria else '✗'}")
            print(f"Equation: {best_eq[:100]}...")
            
            # Save result
            if r2 > best_r2:
                best_r2 = r2
                best_equation = best_eq
                print(f"🎉 NEW BEST! R² = {r2:.4f}")
                
    except Exception as e:
        print(f"Error: {e}")
        continue

# ==============================
# FINAL RESULTS
# ==============================

print("\n" + "⭐"*70)
print("FINAL RESULTS")
print("⭐"*70)
print(f"Best R² Achieved: {best_r2:.4f}")
print(f"Improvement: +{(best_r2 - 0.7749)*100:.2f}%")
print(f"Gap to theoretical max: {0.9173 - best_r2:.4f}")

if best_equation:
    print(f"\n🏆 BEST EQUATION:")
    print(f"{best_equation}")
    
    # Check bacteria
    if 'x0' in best_equation:
        print("\n✅ Bacteria (Cbac) is included!")
    else:
        print("\n⚠️ Bacteria NOT in best equation!")
    
    # Create final equation with feature names
    final_eq = best_equation
    for i, name in enumerate(feature_names):
        final_eq = final_eq.replace(f'x{i}', name)
    
    # Save final results
    final_path = os.path.join(output_dir, "final_best_equation.txt")
    with open(final_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("DIZ EQUATION DISCOVERY RESULTS\n")
        f.write("="*60 + "\n\n")
        f.write(f"Date: {pd.Timestamp.now()}\n")
        f.write(f"Dataset size: {len(y)} samples\n")
        f.write(f"R² Score: {best_r2:.6f}\n")
        f.write(f"R² Improvement: +{(best_r2 - 0.7749)*100:.2f}%\n")
        f.write(f"Gap to theoretical maximum: {0.9173 - best_r2:.4f}\n\n")
        f.write("="*60 + "\n")
        f.write("BEST EQUATION (with feature names):\n")
        f.write("="*60 + "\n")
        f.write(f"DIZ = {final_eq}\n\n")
        f.write("="*60 + "\n")
        f.write("BEST EQUATION (original format):\n")
        f.write("="*60 + "\n")
        f.write(f"{best_equation}\n\n")
        f.write("="*60 + "\n")
        f.write("FEATURE MAPPING:\n")
        f.write("="*60 + "\n")
        for i, name in enumerate(feature_names):
            f.write(f"x{i} = {name}\n")
    
    print(f"\n✅ Final results saved to: {final_path}")

# ==============================
# VALIDATION
# ==============================

if best_equation:
    print("\n" + "="*70)
    print("VALIDATION")
    print("="*70)
    
    # Create safe evaluation environment
    safe_dict = {
        'x0': X[:, 0], 'x1': X[:, 1], 'x2': X[:, 2],
        'sin': np.sin, 'cos': np.cos, 'tan': np.tan,
        'exp': np.exp, 'log': np.log, 'sqrt': np.sqrt,
        'abs': np.abs, 'square': np.square, 'cube': lambda x: x**3,
        'tanh': np.tanh
    }
    
    try:
        # Evaluate best equation
        eq_clean = best_equation.replace('^', '**')
        y_pred = eval(eq_clean, {"__builtins__": {}}, safe_dict)
        y_pred = np.nan_to_num(y_pred, nan=np.nanmean(y))
        
        final_r2 = r2_score(y, y_pred)
        
        print(f"Final verified R²: {final_r2:.4f}")
        
        # Calculate statistics
        residuals = y - y_pred
        mae = np.mean(np.abs(residuals))
        rmse = np.sqrt(np.mean(residuals**2))
        
        print(f"MAE: {mae:.4f}")
        print(f"RMSE: {rmse:.4f}")
        
        # Save predictions
        pred_df = pd.DataFrame({
            'Actual': y,
            'Predicted': y_pred,
            'Residual': residuals
        })
        pred_path = os.path.join(output_dir, "predictions.csv")
        pred_df.to_csv(pred_path, index=False)
        print(f"\n✅ Predictions saved to: {pred_path}")
        
    except Exception as e:
        print(f"Validation error: {e}")

print("\n" + "="*70)
print("✅ OPTIMIZATION COMPLETE")
print(f"📁 All files saved to: {output_dir}")
print("="*70)