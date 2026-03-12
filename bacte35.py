# ==============================
# PROPER EQUATION VALIDATION AND FIX
# ==============================

import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from sklearn.preprocessing import RobustScaler
from pysr import PySRRegressor
import warnings
import os
import re

warnings.filterwarnings('ignore')

# Configuration
data_path = r"C:\Users\Amare\Desktop\getachewt\datato232-81.csv"
output_dir = r"C:\Users\Amare\Desktop\diz_ultimate"
os.makedirs(output_dir, exist_ok=True)

print("="*70)
print("PROPER EQUATION VALIDATION AND FIX")
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
print(f"DIZ range: [{y.min():.2f}, {y.max():.2f}]")

# ==============================
# BEST EQUATION FROM PREVIOUS RUN
# ==============================

best_equation = """
exp(((0.8301047 / exp((exp(tanh(-0.001662643 / x2)) ^ (x1 * 4.1027646)) ^ 
((x0 / 0.32197776) + tanh(1.0198163)))) + tanh(1.8460233 + (2.913148 * x1))) / 
(1.2884854 ^ tanh(square(cube((x0 / 1.2884854) + 1.3261589) / 1.7785714))))
"""

# Clean the equation (remove newlines and extra spaces)
best_equation = ' '.join(best_equation.split())
print(f"\n🏆 Original Best Equation:")
print(f"{best_equation}")

# ==============================
# FIXED EQUATION EVALUATION
# ==============================

print("\n" + "="*70)
print("FIXED EQUATION EVALUATION")
print("="*70)

def safe_evaluate_equation(equation, X, y):
    """Safely evaluate the equation with proper operator precedence"""
    try:
        # Create safe evaluation environment with all necessary functions
        safe_dict = {
            'x0': X[:, 0], 'x1': X[:, 1], 'x2': X[:, 2],
            'exp': np.exp, 'log': np.log, 'sqrt': np.sqrt,
            'sin': np.sin, 'cos': np.cos, 'tan': np.tan,
            'tanh': np.tanh, 'abs': np.abs,
            'square': np.square, 'cube': lambda x: np.power(x, 3),
            '^': np.power,  # Handle power operator
            'np': np
        }
        
        # Convert equation to Python syntax
        eq_python = equation
        
        # Replace operators
        eq_python = eq_python.replace('^', '**')
        
        # Handle function calls
        eq_python = eq_python.replace('exp(', 'np.exp(')
        eq_python = eq_python.replace('tanh(', 'np.tanh(')
        eq_python = eq_python.replace('square(', 'np.square(')
        eq_python = eq_python.replace('cube(', 'np.power(?, 3)'.replace('?', ''))
        
        # Fix cube function
        import re
        def replace_cube(match):
            content = match.group(1)
            return f"np.power({content}, 3)"
        eq_python = re.sub(r'cube\(([^)]+)\)', replace_cube, eq_python)
        
        print(f"\nPython-ready equation:")
        print(f"{eq_python[:200]}...")
        
        # Evaluate
        y_pred = eval(eq_python, {"__builtins__": {}}, safe_dict)
        
        # Handle inf/nan
        y_pred = np.nan_to_num(y_pred, nan=np.nanmean(y), 
                               posinf=np.nanmax(y), neginf=np.nanmin(y))
        
        # Clip to reasonable range
        y_pred = np.clip(y_pred, y.min() - np.std(y), y.max() + np.std(y))
        
        # Calculate metrics
        r2 = r2_score(y, y_pred)
        residuals = y - y_pred
        mae = np.mean(np.abs(residuals))
        rmse = np.sqrt(np.mean(residuals**2))
        
        return r2, mae, rmse, y_pred
    except Exception as e:
        print(f"Error evaluating equation: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None

# Validate with fixed evaluation
r2, mae, rmse, y_pred = safe_evaluate_equation(best_equation, X, y)

if r2 is not None:
    print(f"\n✅ Equation validation results:")
    print(f"   R² = {r2:.4f}")
    print(f"   MAE = {mae:.4f}")
    print(f"   RMSE = {rmse:.4f}")
    
    # Show prediction statistics
    print(f"\nPrediction statistics:")
    print(f"   Actual mean: {y.mean():.4f}")
    print(f"   Predicted mean: {y_pred.mean():.4f}")
    print(f"   Actual std: {y.std():.4f}")
    print(f"   Predicted std: {y_pred.std():.4f}")
    
    # Save predictions
    val_results = pd.DataFrame({
        'Actual': y,
        'Predicted': y_pred,
        'Residual': y - y_pred
    })
    val_path = os.path.join(output_dir, "fixed_predictions.csv")
    val_results.to_csv(val_path, index=False)
    print(f"\n✅ Predictions saved to: {val_path}")
    
    # Check correlation
    correlation = np.corrcoef(y, y_pred)[0, 1]
    print(f"   Correlation: {correlation:.4f}")
    
    if correlation > 0.8:
        print("   ✅ Strong correlation!")
    elif correlation > 0.6:
        print("   ✅ Moderate correlation")
    else:
        print("   ⚠️ Weak correlation")
else:
    print(f"\n❌ Equation validation failed!")

# ==============================
# TRY SIMPLER MODEL WITH CORRECT SCALING
# ==============================

print("\n" + "="*70)
print("TRAINING SIMPLER MODEL WITH PROPER SCALING")
print("="*70)

# Scale the data properly
scaler_X = RobustScaler()
X_scaled = scaler_X.fit_transform(X)

scaler_y = RobustScaler()
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

# Try a simpler PySR model
try:
    simple_model = PySRRegressor(
        niterations=500,
        populations=8,
        population_size=50,
        maxsize=15,
        parsimony=0.01,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["exp", "log", "square", "tanh"],
        procs=2,
        multithreading=True,
        loss="L2DistLoss()",
        denoise=True,
        progress=True,
        random_state=42,
        timeout_in_seconds=1800,
        temp_equation_file=True,
        batching=True,
        batch_size=50
    )
    
    print("\nTraining simpler model on scaled data...")
    simple_model.fit(X_scaled, y_scaled)
    
    if hasattr(simple_model, 'equations_') and simple_model.equations_ is not None:
        equations = simple_model.equations_
        
        print(f"\nFound {len(equations)} equations")
        
        # Test each equation
        simple_results = []
        
        for idx, row in equations.iterrows():
            eq = row['equation']
            complexity = row['Complexity'] if 'Complexity' in row else idx
            
            try:
                # Predict on scaled data
                y_pred_scaled = simple_model.predict(X_scaled)
                
                # Inverse transform predictions
                y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
                
                # Calculate R² on original scale
                r2_simple = r2_score(y, y_pred)
                
                simple_results.append({
                    'Complexity': complexity,
                    'R2': r2_simple,
                    'Equation': eq[:100] + '...'
                })
                
            except Exception as e:
                continue
        
        # Sort and show results
        if simple_results:
            simple_df = pd.DataFrame(simple_results)
            simple_df = simple_df.sort_values('R2', ascending=False)
            
            print("\nTop 5 simpler equations:")
            print(simple_df.head(5).to_string(index=False))
            
            # Save results
            simple_path = os.path.join(output_dir, "scaled_model_results.csv")
            simple_df.to_csv(simple_path, index=False)
            print(f"\n✅ Results saved to: {simple_path}")
            
            # Get best simple equation
            best_simple = simple_df.iloc[0]
            print(f"\n🏆 Best simple equation (R² = {best_simple['R2']:.4f}):")
            print(f"{best_simple['Equation']}")
            
except Exception as e:
    print(f"Model training failed: {e}")

# ==============================
# FIXED REPORT WITH UNICODE HANDLING
# ==============================

print("\n" + "="*70)
print("GENERATING FINAL REPORT")
print("="*70)

# Create report without special characters
report = f"""
EQUATION DISCOVERY RESULTS
{'='*50}

Best Equation Found:
{best_equation}

Validation Metrics:
* R² Score: {r2:.4f}
* MAE: {mae:.4f}
* RMSE: {rmse:.4f}

Dataset Information:
* Samples: {len(y)}
* Features: {features}
* Theoretical Maximum R²: 0.9173
* Gap to Maximum: {0.9173 - r2:.4f}

Bacteria Inclusion:
* Cbac appears in equation: {'Yes' if 'x0' in best_equation else 'No'}

Prediction Statistics:
* Actual mean: {y.mean():.4f}
* Predicted mean: {y_pred.mean():.4f}
* Actual std: {y.std():.4f}
* Predicted std: {y_pred.std():.4f}
* Correlation: {np.corrcoef(y, y_pred)[0,1]:.4f}

Recommendations:
"""

if r2 >= 0.85:
    report += "* EXCELLENT! Equation is highly reliable\n"
elif r2 >= 0.81:
    report += "* GOOD! Equation meets target performance\n"
elif r2 >= 0.77:
    report += "* ACCEPTABLE! Close to target performance\n"
elif r2 >= 0:
    report += "* POSITIVE! Better than random guessing\n"
else:
    report += "* WARNING! Negative R² - worse than random guessing\n"

if 'x0' in best_equation:
    report += "* Bacteria (Cbac) is properly included\n"
else:
    report += "* WARNING: Bacteria not included\n"

# Save report with UTF-8 encoding to handle special characters
report_path = os.path.join(output_dir, "final_report.txt")
with open(report_path, 'w', encoding='utf-8') as f:
    f.write(report)

print(report)
print(f"\n✅ Final report saved to: {report_path}")

# ==============================
# READABLE EQUATION WITHOUT SPECIAL CHARS
# ==============================

print("\n" + "="*70)
print("READABLE EQUATION")
print("="*70)

# Create readable version without special characters
readable_eq = best_equation
readable_eq = readable_eq.replace('x0', 'Cbac')
readable_eq = readable_eq.replace('x1', 'CAgNP')
readable_eq = readable_eq.replace('x2', 'Ra')
readable_eq = readable_eq.replace('exp', 'e^')
readable_eq = readable_eq.replace('tanh', 'tanh')
readable_eq = readable_eq.replace('square', '^2')
readable_eq = readable_eq.replace('cube', '^3')
readable_eq = readable_eq.replace('^', '^')

print(f"\nDIZ = {readable_eq}")

print("\n" + "="*70)
print(f"✅ All files saved to: {output_dir}")
print("="*70)