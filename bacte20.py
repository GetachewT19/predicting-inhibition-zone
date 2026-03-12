# ==============================
# FORCED SAVING - DIZ PREDICTION
# ==============================

import pandas as pd
import numpy as np
from pysr import PySRRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score
import os
import pickle
import datetime

# Create output directory with timestamp to avoid conflicts
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = rf"C:\Users\Amare\Desktop\diz_results_{timestamp}"
os.makedirs(output_dir, exist_ok=True)

print("="*70)
print("FORCED SAVING - DIZ PREDICTION")
print("="*70)
print(f"Results will be saved to: {output_dir}")

# Load data
data_path = r"C:\Users\Amare\Desktop\getachewt\datato232-81.csv"
df = pd.read_csv(data_path)
target = 'DIZ'
features = ['Cbac', 'CAgNP', 'Ra']

print(f"\n📊 Dataset shape: {df.shape}")

# Clean data
df_clean = df.copy()
for col in [target] + features:
    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
df_clean = df_clean.dropna()

X = df_clean[features].values.astype(float)
y = df_clean[target].values.astype(float)

print(f"Final samples: {len(y)}")

# Scale features
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)

# Save scaler immediately
scaler_path = os.path.join(output_dir, "scaler.pkl")
with open(scaler_path, 'wb') as f:
    pickle.dump(scaler, f)
print(f"✅ Scaler saved to: {scaler_path}")

# Simple PySR model with guaranteed equation saving
print("\n🚀 Running PySR (this will take a few minutes)...")

# Use a very simple configuration first
model = PySRRegressor(
    niterations=200,  # Fewer iterations for speed
    populations=5,
    population_size=50,
    maxsize=15,
    binary_operators=["+", "-", "*", "/"],
    unary_operators=["square", "sqrt"],
    procs=2,
    loss="L2DistLoss()",
    progress=True,
    random_state=42,
    temp_equation_file=True,  # This creates temporary files
    update_verbosity=2  # More verbose output
)

try:
    # Fit model
    model.fit(X_scaled, y)
    print("\n✅ PySR fitting completed!")
    
    # Save model immediately
    model_path = os.path.join(output_dir, "pysr_model.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"✅ Model saved to: {model_path}")
    
    # Check and save equations
    if hasattr(model, 'equations_') and model.equations_ is not None:
        equations = model.equations_
        
        # Save to CSV
        csv_path = os.path.join(output_dir, "all_equations.csv")
        equations.to_csv(csv_path, index=False)
        print(f"✅ Equations saved to: {csv_path}")
        
        # Display results
        print("\n🏆 TOP 5 EQUATIONS:")
        top5 = equations.sort_values('loss').head(5)
        for i, (idx, row) in enumerate(top5.iterrows()):
            contains_bacteria = 'x0' in str(row['equation'])
            marker = '✓' if contains_bacteria else '✗'
            print(f"\n{i+1}. Loss: {row['loss']:.6f} | Complexity: {row['complexity']} | Bacteria: {marker}")
            print(f"   Equation: {row['equation']}")
        
        # Best equation
        best_idx = equations['loss'].idxmin()
        best_eq = equations.loc[best_idx, 'equation']
        best_loss = equations.loc[best_idx, 'loss']
        
        # Calculate R²
        y_pred = model.predict(X_scaled)
        r2 = r2_score(y, y_pred)
        
        print("\n" + "⭐"*70)
        print("🌟 BEST EQUATION")
        print("⭐"*70)
        print(f"R² = {r2:.4f}")
        print(f"Loss = {best_loss:.6f}")
        print(f"Equation: {best_eq}")
        
        # Save best equation
        txt_path = os.path.join(output_dir, "best_equation.txt")
        with open(txt_path, 'w') as f:
            f.write(f"R² Score: {r2:.4f}\n")
            f.write(f"Loss: {best_loss:.6f}\n")
            f.write(f"Complexity: {equations.loc[best_idx, 'complexity']}\n\n")
            f.write(f"Equation:\n{best_eq}\n")
        print(f"✅ Best equation saved to: {txt_path}")
        
        # Check bacteria
        if 'x0' in best_eq:
            print("\n✅ Bacteria (Cbac) is included!")
        else:
            print("\n⚠️ Bacteria not in best equation")
            
    else:
        print("❌ No equations found in model")
        
except Exception as e:
    print(f"\n❌ Error: {e}")
    
    # Try even simpler configuration
    print("\n🔄 Trying even simpler configuration...")
    
    model2 = PySRRegressor(
        niterations=100,
        populations=3,
        population_size=30,
        maxsize=10,
        binary_operators=["+", "*"],
        unary_operators=["square"],
        procs=1,
        random_state=42
    )
    
    model2.fit(X_scaled, y)
    
    if hasattr(model2, 'equations_') and model2.equations_ is not None:
        equations = model2.equations_
        csv_path = os.path.join(output_dir, "all_equations_simple.csv")
        equations.to_csv(csv_path, index=False)
        print(f"✅ Simple equations saved to: {csv_path}")
        
        print("\nSimple model results:")
        print(equations.head())

print("\n" + "="*70)
print(f"✅ ALL FILES SAVED TO: {output_dir}")
print("="*70)

# Final verification
print("\n📁 Final directory contents:")
for file in os.listdir(output_dir):
    size = os.path.getsize(os.path.join(output_dir, file))
    print(f"  - {file} ({size:,} bytes)")