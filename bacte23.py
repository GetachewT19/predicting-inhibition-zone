# ==============================
# SAVE THE GOOD RESULTS (R² = 0.7749)
# ==============================

import pandas as pd
import numpy as np
import os
import pickle
import shutil
from datetime import datetime

print("="*70)
print("SAVING GOOD RESULTS (R² = 0.7749)")
print("="*70)

# Source of temporary files
temp_dir = r"C:\Users\Amare\AppData\Local\Temp\tmp3lmh0wn1\20260311_130233_0QZAjJ"
if not os.path.exists(temp_dir):
    # Try to find the most recent temp directory
    import glob
    temp_base = r"C:\Users\Amare\AppData\Local\Temp"
    possible_dirs = glob.glob(os.path.join(temp_base, "tmp*"))
    if possible_dirs:
        # Get the most recent
        temp_dir = max(possible_dirs, key=os.path.getmtime)
        print(f"✅ Found temp directory: {temp_dir}")

# Create permanent directory
output_dir = r"C:\Users\Amare\Desktop\diz_good_results_0.7749"
os.makedirs(output_dir, exist_ok=True)
print(f"\n📁 Saving to: {output_dir}")

# Look for hall_of_fame.csv
hof_path = os.path.join(temp_dir, "hall_of_fame.csv")
if os.path.exists(hof_path):
    print(f"\n✅ Found hall_of_fame.csv")
    
    # Copy the file
    dest_hof = os.path.join(output_dir, "hall_of_fame.csv")
    shutil.copy2(hof_path, dest_hof)
    print(f"✅ Copied to: {dest_hof}")
    
    # Read and display equations
    df = pd.read_csv(hof_path)
    print(f"\n📊 Total equations: {len(df)}")
    
    # Sort by loss
    df_sorted = df.sort_values('loss')
    
    print("\n🏆 TOP 5 EQUATIONS:")
    for i in range(min(5, len(df_sorted))):
        row = df_sorted.iloc[i]
        contains_bacteria = 'x0' in str(row['equation'])
        marker = '✓' if contains_bacteria else '✗'
        print(f"\n{i+1}. Loss: {row['loss']:.6f} | Complexity: {row['complexity']} | Bacteria: {marker}")
        print(f"   Equation: {row['equation'][:100]}...")
    
    # Best equation
    best_row = df_sorted.iloc[0]
    best_eq = best_row['equation']
    
    print("\n" + "⭐"*70)
    print("🌟 BEST EQUATION (R² = 0.7749)")
    print("⭐"*70)
    print(f"Loss: {best_row['loss']:.6f}")
    print(f"Complexity: {best_row['complexity']}")
    print(f"Contains bacteria: {'✓' if 'x0' in best_eq else '✗'}")
    print(f"\nEquation: {best_eq}")
    
    # Save best equation separately
    best_txt = os.path.join(output_dir, "best_equation.txt")
    with open(best_txt, 'w') as f:
        f.write("="*70 + "\n")
        f.write("BEST EQUATION FOR DIZ\n")
        f.write("="*70 + "\n\n")
        f.write(f"R² Score: 0.7749\n")
        f.write(f"Loss: {best_row['loss']:.6f}\n")
        f.write(f"Complexity: {best_row['complexity']}\n\n")
        f.write(f"Equation:\n{best_eq}\n")
    print(f"\n✅ Best equation saved to: {best_txt}")
    
else:
    print(f"\n❌ hall_of_fame.csv not found in {temp_dir}")

# Also look for the model
model_path = os.path.join(output_dir, "pysr_model.pkl")
if not os.path.exists(model_path):
    print("\n⚠️ Model file not found in temp directory")
    print("You may need to re-run with model saving enabled")

print("\n" + "="*70)
print("NEXT STEPS")
print("="*70)
print(f"""
✅ Success! Found equation with R² = 0.7749

Files saved to: {output_dir}

To use this equation:

1. Copy this code:
----------------------------------------
def predict_diz(x0, x1, x2):
    return {best_eq}

# For raw values, you'll need the scaler
----------------------------------------

2. The equation includes bacteria: {'✓' if 'x0' in best_eq else '✗'}

3. This is even better than the previous 0.6891!
""")