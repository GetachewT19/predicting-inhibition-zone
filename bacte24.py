# ==============================
# SAVE GOOD RESULTS - FIXED VERSION
# ==============================

import pandas as pd
import numpy as np
import os
import shutil
from datetime import datetime

print("="*70)
print("SAVING GOOD RESULTS - FIXED VERSION")
print("="*70)

# Source of temporary files
temp_dir = r"C:\Users\Amare\AppData\Local\Temp\tmp3lmh0wn1\20260311_130233_0QZAjJ"
output_dir = r"C:\Users\Amare\Desktop\diz_good_results_0.7749"
os.makedirs(output_dir, exist_ok=True)

print(f"\n📁 Source: {temp_dir}")
print(f"📁 Destination: {output_dir}")

# Look for hall_of_fame.csv
hof_path = os.path.join(temp_dir, "hall_of_fame.csv")
if os.path.exists(hof_path):
    print(f"\n✅ Found hall_of_fame.csv")
    
    # Copy the file
    dest_hof = os.path.join(output_dir, "hall_of_fame.csv")
    shutil.copy2(hof_path, dest_hof)
    print(f"✅ Copied to: {dest_hof}")
    
    # Read and display column names
    df = pd.read_csv(hof_path)
    print(f"\n📊 File contains {len(df)} rows")
    print(f"📋 Column names: {df.columns.tolist()}")
    
    # Try to identify the relevant columns
    possible_loss_cols = ['loss', 'Loss', 'score', 'Score', 'mse', 'MSE']
    possible_eq_cols = ['equation', 'Equation', 'expr', 'Expr', 'expression']
    possible_complexity_cols = ['complexity', 'Complexity', 'size', 'Size']
    
    # Find loss column
    loss_col = None
    for col in possible_loss_cols:
        if col in df.columns:
            loss_col = col
            break
    
    # Find equation column
    eq_col = None
    for col in possible_eq_cols:
        if col in df.columns:
            eq_col = col
            break
    
    # Find complexity column
    comp_col = None
    for col in possible_complexity_cols:
        if col in df.columns:
            comp_col = col
            break
    
    print(f"\n📊 Identified columns:")
    print(f"  Loss column: {loss_col}")
    print(f"  Equation column: {eq_col}")
    print(f"  Complexity column: {comp_col}")
    
    if loss_col and eq_col:
        # Sort by loss
        df_sorted = df.sort_values(loss_col)
        
        print("\n🏆 TOP 5 EQUATIONS:")
        for i in range(min(5, len(df_sorted))):
            row = df_sorted.iloc[i]
            eq_str = str(row[eq_col]) if eq_col else "N/A"
            loss_val = row[loss_col] if loss_col else 0
            comp_val = row[comp_col] if comp_col else "?"
            
            contains_bacteria = 'x0' in eq_str
            marker = '✓' if contains_bacteria else '✗'
            
            print(f"\n{i+1}. Loss: {loss_val:.6f} | Complexity: {comp_val} | Bacteria: {marker}")
            print(f"   Equation: {eq_str[:100]}...")
        
        # Best equation
        best_row = df_sorted.iloc[0]
        best_eq = str(best_row[eq_col])
        best_loss = best_row[loss_col]
        best_comp = best_row[comp_col] if comp_col else "?"
        
        print("\n" + "⭐"*70)
        print("🌟 BEST EQUATION")
        print("⭐"*70)
        print(f"Loss: {best_loss:.6f}")
        print(f"Complexity: {best_comp}")
        print(f"Contains bacteria: {'✓' if 'x0' in best_eq else '✗'}")
        print(f"\nEquation: {best_eq}")
        
        # Save best equation
        best_txt = os.path.join(output_dir, "best_equation.txt")
        with open(best_txt, 'w') as f:
            f.write("="*70 + "\n")
            f.write("BEST EQUATION FOR DIZ\n")
            f.write("="*70 + "\n\n")
            f.write(f"R² Score: 0.7749 (from diagnostic)\n")
            f.write(f"Loss: {best_loss:.6f}\n")
            f.write(f"Complexity: {best_comp}\n\n")
            f.write(f"Equation:\n{best_eq}\n")
        print(f"\n✅ Best equation saved to: {best_txt}")
        
        # Save all equations with proper names
        all_eq_path = os.path.join(output_dir, "all_equations.csv")
        df.to_csv(all_eq_path, index=False)
        print(f"✅ All equations saved to: {all_eq_path}")
        
    else:
        print("\n❌ Could not identify required columns")
        print("Full dataframe contents:")
        print(df.head())
        
        # Save raw file anyway
        raw_path = os.path.join(output_dir, "raw_results.csv")
        df.to_csv(raw_path, index=False)
        print(f"✅ Raw results saved to: {raw_path}")
else:
    print(f"\n❌ hall_of_fame.csv not found in {temp_dir}")
    
    # Look for any CSV files in temp directory
    import glob
    csv_files = glob.glob(os.path.join(temp_dir, "*.csv"))
    if csv_files:
        print(f"\nFound {len(csv_files)} CSV files:")
        for csv_file in csv_files:
            print(f"  - {os.path.basename(csv_file)}")
            # Copy each file
            dest_file = os.path.join(output_dir, os.path.basename(csv_file))
            shutil.copy2(csv_file, dest_file)
            print(f"    ✅ Copied to: {dest_file}")

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"\n✅ Files saved to: {output_dir}")
print("\nCheck the directory for:")
print("  - hall_of_fame.csv (all equations)")
print("  - best_equation.txt (best equation)")
print("  - all_equations.csv (renamed copy)")

print("\n" + "="*70)