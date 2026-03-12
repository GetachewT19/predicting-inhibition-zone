# ==============================
# Continue Analysis - Results from Clean Data
# ==============================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Paths
output_dir = r"C:\Users\Amare\Desktop\pysr_cleaned_data"

print("="*70)
print("ANALYZING PySR RESULTS ON CLEAN DATA")
print("="*70)

# ------------------------------
# Check for results files
# ------------------------------
csv_path = os.path.join(output_dir, "clean_data_equations.csv")
txt_path = os.path.join(output_dir, "clean_data_best_equation.txt")
plot_path = os.path.join(output_dir, "clean_data_results.png")

if os.path.exists(csv_path):
    print(f"\n✅ Found results file: {csv_path}")
    
    # Load equations
    df_eq = pd.read_csv(csv_path)
    print(f"\n📊 Total equations found: {len(df_eq)}")
    
    # Sort by loss
    df_sorted = df_eq.sort_values('loss')
    
    # Show top 10 equations
    print("\n🏆 TOP 10 EQUATIONS (by loss):")
    for i in range(min(10, len(df_sorted))):
        row = df_sorted.iloc[i]
        print(f"\n  {i+1}. Loss: {row['loss']:.6f}")
        print(f"     Complexity: {row['complexity']}")
        print(f"     Equation: {row['equation']}")
    
    # Filter equations with bacteria
    bacteria_eqs = df_eq[df_eq['equation'].str.contains('x0', na=False)]
    print(f"\n🧫 Equations containing bacteria (x0): {len(bacteria_eqs)}/{len(df_eq)}")
    
    if len(bacteria_eqs) > 0:
        print("\nTop bacteria-inclusive equations:")
        bacteria_sorted = bacteria_eqs.sort_values('loss').head(5)
        for i, (idx, row) in enumerate(bacteria_sorted.iterrows()):
            print(f"\n  {i+1}. Loss: {row['loss']:.6f}")
            print(f"     Equation: {row['equation']}")
    
    # Best equation
    best_row = df_sorted.iloc[0]
    print("\n" + "⭐"*70)
    print("🌟 BEST EQUATION ON CLEAN DATA")
    print("⭐"*70)
    print(f"Loss: {best_row['loss']:.6f}")
    print(f"Complexity: {best_row['complexity']}")
    print(f"Equation: {best_row['equation']}")
    print(f"Contains bacteria: {'x0' in best_row['equation']}")
    
    # Check if best equation uses bacteria
    if 'x0' not in best_row['equation'] and len(bacteria_eqs) > 0:
        print("\n⚠️  Best equation doesn't use bacteria!")
        print("   Checking best bacteria-inclusive equation...")
        
        best_bacteria = bacteria_sorted.iloc[0]
        print(f"\nBest bacteria equation:")
        print(f"  Loss: {best_bacteria['loss']:.6f}")
        print(f"  Equation: {best_bacteria['equation']}")
        
        # Compare losses
        loss_diff = ((best_bacteria['loss'] - best_row['loss']) / best_row['loss']) * 100
        print(f"\n  Loss difference: {loss_diff:+.1f}%")
        
else:
    print(f"\n❌ Results file not found: {csv_path}")
    print("PySR may still be running or encountered an error.")

# ------------------------------
# Check for best equation text file
# ------------------------------
if os.path.exists(txt_path):
    print(f"\n✅ Best equation details saved to: {txt_path}")
    with open(txt_path, 'r') as f:
        content = f.read()
    print("\n" + content)
else:
    print(f"\n❌ Best equation text file not found")

# ------------------------------
# Check for plot
# ------------------------------
if os.path.exists(plot_path):
    print(f"\n✅ Plot saved to: {plot_path}")
    from PIL import Image
    img = Image.open(plot_path)
    plt.figure(figsize=(12, 8))
    plt.imshow(img)
    plt.axis('off')
    plt.title('PySR Results on Clean Data', fontsize=14, fontweight='bold')
    plt.show()
else:
    print(f"\n❌ Plot file not found")

print("\n" + "="*70)
print("ANALYSIS COMPLETE")
print("="*70)