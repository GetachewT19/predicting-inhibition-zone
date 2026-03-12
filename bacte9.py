# ==============================
# View Final PySR Results
# ==============================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Path to results
output_dir = r"C:\Users\Amare\Desktop\pysr_final_working"

print("="*70)
print("FINAL PySR RESULTS")
print("="*70)

# ------------------------------
# 1. Check all equations
# ------------------------------
csv_path = os.path.join(output_dir, "all_equations.csv")
if os.path.exists(csv_path):
    print(f"\n📊 Loading equations from: {csv_path}")
    df_eq = pd.read_csv(csv_path)
    print(f"Total equations found: {len(df_eq)}")
    
    # Sort by loss
    df_sorted = df_eq.sort_values('loss')
    
    # Display top 20 equations
    print("\n🏆 TOP 20 EQUATIONS:")
    print("-"*80)
    for i in range(min(20, len(df_sorted))):
        row = df_sorted.iloc[i]
        contains_bacteria = 'x0' in str(row['equation'])
        marker = '✓' if contains_bacteria else '✗'
        
        print(f"\n{i+1}. Loss: {row['loss']:.6f} | Complexity: {row['complexity']} | Bacteria: {marker}")
        print(f"   Equation: {row['equation']}")
    
    # Best equation
    best_row = df_sorted.iloc[0]
    print("\n" + "⭐"*80)
    print("🌟 BEST EQUATION OVERALL")
    print("⭐"*80)
    print(f"Loss: {best_row['loss']:.6f}")
    print(f"Complexity: {best_row['complexity']}")
    print(f"Contains bacteria (x0): {'✓ YES' if 'x0' in str(best_row['equation']) else '✗ NO'}")
    print(f"\nEquation: {best_row['equation']}")
    
    # Check for bacteria equations
    bacteria_eqs = df_eq[df_eq['equation'].str.contains('x0', na=False)]
    print(f"\n🧫 Equations containing bacteria: {len(bacteria_eqs)}/{len(df_eq)}")
    
    if len(bacteria_eqs) > 0:
        print("\nBest bacteria-inclusive equations:")
        bact_sorted = bacteria_eqs.sort_values('loss').head(5)
        for i, (idx, row) in enumerate(bact_sorted.iterrows()):
            print(f"\n  {i+1}. Loss: {row['loss']:.6f} | Complexity: {row['complexity']}")
            print(f"     Equation: {row['equation']}")
    
else:
    print(f"\n❌ Equations file not found: {csv_path}")

# ------------------------------
# 2. Check best equation text file
# ------------------------------
txt_path = os.path.join(output_dir, "best_equation.txt")
if os.path.exists(txt_path):
    print("\n" + "="*70)
    print("📄 BEST EQUATION DETAILS")
    print("="*70)
    with open(txt_path, 'r') as f:
        print(f.read())
else:
    print(f"\n❌ Best equation file not found: {txt_path}")

# ------------------------------
# 3. Check for bacteria-specific file
# ------------------------------
bact_path = os.path.join(output_dir, "bacteria_equations.csv")
if os.path.exists(bact_path):
    print("\n" + "="*70)
    print("🧫 BACTERIA EQUATIONS FILE FOUND")
    print("="*70)
    df_bact = pd.read_csv(bact_path)
    print(f"Contains {len(df_bact)} equations with bacteria")
    
    # Show top 3
    print("\nTop 3 bacteria equations:")
    bact_sorted = df_bact.sort_values('loss').head(3)
    for i, (idx, row) in enumerate(bact_sorted.iterrows()):
        print(f"\n  {i+1}. Loss: {row['loss']:.6f}")
        print(f"     Equation: {row['equation']}")

# ------------------------------
# 4. Check for plot
# ------------------------------
plot_path = os.path.join(output_dir, "results.png")
if os.path.exists(plot_path):
    print("\n" + "="*70)
    print("📊 RESULTS PLOT")
    print("="*70)
    from PIL import Image
    img = Image.open(plot_path)
    plt.figure(figsize=(12, 5))
    plt.imshow(img)
    plt.axis('off')
    plt.title('PySR Results', fontsize=14, fontweight='bold')
    plt.show()
else:
    print(f"\n❌ Plot not found: {plot_path}")

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"\nAll results are saved in: {output_dir}")
print("\nKey files:")
print(f"  - all_equations.csv: All {len(df_eq) if 'df_eq' in locals() else '?'} discovered equations")
print(f"  - best_equation.txt: Best equation with metrics")
print(f"  - results.png: Visualization plot")
if os.path.exists(bact_path):
    print(f"  - bacteria_equations.csv: {len(df_bact)} equations containing bacteria")
print(f"  - pysr_model.pkl: Trained model (can be loaded later)")
print(f"  - scaler.pkl: Feature scaler for new predictions")

print("\n" + "="*70)