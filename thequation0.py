# ==============================
# Analyze Diagnostic Results
# ==============================

import pandas as pd
import os
import glob

output_dir = r"C:\Users\Amare\Desktop\pysr_diagnostic_fixed"

print("="*70)
print("ANALYZING DIAGNOSTIC RESULTS")
print("="*70)

# Check all CSV files in the output directory
csv_files = glob.glob(os.path.join(output_dir, "*.csv"))

if not csv_files:
    print("\n❌ No CSV files found in the output directory!")
else:
    print(f"\n📊 Found {len(csv_files)} result files:")
    
    for csv_file in csv_files:
        filename = os.path.basename(csv_file)
        print(f"\n{'-'*50}")
        print(f"File: {filename}")
        print(f"{'-'*50}")
        
        try:
            df = pd.read_csv(csv_file)
            print(f"  Shape: {df.shape}")
            
            if 'loss' in df.columns and 'equation' in df.columns:
                # Show top 3 equations
                top3 = df.sort_values('loss').head(3)
                for i, (idx, row) in enumerate(top3.iterrows()):
                    print(f"\n  Rank {i+1}:")
                    print(f"    Complexity: {row.get('complexity', 'N/A')}")
                    print(f"    Loss: {row['loss']:.6f}")
                    if 'score' in row:
                        print(f"    Score: {row['score']:.4f}")
                    print(f"    Equation: {row['equation'][:100]}...")
            else:
                print("  No equation data found")
                
        except Exception as e:
            print(f"  Error reading file: {e}")

print("\n" + "="*70)