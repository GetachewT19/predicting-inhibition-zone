# ==============================
# FIND PySR TEMPORARY FILES
# ==============================

import os
import glob
import pandas as pd
from datetime import datetime, timedelta

print("="*70)
print("FINDING PySR TEMPORARY FILES")
print("="*70)

# Check common temporary directories
temp_dirs = [
    r"C:\Users\Amare\AppData\Local\Temp",
    os.getcwd(),
    r"C:\Users\Amare\Desktop"
]

found_files = []

for temp_dir in temp_dirs:
    print(f"\n📁 Searching in: {temp_dir}")
    
    # Look for hall_of_fame.csv files
    hof_files = glob.glob(os.path.join(temp_dir, "**/hall_of_fame.csv"), recursive=True)
    hof_files += glob.glob(os.path.join(temp_dir, "hall_of_fame.csv"))
    
    # Look for any CSV files created in the last hour
    recent_csv = glob.glob(os.path.join(temp_dir, "*.csv"))
    recent_csv = [f for f in recent_csv if os.path.getmtime(f) > (datetime.now() - timedelta(hours=1)).timestamp()]
    
    # Look for PySR temp directories
    pysr_temp = glob.glob(os.path.join(temp_dir, "tmp*"))
    pysr_temp = [d for d in pysr_temp if os.path.isdir(d)]
    
    if hof_files:
        print(f"  ✅ Found hall_of_fame.csv:")
        for f in hof_files:
            size = os.path.getsize(f)
            mtime = datetime.fromtimestamp(os.path.getmtime(f))
            print(f"     - {f} ({size:,} bytes, modified: {mtime})")
            found_files.append(f)
    
    if recent_csv:
        print(f"  ✅ Found recent CSV files:")
        for f in recent_csv[:5]:  # Show first 5
            size = os.path.getsize(f)
            mtime = datetime.fromtimestamp(os.path.getmtime(f))
            print(f"     - {os.path.basename(f)} ({size:,} bytes, modified: {mtime})")
            found_files.append(f)
    
    if pysr_temp:
        print(f"  ✅ Found PySR temp directories:")
        for d in pysr_temp:
            print(f"     - {d}")
            # Look inside these directories
            inner_files = glob.glob(os.path.join(d, "*"))
            for inner in inner_files:
                if inner.endswith('.csv') or inner.endswith('.pkl'):
                    size = os.path.getsize(inner)
                    print(f"       └─ {os.path.basename(inner)} ({size:,} bytes)")
                    found_files.append(inner)

# If we found files, let's examine them
if found_files:
    print("\n" + "="*70)
    print("EXAMINING FOUND FILES")
    print("="*70)
    
    for file_path in found_files[:3]:  # Check first 3
        print(f"\n📄 File: {os.path.basename(file_path)}")
        print(f"   Path: {file_path}")
        
        if file_path.endswith('.csv'):
            try:
                df = pd.read_csv(file_path)
                print(f"   Rows: {len(df)}")
                print(f"   Columns: {df.columns.tolist()}")
                
                if 'equation' in df.columns or 'Equation' in df.columns:
                    eq_col = 'equation' if 'equation' in df.columns else 'Equation'
                    loss_col = 'loss' if 'loss' in df.columns else 'Loss' if 'Loss' in df.columns else None
                    
                    if loss_col:
                        df_sorted = df.sort_values(loss_col)
                        print(f"\n   🏆 Top equation:")
                        best = df_sorted.iloc[0]
                        print(f"      {best[eq_col][:150]}...")
                        
                        # Save this file to Desktop
                        from shutil import copy2
                        desktop = r"C:\Users\Amare\Desktop"
                        dest = os.path.join(desktop, f"found_equations_{datetime.now().strftime('%H%M%S')}.csv")
                        copy2(file_path, dest)
                        print(f"\n   ✅ Copied to: {dest}")
            except Exception as e:
                print(f"   Error reading: {e}")

print("\n" + "="*70)
print("RECOVERY ATTEMPT COMPLETE")
print("="*70)
print("""
If files were found, they've been copied to your Desktop.
Check for files named 'found_equations_*.csv'

If no files were found, PySR may have encountered an error during fitting.
Try running with a simpler configuration:

model = PySRRegressor(
    niterations=100,
    populations=3,
    population_size=50,
    maxsize=10,
    binary_operators=["+", "*"],
    unary_operators=["square"],
    procs=1
)
""")