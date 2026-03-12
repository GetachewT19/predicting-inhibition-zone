# ==============================
# Emergency Recovery - Check What Was Actually Saved
# ==============================

import os
import glob
import pickle

# Path to results
output_dir = r"C:\Users\Amare\Desktop\pysr_final_working"

print("="*70)
print("EMERGENCY RECOVERY - CHECKING DIRECTORY")
print("="*70)

# List ALL files in the directory
print(f"\n📁 Directory contents: {output_dir}")
print("-"*70)

all_files = glob.glob(os.path.join(output_dir, "*"))
if all_files:
    for file in all_files:
        size = os.path.getsize(file)
        modified = os.path.getmtime(file)
        print(f"  {os.path.basename(file)} ({size:,} bytes) - Last modified: {modified}")
else:
    print("  Directory is EMPTY!")

# Check for pickle files (model and scaler)
print("\n" + "="*70)
print("CHECKING PICKLE FILES")
print("="*70)

model_path = os.path.join(output_dir, "pysr_model.pkl")
scaler_path = os.path.join(output_dir, "scaler.pkl")

if os.path.exists(model_path):
    print(f"\n✅ Model file found: {model_path}")
    model_size = os.path.getsize(model_path)
    print(f"   Size: {model_size:,} bytes")
    
    # Try to load the model
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print(f"   ✅ Model loaded successfully!")
        
        # Check if model has equations
        if hasattr(model, 'equations_') and model.equations_ is not None:
            equations = model.equations_
            print(f"   📊 Model contains {len(equations)} equations in memory")
            
            # Save them now
            csv_path = os.path.join(output_dir, "recovered_equations.csv")
            equations.to_csv(csv_path, index=False)
            print(f"   ✅ Equations recovered and saved to: {csv_path}")
            
            # Display top equations
            print("\n🏆 TOP 10 EQUATIONS (recovered from model):")
            top10 = equations.sort_values('loss').head(10)
            for i, (idx, row) in enumerate(top10.iterrows()):
                contains_bacteria = 'x0' in str(row['equation'])
                marker = '✓' if contains_bacteria else '✗'
                print(f"\n  {i+1}. Loss: {row['loss']:.6f} | Complexity: {row['complexity']} | Bacteria: {marker}")
                print(f"     Equation: {row['equation'][:100]}...")
            
            # Best equation
            best_idx = equations['loss'].idxmin()
            best_eq = equations.loc[best_idx, 'equation']
            best_loss = equations.loc[best_idx, 'loss']
            best_complexity = equations.loc[best_idx, 'complexity']
            
            print("\n" + "⭐"*70)
            print("🌟 BEST EQUATION (recovered)")
            print("⭐"*70)
            print(f"Loss: {best_loss:.6f}")
            print(f"Complexity: {best_complexity}")
            print(f"Contains bacteria: {'✓' if 'x0' in best_eq else '✗'}")
            print(f"\nEquation: {best_eq}")
            
            # Check for bacteria equations
            bacteria_eqs = equations[equations['equation'].str.contains('x0', na=False)]
            if len(bacteria_eqs) > 0:
                print(f"\n🧫 Found {len(bacteria_eqs)} equations with bacteria")
                bact_path = os.path.join(output_dir, "recovered_bacteria_equations.csv")
                bacteria_eqs.to_csv(bact_path, index=False)
                print(f"   ✅ Bacteria equations saved to: {bact_path}")
                
                # Show best bacteria equation
                best_bact = bacteria_eqs.sort_values('loss').iloc[0]
                print(f"\nBest bacteria equation:")
                print(f"  Loss: {best_bact['loss']:.6f}")
                print(f"  Equation: {best_bact['equation']}")
            
        else:
            print("   ❌ Model has no equations attribute")
            
    except Exception as e:
        print(f"   ❌ Failed to load model: {e}")
else:
    print(f"\n❌ Model file not found: {model_path}")

if os.path.exists(scaler_path):
    print(f"\n✅ Scaler file found: {scaler_path}")
    scaler_size = os.path.getsize(scaler_path)
    print(f"   Size: {scaler_size:,} bytes")
else:
    print(f"\n❌ Scaler file not found: {scaler_path}")

# Check for any CSV files in the directory
print("\n" + "="*70)
print("CHECKING FOR ANY CSV FILES")
print("="*70)

csv_files = glob.glob(os.path.join(output_dir, "*.csv"))
if csv_files:
    print(f"\nFound {len(csv_files)} CSV files:")
    for csv_file in csv_files:
        size = os.path.getsize(csv_file)
        print(f"  {os.path.basename(csv_file)} ({size:,} bytes)")
        
        # Try to read it
        try:
            df = pd.read_csv(csv_file)
            print(f"    Contains {len(df)} rows, columns: {df.columns.tolist()}")
        except:
            print(f"    Could not read file")
else:
    print("\nNo CSV files found in directory")

print("\n" + "="*70)
print("RECOVERY ATTEMPT COMPLETE")
print("="*70)
print(f"\nCheck {output_dir} for recovered files:")
print("  - recovered_equations.csv (if model was saved)")
print("  - recovered_bacteria_equations.csv (if bacteria equations exist)")