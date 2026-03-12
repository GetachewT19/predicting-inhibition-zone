# ==============================
# Debug: Check Current Directory
# ==============================

import os
import sys

print("="*70)
print("DEBUG INFORMATION")
print("="*70)

# Current working directory
cwd = os.getcwd()
print(f"\nCurrent working directory: {cwd}")

# Check if diz_predictor.py exists in current directory
if os.path.exists(os.path.join(cwd, "diz_predictor.py")):
    print("✅ diz_predictor.py found in current directory")
else:
    print("❌ diz_predictor.py NOT found in current directory")

# Check the model directory
model_dir = r"C:\Users\Amare\Desktop\pysr_final_manual_save"
print(f"\nModel directory: {model_dir}")
if os.path.exists(model_dir):
    print("✅ Model directory exists")
    
    # List files in model directory
    print("\nFiles in model directory:")
    for file in os.listdir(model_dir):
        size = os.path.getsize(os.path.join(model_dir, file))
        print(f"  - {file} ({size} bytes)")
else:
    print("❌ Model directory NOT found")

print("\n" + "="*70)
print("SOLUTION")
print("="*70)
print("""
To fix the import error, either:

1. Copy diz_predictor.py to your current directory:
   C:\\Users\\Amare\\source\\repos\\bacterialinhibition\\

2. Or use the standalone script above (Option 2)

3. Or add the model directory to Python path:
   import sys
   sys.path.append(r"C:\\Users\\Amare\\Desktop\\pysr_final_manual_save")
   from diz_predictor import DIZPredictor
""")