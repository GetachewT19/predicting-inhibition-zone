# ==============================
# Minimal PySR Test
# ==============================

import numpy as np
from pysr import PySRRegressor
import os

print("="*70)
print("MINIMAL PySR TEST")
print("="*70)

# Create simple test data
np.random.seed(42)
X_test = np.random.randn(100, 3)
y_test = 2*X_test[:, 0] - 1.5*X_test[:, 1] + 0.5*X_test[:, 2] + 0.1*np.random.randn(100)

print(f"\nTest data shape: X {X_test.shape}, y {y_test.shape}")

# Simplest possible model
model_test = PySRRegressor(
    niterations=100,
    populations=3,
    population_size=50,
    maxsize=10,
    binary_operators=["+", "*"],
    unary_operators=["square"],
    procs=1,
    loss="L2DistLoss()",
    random_state=42,
    temp_equation_file=True
)

try:
    print("\nFitting test model...")
    model_test.fit(X_test, y_test)
    
    if hasattr(model_test, 'equations_') and model_test.equations_ is not None:
        print(f"\n✅ Test successful! Found {len(model_test.equations_)} equations")
        print("\nTop equations:")
        print(model_test.equations_.head())
        
        # Try to save
        test_csv = r"C:\Users\Amare\Desktop\test_equations.csv"
        model_test.equations_.to_csv(test_csv, index=False)
        print(f"\n✅ Test equations saved to: {test_csv}")
    else:
        print("\n❌ No equations found in test")
        
except Exception as e:
    print(f"\n❌ Test failed: {e}")

print("\n" + "="*70)