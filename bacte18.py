# ==============================
# COMPLETE WORKING DIZ PREDICTOR
# ==============================

import pickle
import numpy as np
import os

# Path to your model files
model_dir = r"C:\Users\Amare\Desktop\pysr_final_manual_save"
scaler_path = os.path.join(model_dir, "scaler.pkl")

# Load the scaler
print("Loading scaler...")
with open(scaler_path, 'rb') as f:
    scaler = pickle.load(f)
print("✅ Scaler loaded!")

# DEFINE THE PREDICTION FUNCTION (this was missing!)
def predict_diz(Cbac, CAgNP, Ra):
    """
    Predict DIZ from raw feature values
    
    Parameters:
    Cbac: Bacteria concentration (raw units)
    CAgNP: Silver nanoparticles concentration (raw units)
    Ra: Roughness (raw units)
    
    Returns:
    Predicted DIZ value
    """
    # Scale the input features
    X_raw = np.array([[Cbac, CAgNP, Ra]])
    X_scaled = scaler.transform(X_raw)
    
    # Apply the equation: 13.503544 - ((x2 + ((x1 + 0.3436411) * x0)) / ((x0 + 1.1927102) ** 2))
    x0, x1, x2 = X_scaled[0]
    result = 13.503544 - ((x2 + ((x1 + 0.3436411) * x0)) / ((x0 + 1.1927102) ** 2))
    
    return result

# NOW test different combinations
print("\n🔬 TESTING DIFFERENT COMBINATIONS")
print("-"*50)

combinations = [
    (5000000, 100, 15),   # Low everything
    (10000000, 500, 30),  # Medium
    (20000000, 1000, 45), # High
    (15000000, 750, 35),  # Medium-high
    (3000000, 50, 10),    # Very low
    (25000000, 1500, 50), # Very high
]

for cbac, cagnp, ra in combinations:
    diz = predict_diz(cbac, cagnp, ra)
    print(f"Cbac={cbac:8.0f}, CAgNP={cagnp:4.0f}, Ra={ra:2.0f} → DIZ={diz:6.2f}")

print("\n" + "="*70)
print("✅ PREDICTOR IS WORKING!")
print("="*70)

# Optional: Save as a module for easy import
module_code = '''
import pickle
import numpy as np
import os

class DIZPredictor:
    """Predictor for DIZ using the discovered equation"""
    
    def __init__(self, scaler_path):
        """Initialize with path to scaler.pkl"""
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
    
    def predict(self, Cbac, CAgNP, Ra):
        """Predict DIZ from raw values"""
        X_raw = np.array([[Cbac, CAgNP, Ra]])
        X_scaled = self.scaler.transform(X_raw)
        x0, x1, x2 = X_scaled[0]
        return 13.503544 - ((x2 + ((x1 + 0.3436411) * x0)) / ((x0 + 1.1927102) ** 2))
    
    def predict_batch(self, X):
        """Predict DIZ for multiple samples"""
        X_scaled = self.scaler.transform(X)
        predictions = []
        for x in X_scaled:
            x0, x1, x2 = x
            pred = 13.503544 - ((x2 + ((x1 + 0.3436411) * x0)) / ((x0 + 1.1927102) ** 2))
            predictions.append(pred)
        return np.array(predictions)

# Example usage
if __name__ == "__main__":
    predictor = DIZPredictor(r"C:\\Users\\Amare\\Desktop\\pysr_final_manual_save\\scaler.pkl")
    diz = predictor.predict(10000000, 500, 30)
    print(f"Predicted DIZ: {diz:.2f}")
'''

# Save the module
module_path = os.path.join(os.path.dirname(scaler_path), "diz_predictor.py")
with open(module_path, 'w') as f:
    f.write(module_code)
print(f"\n✅ Module saved to: {module_path}")

print("\n" + "="*70)
print("USAGE INSTRUCTIONS")
print("="*70)
print("""
Option 1: Use the function directly in your script:
----------------------------------------
from diz_predictor_script import predict_diz, scaler

diz = predict_diz(10000000, 500, 30)
print(f"Predicted DIZ: {diz:.2f}")

Option 2: Use the class (recommended for multiple predictions):
----------------------------------------
from diz_predictor import DIZPredictor

predictor = DIZPredictor(r"C:\\Users\\Amare\\Desktop\\pysr_final_manual_save\\scaler.pkl")
diz = predictor.predict(10000000, 500, 30)
print(f"Predicted DIZ: {diz:.2f}")

# Batch prediction
import numpy as np
X_test = np.array([[10000000, 500, 30], [5000000, 200, 20]])
results = predictor.predict_batch(X_test)
""")