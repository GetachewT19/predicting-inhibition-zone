# ==============================
# WORKING DIZ PREDICTOR - READY TO USE
# ==============================

import pickle
import numpy as np
import os

# Path to your model files
model_dir = r"C:\Users\Amare\Desktop\pysr_final_manual_save"
scaler_path = os.path.join(model_dir, "scaler.pkl")

print("="*70)
print("DIZ PREDICTOR - READY TO USE")
print("="*70)

# Load the scaler
print(f"\n📂 Loading scaler from: {scaler_path}")
with open(scaler_path, 'rb') as f:
    scaler = pickle.load(f)
print("✅ Scaler loaded successfully!")

# Define the prediction function
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

# Test the function with some example values
print("\n📊 TEST PREDICTIONS")
print("-"*50)

test_cases = [
    (10000000, 500, 30, "Medium bacteria"),
    (5000000, 200, 20, "Low bacteria"),
    (20000000, 1000, 40, "High bacteria"),
    (15000000, 750, 35, "Medium-high bacteria"),
    (8000000, 300, 25, "Low-medium bacteria"),
]

for Cbac, CAgNP, Ra, description in test_cases:
    diz = predict_diz(Cbac, CAgNP, Ra)
    print(f"{description:20} | Cbac={Cbac:8.0f}, CAgNP={CAgNP:4.0f}, Ra={Ra:2.0f} → DIZ={diz:6.2f}")

print("\n" + "="*70)
print("BATCH PREDICTION EXAMPLE")
print("="*70)

# Create batch of samples
X_new = np.array([
    [10000000, 500, 30],
    [5000000, 200, 20],
    [20000000, 1000, 40],
    [15000000, 750, 35],
    [12000000, 600, 32],
])

# Scale all at once
X_scaled = scaler.transform(X_new)

# Predict all
predictions = []
for x in X_scaled:
    x0, x1, x2 = x
    pred = 13.503544 - ((x2 + ((x1 + 0.3436411) * x0)) / ((x0 + 1.1927102) ** 2))
    predictions.append(pred)

print("\nBatch predictions:")
for i, (sample, pred) in enumerate(zip(X_new, predictions)):
    print(f"  Sample {i+1}: Cbac={sample[0]:8.0f}, CAgNP={sample[1]:4.0f}, Ra={sample[2]:2.0f} → DIZ={pred:6.2f}")

print("\n" + "="*70)
print("SUMMARY STATISTICS")
print("="*70)

predictions = np.array(predictions)
print(f"\nPredictions statistics:")
print(f"  Mean: {predictions.mean():.2f}")
print(f"  Std:  {predictions.std():.2f}")
print(f"  Min:  {predictions.min():.2f}")
print(f"  Max:  {predictions.max():.2f}")

print("\n" + "="*70)
print("✅ PREDICTOR IS READY TO USE!")
print("="*70)
print("\nTo use in your own code, simply copy these lines:")

print("""
import pickle
import numpy as np

# Load the scaler
with open(r"C:\\Users\\Amare\\Desktop\\pysr_final_manual_save\\scaler.pkl", 'rb') as f:
    scaler = pickle.load(f)

def predict_diz(Cbac, CAgNP, Ra):
    X_raw = np.array([[Cbac, CAgNP, Ra]])
    X_scaled = scaler.transform(X_raw)
    x0, x1, x2 = X_scaled[0]
    return 13.503544 - ((x2 + ((x1 + 0.3436411) * x0)) / ((x0 + 1.1927102) ** 2))

# Example
diz = predict_diz(10000000, 500, 30)
print(f"Predicted DIZ: {diz:.2f}")
""")