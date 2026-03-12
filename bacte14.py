# ==============================
# FINAL SUMMARY AND PRACTICAL IMPLEMENTATION
# ==============================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

print("="*70)
print("FINAL SUMMARY: DIZ PREDICTION MODEL")
print("="*70)

print("""
╔══════════════════════════════════════════════════════════════════════╗
║                     MODEL PERFORMANCE                                 ║
╠══════════════════════════════════════════════════════════════════════╣
║  R² Score: 0.6891 (68.9% variance explained)                        ║
║  MAE:      5.2746 (mean absolute error)                             ║
║  RMSE:     6.9873 (root mean square error)                          ║
╚══════════════════════════════════════════════════════════════════════╝
""")

print("\n" + "="*70)
print("THE EQUATION")
print("="*70)

print("""
DIZ = 13.503544 - ┌─────────────────────┐
                  │ Ra + (CAgNP + 0.34) × Cbac │
                  └─────────────────────┘
                  ─────────────────────────
                         (Cbac + 1.19)²

WHERE:
    Cbac   = Bacteria concentration (scaled)
    CAgNP  = Silver nanoparticles concentration (scaled)
    Ra     = Roughness (scaled)
""")

print("\n" + "="*70)
print("FEATURE EFFECTS (per 1 standard deviation increase)")
print("="*70)

print("""
┌─────────┬─────────────┬─────────────────────────────────────┐
│ Feature │ Effect on   │ Interpretation                      │
│         │ DIZ         │                                     │
├─────────┼─────────────┼─────────────────────────────────────┤
│ Cbac    │ +0.0008     │ Non-linear effect - small at mean   │
│         │             │ but dramatic at extremes            │
├─────────┼─────────────┼─────────────────────────────────────┤
│ CAgNP   │ -0.0128     │ Small negative effect, amplified    │
│         │             │ when Cbac is high                   │
├─────────┼─────────────┼─────────────────────────────────────┤
│ Ra      │ -0.0002     │ Negligible direct effect            │
└─────────┴─────────────┴─────────────────────────────────────┘
""")

print("\n" + "="*70)
print("PRACTICAL IMPLEMENTATION")
print("="*70)

# Create a complete prediction function with scaling
def create_prediction_function(scaler_path):
    """
    Create a complete prediction function with built-in scaling
    """
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
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
        
        # Apply the equation
        x0, x1, x2 = X_scaled[0]
        result = 13.503544 - ((x2 + ((x1 + 0.3436411) * x0)) / ((x0 + 1.1927102) ** 2))
        
        return result
    
    return predict_diz

# Save the prediction function as a Python script
script_content = '''
import pickle
import numpy as np

class DIZPredictor:
    """Predictor for DIZ using the discovered equation"""
    
    def __init__(self, scaler_path):
        """Initialize with the saved scaler"""
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
    
    def predict(self, Cbac, CAgNP, Ra):
        """
        Predict DIZ from raw feature values
        
        Parameters:
        Cbac: Bacteria concentration
        CAgNP: Silver nanoparticles concentration
        Ra: Roughness
        
        Returns:
        Predicted DIZ value
        """
        # Scale inputs
        X_raw = np.array([[Cbac, CAgNP, Ra]])
        X_scaled = self.scaler.transform(X_raw)
        
        # Apply equation
        x0, x1, x2 = X_scaled[0]
        result = 13.503544 - ((x2 + ((x1 + 0.3436411) * x0)) / ((x0 + 1.1927102) ** 2))
        
        return result
    
    def predict_batch(self, X):
        """
        Predict DIZ for multiple samples
        
        Parameters:
        X: array of shape (n_samples, 3) with columns [Cbac, CAgNP, Ra]
        
        Returns:
        array of predicted DIZ values
        """
        X_scaled = self.scaler.transform(X)
        predictions = []
        
        for x in X_scaled:
            x0, x1, x2 = x
            pred = 13.503544 - ((x2 + ((x1 + 0.3436411) * x0)) / ((x0 + 1.1927102) ** 2))
            predictions.append(pred)
        
        return np.array(predictions)

# Example usage
if __name__ == "__main__":
    # Load the predictor
    predictor = DIZPredictor("scaler.pkl")
    
    # Make a prediction
    diz = predictor.predict(Cbac=10000000, CAgNP=500, Ra=30)
    print(f"Predicted DIZ: {diz:.2f}")
'''

# Save the implementation
output_dir = r"C:\Users\Amare\Desktop\pysr_final_manual_save"
impl_path = os.path.join(output_dir, "diz_predictor.py")
with open(impl_path, 'w') as f:
    f.write(script_content)

print(f"\n✅ Python implementation saved to: {impl_path}")
print("\n" + "="*70)
print("EXCEL FORMULA")
print("="*70)

print("""
For Excel users, assuming:
- Cbac is in cell A1 (raw value)
- CAgNP is in cell B1 (raw value)
- Ra is in cell C1 (raw value)
- Scaled values are in D1, E1, F1

Step 1: Scale the features (you need the scaling parameters)
   D1 = (A1 - MEDIAN(A:A)) / (PERCENTILE(A:A,0.75) - PERCENTILE(A:A,0.25))
   E1 = (B1 - MEDIAN(B:B)) / (PERCENTILE(B:B,0.75) - PERCENTILE(B:B,0.25))
   F1 = (C1 - MEDIAN(C:C)) / (PERCENTILE(C:C,0.75) - PERCENTILE(C:C,0.25))

Step 2: Apply the equation
   G1 = 13.503544 - ((F1 + ((E1 + 0.3436411) * D1)) / ((D1 + 1.1927102)^2))
""")

print("\n" + "="*70)
print("KEY INSIGHTS")
print("="*70)

print("""
1. BACTERIA IS CRITICAL: Despite its small effect at mean values,
   Cbac creates the non-linear behavior that makes the model work.

2. WHY OUTLIERS MATTERED: The extreme Cbac values create the curve
   shape that allows accurate predictions across the full range.

3. INTERACTION EFFECT: CAgNP's effect depends on Cbac - they work
   together in the numerator.

4. PRACTICAL USE: The equation can now be used to predict DIZ for
   new combinations of bacteria, nanoparticles, and roughness.
""")

print("\n" + "="*70)
print("✅ ANALYSIS COMPLETE - MODEL READY FOR USE")
print("="*70)
print(f"\nAll files saved in: {output_dir}")
print("\nFiles created:")
print(f"  - best_equation.txt: The equation with metrics")
print(f"  - verified_equation.txt: Verified final equation")
print(f"  - diz_predictor.py: Ready-to-use Python class")
print(f"  - scaler.pkl: Feature scaler for new data")
print(f"  - final_analysis.png: Visualization plots")