# ==============================
# DIZ PREDICTOR - R² = 0.7749
# Complete Ready-to-Use Code
# ==============================

import pickle
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

# ==============================
# PART 1: THE PREDICTOR CLASS
# ==============================

class DIZPredictor:
    """
    Predictor for DIZ using the discovered equation
    R² = 0.7749 (77.5% variance explained)
    Equation: 13.052508 + (x0 * ((0.6055002 + x1) / ((x2 + 1.1389034) - square(x0))))
    """
    
    def __init__(self, scaler_path=None):
        """
        Initialize the predictor
        
        Parameters:
        scaler_path: path to the saved RobustScaler.pkl file
                     If None, assumes inputs are already scaled
        """
        self.scaler = None
        if scaler_path and os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            print(f"✅ Scaler loaded from: {scaler_path}")
        elif scaler_path:
            print(f"⚠️ Scaler not found at: {scaler_path}")
    
    def predict(self, x0, x1, x2):
        """
        Predict DIZ from scaled feature values
        
        Parameters:
        x0: Cbac (bacteria concentration) - SCALED value
        x1: CAgNP (silver nanoparticles) - SCALED value
        x2: Ra (roughness) - SCALED value
        
        Returns:
        Predicted DIZ value
        """
        # Apply the discovered equation
        # 13.052508 + (x0 * ((0.6055002 + x1) / ((x2 + 1.1389034) - (x0 ** 2))))
        numerator = 0.6055002 + x1
        denominator = (x2 + 1.1389034) - (x0 ** 2)
        
        # Avoid division by zero
        if abs(denominator) < 1e-10:
            denominator = 1e-10
            
        result = 13.052508 + (x0 * (numerator / denominator))
        return result
    
    def predict_raw(self, Cbac_raw, CAgNP_raw, Ra_raw):
        """
        Predict DIZ from raw (unscaled) feature values
        
        Parameters:
        Cbac_raw: Bacteria concentration (original units)
        CAgNP_raw: Silver nanoparticles concentration (original units)
        Ra_raw: Roughness (original units)
        
        Returns:
        Predicted DIZ value
        """
        if self.scaler is None:
            raise ValueError("Scaler not loaded. Cannot predict from raw values.")
        
        # Scale the raw values
        X_raw = np.array([[Cbac_raw, CAgNP_raw, Ra_raw]])
        X_scaled = self.scaler.transform(X_raw)
        
        # Predict using scaled values
        return self.predict(X_scaled[0, 0], X_scaled[0, 1], X_scaled[0, 2])
    
    def predict_batch(self, X_scaled):
        """
        Predict DIZ for multiple samples
        
        Parameters:
        X_scaled: array of shape (n_samples, 3) with scaled features
                  Columns: [Cbac, CAgNP, Ra]
        
        Returns:
        array of predicted DIZ values
        """
        predictions = []
        for i in range(len(X_scaled)):
            pred = self.predict(X_scaled[i, 0], X_scaled[i, 1], X_scaled[i, 2])
            predictions.append(pred)
        return np.array(predictions)
    
    def predict_batch_raw(self, X_raw):
        """
        Predict DIZ for multiple raw samples
        
        Parameters:
        X_raw: array of shape (n_samples, 3) with raw features
               Columns: [Cbac_raw, CAgNP_raw, Ra_raw]
        
        Returns:
        array of predicted DIZ values
        """
        if self.scaler is None:
            raise ValueError("Scaler not loaded. Cannot predict from raw values.")
        
        X_scaled = self.scaler.transform(X_raw)
        return self.predict_batch(X_scaled)


# ==============================
# PART 2: TRAIN THE SCALER (if you don't have it)
# ==============================

def create_scaler_from_data(data_path, save_path=None):
    """
    Create and save a RobustScaler from your data
    
    Parameters:
    data_path: path to your CSV file
    save_path: where to save the scaler (optional)
    
    Returns:
    trained scaler
    """
    from sklearn.preprocessing import RobustScaler
    
    # Load data
    df = pd.read_csv(data_path)
    features = ['Cbac', 'CAgNP', 'Ra']
    
    # Clean data
    df_clean = df.copy()
    for col in features:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    df_clean = df_clean.dropna()
    
    X = df_clean[features].values.astype(float)
    
    # Train scaler
    scaler = RobustScaler()
    scaler.fit(X)
    
    # Save if path provided
    if save_path:
        with open(save_path, 'wb') as f:
            pickle.dump(scaler, f)
        print(f"✅ Scaler saved to: {save_path}")
    
    return scaler


# ==============================
# PART 3: EXAMPLE USAGE
# ==============================

if __name__ == "__main__":
    
    print("="*70)
    print("DIZ PREDICTOR - R² = 0.7749")
    print("="*70)
    
    # Option A: If you have the scaler file
    scaler_path = r"C:\Users\Amare\Desktop\diz_good_results_0.7749\scaler.pkl"
    
    # Option B: Create scaler from your data (uncomment if needed)
    # data_path = r"C:\Users\Amare\Desktop\getachewt\datato232-81.csv"
    # scaler = create_scaler_from_data(data_path, scaler_path)
    
    # Initialize predictor
    predictor = DIZPredictor(scaler_path)
    
    # Example 1: Predict from scaled values
    print("\n📊 EXAMPLE 1: Predict from scaled values")
    print("-"*50)
    
    # These are example scaled values (mean ≈ 0, std ≈ 1)
    test_scaled = [
        [0.0, 0.0, 0.0],      # Average values
        [1.0, 0.5, -0.3],      # Above average Cbac
        [-0.8, 1.2, 0.7],      # High CAgNP
        [1.5, -0.5, 1.0],      # High Cbac and Ra
    ]
    
    for i, sample in enumerate(test_scaled):
        diz = predictor.predict(sample[0], sample[1], sample[2])
        print(f"  Sample {i+1}: Cbac={sample[0]:.2f}, CAgNP={sample[1]:.2f}, Ra={sample[2]:.2f} → DIZ={diz:.2f}")
    
    # Example 2: Predict from raw values (if scaler is available)
    if predictor.scaler:
        print("\n📊 EXAMPLE 2: Predict from raw values")
        print("-"*50)
        
        test_raw = [
            [10000000, 500, 30],    # Medium bacteria
            [5000000, 200, 20],      # Low bacteria
            [20000000, 1000, 40],    # High bacteria
            [15000000, 750, 35],     # Medium-high bacteria
        ]
        
        for i, (cbac, cagnp, ra) in enumerate(test_raw):
            diz = predictor.predict_raw(cbac, cagnp, ra)
            print(f"  Sample {i+1}: Cbac={cbac:8.0f}, CAgNP={cagnp:4.0f}, Ra={ra:2.0f} → DIZ={diz:6.2f}")
    
    # Example 3: Batch prediction
    print("\n📊 EXAMPLE 3: Batch prediction")
    print("-"*50)
    
    X_batch = np.array(test_scaled)
    batch_results = predictor.predict_batch(X_batch)
    print(f"  Batch predictions: {batch_results}")
    
    # Example 4: Sensitivity analysis
    print("\n📊 EXAMPLE 4: How Cbac affects DIZ")
    print("-"*50)
    
    # Fix CAgNP and Ra at average, vary Cbac
    cagnp_fixed = 0.0
    ra_fixed = 0.0
    cbac_values = np.linspace(-2, 2, 10)
    
    print(f"  CAgNP fixed at {cagnp_fixed}, Ra fixed at {ra_fixed}")
    for cbac in cbac_values:
        diz = predictor.predict(cbac, cagnp_fixed, ra_fixed)
        print(f"    Cbac={cbac:5.2f} → DIZ={diz:6.2f}")
    
    print("\n" + "="*70)
    print("✅ Predictor is ready to use!")
    print("="*70)
    print("\n📁 Equation saved in: C:\\Users\\Amare\\Desktop\\diz_good_results_0.7749\\best_equation.txt")
    print("\n🔬 Equation: 13.052508 + (x0 * ((0.6055002 + x1) / ((x2 + 1.1389034) - square(x0))))")
    print("   where: x0 = Cbac (scaled), x1 = CAgNP (scaled), x2 = Ra (scaled)")