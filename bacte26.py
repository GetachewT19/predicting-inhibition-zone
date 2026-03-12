# ==============================
# DIZ PREDICTOR - R² = 0.7749
# Complete with Regression Plots
# ==============================

import pickle
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import seaborn as sns

# Set style for better plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

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
# PART 3: REGRESSION PLOTTING FUNCTIONS
# ==============================

def plot_regression_results(y_true, y_pred, title="DIZ Regression Results", save_path=None):
    """
    Create comprehensive regression plots
    
    Parameters:
    y_true: actual DIZ values
    y_pred: predicted DIZ values
    title: plot title
    save_path: where to save the plot (optional)
    """
    # Calculate metrics
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100
    
    residuals = y_true - y_pred
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Predicted vs Actual (main plot)
    ax1 = plt.subplot(2, 3, 1)
    ax1.scatter(y_true, y_pred, alpha=0.6, c='blue', edgecolors='black', linewidth=0.5, s=60)
    
    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    ax1.set_xlabel('Actual DIZ Values', fontsize=12)
    ax1.set_ylabel('Predicted DIZ Values', fontsize=12)
    ax1.set_title(f'Predicted vs Actual\nR² = {r2:.4f}', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add metrics box
    textstr = f'R² = {r2:.4f}\nMAE = {mae:.4f}\nRMSE = {rmse:.4f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax1.text(0.05, 0.95, textstr, transform=ax1.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    # 2. Residuals vs Predicted
    ax2 = plt.subplot(2, 3, 2)
    ax2.scatter(y_pred, residuals, alpha=0.6, c='green', edgecolors='black', linewidth=0.5, s=60)
    ax2.axhline(y=0, color='r', linestyle='--', lw=2)
    ax2.set_xlabel('Predicted DIZ Values', fontsize=12)
    ax2.set_ylabel('Residuals', fontsize=12)
    ax2.set_title(f'Residual Plot\nMean = {np.mean(residuals):.4f}, Std = {np.std(residuals):.4f}', 
                 fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # 3. Residuals Distribution
    ax3 = plt.subplot(2, 3, 3)
    ax3.hist(residuals, bins=25, edgecolor='black', alpha=0.7, color='purple')
    ax3.axvline(x=0, color='r', linestyle='--', lw=2)
    ax3.axvline(x=np.mean(residuals), color='orange', linestyle='--', lw=2, label=f'Mean: {np.mean(residuals):.3f}')
    ax3.set_xlabel('Residuals', fontsize=12)
    ax3.set_ylabel('Frequency', fontsize=12)
    ax3.set_title('Residuals Distribution', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Q-Q Plot (Normality check)
    ax4 = plt.subplot(2, 3, 4)
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=ax4)
    ax4.set_title('Q-Q Plot (Normality Check)', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # 5. Actual vs Predicted Line Plot (first 50 samples)
    ax5 = plt.subplot(2, 3, 5)
    n_show = min(50, len(y_true))
    indices = np.arange(n_show)
    ax5.plot(indices, y_true[:n_show], 'b-', label='Actual', linewidth=2, marker='o', markersize=4)
    ax5.plot(indices, y_pred[:n_show], 'r--', label='Predicted', linewidth=2, marker='s', markersize=4)
    ax5.set_xlabel('Sample Index', fontsize=12)
    ax5.set_ylabel('DIZ Value', fontsize=12)
    ax5.set_title(f'Actual vs Predicted (First {n_show} Samples)', fontsize=14, fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Error Distribution
    ax6 = plt.subplot(2, 3, 6)
    errors = np.abs(residuals)
    ax6.boxplot(errors, vert=True, patch_artist=True)
    ax6.set_ylabel('Absolute Error', fontsize=12)
    ax6.set_title(f'Error Distribution\nMedian Error: {np.median(errors):.4f}', 
                 fontsize=14, fontweight='bold')
    ax6.set_xticklabels(['Absolute Errors'])
    ax6.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Plot saved to: {save_path}")
    
    plt.show()
    
    # Print summary metrics
    print("\n" + "="*70)
    print("REGRESSION METRICS SUMMARY")
    print("="*70)
    print(f"R² Score:  {r2:.6f} ({r2*100:.2f}% variance explained)")
    print(f"MAE:       {mae:.6f}")
    print(f"RMSE:      {rmse:.6f}")
    print(f"MAPE:      {mape:.2f}%")
    print(f"Residuals: Mean = {np.mean(residuals):.6f}, Std = {np.std(residuals):.6f}")
    print("="*70)


def plot_feature_importance(predictor, X_scaled, y_true, feature_names, save_path=None):
    """
    Plot feature importance analysis
    
    Parameters:
    predictor: trained DIZPredictor
    X_scaled: scaled features
    y_true: actual DIZ values
    feature_names: list of feature names
    save_path: where to save the plot
    """
    y_pred = predictor.predict_batch(X_scaled)
    baseline_r2 = r2_score(y_true, y_pred)
    
    # Calculate permutation importance
    n_features = X_scaled.shape[1]
    importance = []
    
    for i in range(n_features):
        X_permuted = X_scaled.copy()
        np.random.shuffle(X_permuted[:, i])
        y_pred_permuted = predictor.predict_batch(X_permuted)
        r2_permuted = r2_score(y_true, y_pred_permuted)
        importance.append(baseline_r2 - r2_permuted)
    
    # Create plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Bar plot of feature importance
    colors = ['red' if imp < 0 else 'green' for imp in importance]
    bars = ax1.bar(feature_names, importance, color=colors, edgecolor='black', alpha=0.7)
    ax1.set_xlabel('Features', fontsize=12)
    ax1.set_ylabel('Importance (drop in R²)', fontsize=12)
    ax1.set_title('Feature Importance Analysis', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, imp in zip(bars, importance):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{imp:.4f}', ha='center', va='bottom' if height > 0 else 'top', fontsize=10)
    
    # Partial dependence plots
    for i, feat in enumerate(feature_names):
        # Create range for this feature
        feat_range = np.linspace(X_scaled[:, i].min(), X_scaled[:, i].max(), 50)
        
        # Keep other features at mean
        X_temp = np.tile(X_scaled.mean(axis=0), (50, 1))
        X_temp[:, i] = feat_range
        
        # Predict
        y_partial = predictor.predict_batch(X_temp)
        
        ax2.plot(feat_range, y_partial, label=feat, linewidth=2)
    
    ax2.set_xlabel('Feature Value (scaled)', fontsize=12)
    ax2.set_ylabel('Predicted DIZ', fontsize=12)
    ax2.set_title('Partial Dependence Plots', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle('Feature Importance and Effects', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Feature importance plot saved to: {save_path}")
    
    plt.show()


# ==============================
# PART 4: MAIN EXECUTION WITH PLOTS
# ==============================

if __name__ == "__main__":
    
    print("="*70)
    print("DIZ PREDICTOR - R² = 0.7749")
    print("="*70)
    
    # Configuration
    output_dir = r"C:\Users\Amare\Desktop\diz_good_results_0.7749"
    scaler_path = os.path.join(output_dir, "scaler.pkl")
    data_path = r"C:\Users\Amare\Desktop\getachewt\datato232-81.csv"
    
    # Initialize predictor
    predictor = DIZPredictor(scaler_path)
    
    # Load some data for plotting (if available)
    if os.path.exists(data_path):
        print(f"\n📊 Loading data from: {data_path}")
        df = pd.read_csv(data_path)
        features = ['Cbac', 'CAgNP', 'Ra']
        
        # Clean data
        df_clean = df.copy()
        for col in features + ['DIZ']:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        df_clean = df_clean.dropna()
        
        X_raw = df_clean[features].values.astype(float)
        y_true = df_clean['DIZ'].values.astype(float)
        
        # Scale features
        if predictor.scaler:
            X_scaled = predictor.scaler.transform(X_raw)
        else:
            # Create scaler if not available
            predictor.scaler = create_scaler_from_data(data_path, scaler_path)
            X_scaled = predictor.scaler.transform(X_raw)
        
        # Make predictions
        y_pred = predictor.predict_batch(X_scaled)
        
        # Calculate and display metrics
        r2 = r2_score(y_true, y_pred)
        print(f"\n📈 Model Performance on loaded data:")
        print(f"   R² = {r2:.4f} ({r2*100:.2f}%)")
        
        # Create regression plots
        plot_regression_results(
            y_true, y_pred,
            title=f"DIZ Regression Results (R² = {r2:.4f})",
            save_path=os.path.join(output_dir, "regression_plots.png")
        )
        
        # Create feature importance plot
        plot_feature_importance(
            predictor, X_scaled, y_true,
            feature_names=['Cbac', 'CAgNP', 'Ra'],
            save_path=os.path.join(output_dir, "feature_importance.png")
        )
        
    else:
        print(f"\n⚠️ Data file not found: {data_path}")
        print("Creating synthetic data for demonstration...")
        
        # Generate synthetic data for demonstration
        np.random.seed(42)
        n_samples = 200
        X_synth = np.random.randn(n_samples, 3)
        
        # Generate synthetic DIZ using the equation
        y_synth = []
        for i in range(n_samples):
            x0, x1, x2 = X_synth[i]
            y = 13.052508 + (x0 * ((0.6055002 + x1) / ((x2 + 1.1389034) - (x0 ** 2))))
            # Add some noise
            y += np.random.randn() * 2
            y_synth.append(y)
        y_synth = np.array(y_synth)
        
        # Make predictions
        y_pred_synth = predictor.predict_batch(X_synth)
        
        # Plot synthetic results
        plot_regression_results(
            y_synth, y_pred_synth,
            title="DIZ Regression Results (Synthetic Data Demo)",
            save_path=os.path.join(output_dir, "demo_regression_plots.png")
        )
    
    print("\n" + "="*70)
    print("✅ Analysis complete! Check the output directory for plots.")
    print("="*70)
    print(f"\n📁 All files saved to: {output_dir}")
    print("\nFiles created:")
    print("  - regression_plots.png (comprehensive regression diagnostics)")
    print("  - feature_importance.png (feature importance analysis)")
    print("  - best_equation.txt (the equation)")
    print("  - scaler.pkl (feature scaler)")