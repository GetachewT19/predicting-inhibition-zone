# ==============================
# DIAGNOSTIC: Why New Dataset Gives Lower R²
# ==============================

import pandas as pd
import numpy as np
from pysr import PySRRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import os
import pickle

# Paths
data_path = r"C:\Users\Amare\Desktop\getachewt\datato232-81.csv"
output_dir = r"C:\Users\Amare\Desktop\diz_diagnostic_results"
os.makedirs(output_dir, exist_ok=True)

print("="*70)
print("DIAGNOSTIC: Why New Dataset Gives Lower R²")
print("="*70)

# Load data
df = pd.read_csv(data_path)
target = 'DIZ'
features = ['Cbac', 'CAgNP', 'Ra']

# Clean data
df_clean = df.copy()
for col in [target] + features:
    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
df_clean = df_clean.dropna()

X = df_clean[features].values.astype(float)
y = df_clean[target].values.astype(float)

print(f"\n📊 Dataset: {len(y)} samples")

# ------------------------------
# TEST 1: Check if data is identical to old dataset
# ------------------------------
print("\n" + "="*70)
print("TEST 1: Data Integrity Check")
print("="*70)

# Check if data is exactly the same as old dataset
# (This would indicate a file path issue)
print("\nFirst 5 rows of data:")
print(df_clean.head())

print("\nData types:")
print(df_clean.dtypes)

print("\nAny null values?")
print(df_clean.isnull().sum())

# ------------------------------
# TEST 2: Simple Linear Regression Baseline
# ------------------------------
print("\n" + "="*70)
print("TEST 2: Linear Regression Baseline")
print("="*70)

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
r2_lr = r2_score(y_test, y_pred_lr)
print(f"Linear Regression R²: {r2_lr:.4f}")

# ------------------------------
# TEST 3: Random Forest Baseline (theoretical max)
# ------------------------------
print("\n" + "="*70)
print("TEST 3: Random Forest Baseline")
print("="*70)

rf = RandomForestRegressor(n_estimators=100, random_state=42)
scores = cross_val_score(rf, X, y, cv=5, scoring='r2')
print(f"Random Forest R²: {scores.mean():.4f} (+/- {scores.std()*2:.4f})")

# ------------------------------
# TEST 4: Run PySR with Explicit Equation Saving
# ------------------------------
print("\n" + "="*70)
print("TEST 4: PySR with Explicit Equation Saving")
print("="*70)

# Scale features
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)

# Save scaler
scaler_path = os.path.join(output_dir, "scaler.pkl")
with open(scaler_path, 'wb') as f:
    pickle.dump(scaler, f)

# Run PySR with different configurations
configs = [
    {
        'name': 'Simple',
        'niterations': 200,
        'populations': 5,
        'maxsize': 10,
        'binary': ["+", "-", "*"],
        'unary': ["square"]
    },
    {
        'name': 'Medium',
        'niterations': 500,
        'populations': 8,
        'maxsize': 15,
        'binary': ["+", "-", "*", "/"],
        'unary': ["square", "sqrt"]
    }
]

best_r2 = 0
best_eq = None

for config in configs:
    print(f"\n🔄 Testing {config['name']} configuration...")
    
    model = PySRRegressor(
        niterations=config['niterations'],
        populations=config['populations'],
        population_size=50,
        maxsize=config['maxsize'],
        binary_operators=config['binary'],
        unary_operators=config['unary'],
        procs=2,
        loss="L2DistLoss()",
        progress=True,
        random_state=42,
        temp_equation_file=True
    )
    
    model.fit(X_scaled, y)
    
    if hasattr(model, 'equations_') and model.equations_ is not None:
        equations = model.equations_
        
        # Save equations for this config
        csv_path = os.path.join(output_dir, f"equations_{config['name']}.csv")
        equations.to_csv(csv_path, index=False)
        
        # Get best equation
        best_idx = equations['loss'].idxmin()
        best_eq_config = equations.loc[best_idx, 'equation']
        
        # Make predictions
        y_pred = model.predict(X_scaled)
        r2 = r2_score(y, y_pred)
        
        print(f"  Found {len(equations)} equations")
        print(f"  Best R²: {r2:.4f}")
        print(f"  Best equation: {best_eq_config[:100]}...")
        
        if r2 > best_r2:
            best_r2 = r2
            best_eq = best_eq_config
            best_model = model

# ------------------------------
# TEST 5: Try the old equation on new data
# ------------------------------
print("\n" + "="*70)
print("TEST 5: Apply Old Equation to New Data")
print("="*70)

old_eq = "13.503544 - ((x2 + ((x1 + 0.3436411) * x0)) / ((x0 + 1.1927102) ** 2))"

def evaluate_old_eq(x0, x1, x2):
    try:
        return 13.503544 - ((x2 + ((x1 + 0.3436411) * x0)) / ((x0 + 1.1927102) ** 2))
    except:
        return np.nan

y_pred_old = []
for i in range(len(y)):
    pred = evaluate_old_eq(X_scaled[i,0], X_scaled[i,1], X_scaled[i,2])
    y_pred_old.append(pred)

y_pred_old = np.array(y_pred_old)
valid_mask = ~np.isnan(y_pred_old)
if valid_mask.sum() > 0:
    r2_old = r2_score(y[valid_mask], y_pred_old[valid_mask])
    print(f"Old equation R² on new data: {r2_old:.4f}")
else:
    print("Old equation failed on new data")

# ------------------------------
# FINAL SUMMARY
# ------------------------------
print("\n" + "="*70)
print("DIAGNOSTIC SUMMARY")
print("="*70)
print(f"""
Linear Regression R²: {r2_lr:.4f}
Random Forest R²: {scores.mean():.4f}
Best PySR R²: {best_r2:.4f}
Old Equation R²: {r2_old:.4f if 'r2_old' in locals() else 'N/A'}

If PySR R² << Random Forest R²:
→ PySR needs more iterations or better parameters

If all models give low R²:
→ The relationship is genuinely weaker in this dataset

If PySR gives different results each run:
→ Random seed issue - set random_state=42 consistently
""")

print(f"\n✅ Diagnostic results saved to: {output_dir}")