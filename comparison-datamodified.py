# -*- coding: utf-8 -*-
"""
Created on Mon Dec 22 10:51:26 2025

@author: Amare
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor, StackingRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import Ridge
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. LOAD AND EXPLORE DATA
# ============================================================================
print("="*70)
print("LOADING AND EXPLORING DATA")
print("="*70)

file_path = r"C:\Users\Amare\Desktop\worku\newto\datato232-2.csv"
df = pd.read_csv(file_path)

print(f"Dataset shape: {df.shape}")
print(f"\nFirst 5 rows:")
print(df.head())
print(f"\nData types:")
print(df.dtypes)

# ============================================================================
# 2. IDENTIFY AND CLEAN DATA TYPES
# ============================================================================
print("\n" + "="*70)
print("CLEANING DATA TYPES")
print("="*70)

# Identify target column
target_candidates = [col for col in df.columns if 'inhib' in col.lower() or 'size' in col.lower() or 'mm' in col.lower()]
if target_candidates:
    target_column = target_candidates[0]
    print(f"Automatically detected target column: '{target_column}'")
else:
    print("Available columns:", df.columns.tolist())
    target_column = input("Enter the name of the target column: ")

print(f"Target column: {target_column}")

# Separate features and target
X = df.drop(columns=[target_column])
y = df[target_column]

# Convert target to numeric if needed
print(f"\nConverting target to numeric...")
y = pd.to_numeric(y, errors='coerce')

# Function to identify column types
def analyze_column(series, col_name):
    """Analyze a column to determine its type"""
    # Get non-null values
    non_null = series.dropna()
    if len(non_null) == 0:
        return 'empty'
    
    # Check if column contains numeric values
    numeric_mask = pd.to_numeric(non_null, errors='coerce').notna()
    num_numeric = numeric_mask.sum()
    
    # Check if column contains string values that can't be converted
    string_mask = ~numeric_mask
    num_string = string_mask.sum()
    
    if num_numeric == len(non_null):
        return 'numeric'
    elif num_string == len(non_null):
        # Check if it's categorical (limited unique values)
        unique_vals = non_null.astype(str).nunique()
        if unique_vals < 20:
            return 'categorical'
        else:
            return 'text'
    else:
        return 'mixed'

# Analyze each column
print(f"\nAnalyzing column types...")
column_types = {}
for col in X.columns:
    col_type = analyze_column(X[col], col)
    column_types[col] = col_type
    print(f"  {col}: {col_type}")

# Separate columns by type
numeric_cols = [col for col, typ in column_types.items() if typ == 'numeric']
categorical_cols = [col for col, typ in column_types.items() if typ == 'categorical']
text_cols = [col for col, typ in column_types.items() if typ == 'text']
mixed_cols = [col for col, typ in column_types.items() if typ == 'mixed']

print(f"\nSummary:")
print(f"  Numeric columns: {len(numeric_cols)}")
print(f"  Categorical columns: {len(categorical_cols)}")
print(f"  Text columns: {len(text_cols)}")
print(f"  Mixed columns: {len(mixed_cols)}")

# ============================================================================
# 3. CONVERT MIXED COLUMNS TO NUMERIC
# ============================================================================
print("\n" + "="*70)
print("CONVERTING MIXED COLUMNS")
print("="*70)

if mixed_cols:
    print(f"Converting {len(mixed_cols)} mixed columns to numeric...")
    for col in mixed_cols:
        print(f"\n  Converting '{col}':")
        original_non_null = X[col].notna().sum()
        
        # Try to convert to numeric
        X[col] = pd.to_numeric(X[col], errors='coerce')
        
        converted_non_null = X[col].notna().sum()
        conversion_rate = (converted_non_null / original_non_null * 100) if original_non_null > 0 else 0
        
        print(f"    Successfully converted {converted_non_null}/{original_non_null} values ({conversion_rate:.1f}%)")
        
        # Update column type
        if converted_non_null > 0:
            numeric_cols.append(col)
        else:
            categorical_cols.append(col)

# ============================================================================
# 4. HANDLE MISSING VALUES
# ============================================================================
print("\n" + "="*70)
print("HANDLING MISSING VALUES")
print("="*70)

# Remove rows with missing target
if y.isnull().sum() > 0:
    print(f"Removing {y.isnull().sum()} rows with missing target values...")
    mask = y.notna()
    X = X[mask]
    y = y[mask]

print(f"Dataset after removing missing target: {X.shape}")

# Handle missing values in features
print(f"\nMissing values in features: {X.isnull().sum().sum()}")

# Impute numeric columns
if numeric_cols:
    print(f"\nImputing {len(numeric_cols)} numeric columns with mean...")
    num_imputer = SimpleImputer(strategy='mean')
    X[numeric_cols] = num_imputer.fit_transform(X[numeric_cols])

# Impute categorical columns
if categorical_cols:
    print(f"Imputing {len(categorical_cols)} categorical columns with most_frequent...")
    cat_imputer = SimpleImputer(strategy='most_frequent')
    X[categorical_cols] = cat_imputer.fit_transform(X[categorical_cols])

# Drop text columns (they're hard to use in regression)
if text_cols:
    print(f"\nDropping {len(text_cols)} text columns (not suitable for regression): {text_cols}")
    X = X.drop(columns=text_cols)

print(f"\nMissing values after imputation: {X.isnull().sum().sum()}")

# ============================================================================
# 5. ENCODE CATEGORICAL VARIABLES
# ============================================================================
print("\n" + "="*70)
print("ENCODING CATEGORICAL VARIABLES")
print("="*70)

# For CatBoost: Keep categorical columns as is (strings)
X_catboost = X.copy()

# For other models: One-hot encode categorical columns
if categorical_cols:
    print(f"One-hot encoding {len(categorical_cols)} categorical columns...")
    X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    print(f"  Features before encoding: {X.shape[1]}")
    print(f"  Features after encoding: {X_encoded.shape[1]}")
else:
    X_encoded = X.copy()

# ============================================================================
# 6. SPLIT AND SCALE DATA
# ============================================================================
print("\n" + "="*70)
print("SPLITTING AND SCALING DATA")
print("="*70)

# Split the data
X_train_encoded, X_test_encoded, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42
)

# Also split the CatBoost version
X_train_catboost, X_test_catboost = train_test_split(
    X_catboost, test_size=0.2, random_state=42
)

print(f"Training set: {X_train_encoded.shape}")
print(f"Test set: {X_test_encoded.shape}")

# Scale features for GradientBoosting and XGBoost
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_encoded)
X_test_scaled = scaler.transform(X_test_encoded)

X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_encoded.columns)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_encoded.columns)

# ============================================================================
# 7. HYPERPARAMETER TUNING
# ============================================================================
print("\n" + "="*70)
print("HYPERPARAMETER TUNING")
print("="*70)

def tune_gradient_boosting(X_train, y_train):
    """Tune Gradient Boosting hyperparameters"""
    print("\n" + "-"*40)
    print("TUNING GRADIENT BOOSTING")
    print("-"*40)
    
    param_dist = {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 4, 5],
        'min_samples_split': [2, 5]
    }
    
    gb = GradientBoostingRegressor(random_state=42)
    gb_search = RandomizedSearchCV(
        gb, param_dist, n_iter=10, cv=3, 
        scoring='r2', n_jobs=-1, random_state=42, verbose=0
    )
    
    gb_search.fit(X_train, y_train)
    print(f"Best CV R²: {gb_search.best_score_:.4f}")
    return gb_search.best_estimator_, gb_search.best_params_

def tune_catboost(X_train, y_train, cat_features_indices=None):
    """Tune CatBoost hyperparameters"""
    print("\n" + "-"*40)
    print("TUNING CATBOOST")
    print("-"*40)
    
    # Prepare CatBoost data
    X_train_cb = X_train.copy()
    
    # Ensure categorical columns are strings
    for col in categorical_cols:
        if col in X_train_cb.columns:
            X_train_cb[col] = X_train_cb[col].astype(str)
    
    param_dist = {
        'iterations': [100, 200],
        'learning_rate': [0.01, 0.05, 0.1],
        'depth': [4, 6],
        'l2_leaf_reg': [1, 3]
    }
    
    cb = CatBoostRegressor(random_seed=42, verbose=0)
    cb_search = RandomizedSearchCV(
        cb, param_dist, n_iter=8, cv=3, 
        scoring='r2', n_jobs=-1, random_state=42, verbose=0
    )
    
    if categorical_cols:
        cat_indices = [list(X_train_cb.columns).index(col) for col in categorical_cols if col in X_train_cb.columns]
        print(f"Using {len(cat_indices)} categorical features")
        cb_search.fit(X_train_cb, y_train, cat_features=cat_indices)
    else:
        cb_search.fit(X_train_cb, y_train)
    
    print(f"Best CV R²: {cb_search.best_score_:.4f}")
    return cb_search.best_estimator_, cb_search.best_params_

def tune_xgboost(X_train, y_train):
    """Tune XGBoost hyperparameters"""
    print("\n" + "-"*40)
    print("TUNING XGBOOST")
    print("-"*40)
    
    param_dist = {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 4, 5],
        'subsample': [0.8, 0.9]
    }
    
    xgb = XGBRegressor(random_state=42)
    xgb_search = RandomizedSearchCV(
        xgb, param_dist, n_iter=10, cv=3, 
        scoring='r2', n_jobs=-1, random_state=42, verbose=0
    )
    
    xgb_search.fit(X_train, y_train)
    print(f"Best CV R²: {xgb_search.best_score_:.4f}")
    return xgb_search.best_estimator_, xgb_search.best_params_

# Perform hyperparameter tuning
print("\nStarting hyperparameter tuning...")

gb_best_model, gb_best_params = tune_gradient_boosting(X_train_scaled_df, y_train)

# For CatBoost, use the unencoded data
cb_best_model, cb_best_params = tune_catboost(X_train_catboost, y_train)

xgb_best_model, xgb_best_params = tune_xgboost(X_train_scaled_df, y_train)

# ============================================================================
# 8. TRAIN MODELS
# ============================================================================
print("\n" + "="*70)
print("TRAINING MODELS")
print("="*70)

# Create models with best parameters
gb_model = GradientBoostingRegressor(**gb_best_params, random_state=42)
cb_model = CatBoostRegressor(**cb_best_params, random_seed=42, verbose=0)
xgb_model = XGBRegressor(**xgb_best_params, random_state=42)

# Train models
print("\nTraining Gradient Boosting...")
gb_model.fit(X_train_scaled_df, y_train)
gb_pred = gb_model.predict(X_test_scaled_df)
gb_r2 = r2_score(y_test, gb_pred)
print(f"  Test R²: {gb_r2:.4f}")

print("\nTraining CatBoost...")
# Prepare CatBoost test data
X_test_cb = X_test_catboost.copy()
for col in categorical_cols:
    if col in X_test_cb.columns:
        X_test_cb[col] = X_test_cb[col].astype(str)

if categorical_cols:
    cat_indices = [list(X_train_catboost.columns).index(col) for col in categorical_cols if col in X_train_catboost.columns]
    cb_model.fit(X_train_catboost, y_train, cat_features=cat_indices, verbose=0)
else:
    cb_model.fit(X_train_catboost, y_train, verbose=0)

cb_pred = cb_model.predict(X_test_cb)
cb_r2 = r2_score(y_test, cb_pred)
print(f"  Test R²: {cb_r2:.4f}")

print("\nTraining XGBoost...")
xgb_model.fit(X_train_scaled_df, y_train)
xgb_pred = xgb_model.predict(X_test_scaled_df)
xgb_r2 = r2_score(y_test, xgb_pred)
print(f"  Test R²: {xgb_r2:.4f}")

# ============================================================================
# 9. CREATE STACKING ENSEMBLE
# ============================================================================
print("\n" + "="*70)
print("CREATING STACKING ENSEMBLE")
print("="*70)

# Create stacking ensemble
estimators = [
    ('gradient_boosting', gb_model),
    ('catboost', cb_model),
    ('xgboost', xgb_model)
]

stacking_model = StackingRegressor(
    estimators=estimators,
    final_estimator=Ridge(alpha=1.0),
    cv=5,
    n_jobs=-1
)

print("Training Stacking Ensemble...")
stacking_model.fit(X_train_scaled_df, y_train)
stack_pred = stacking_model.predict(X_test_scaled_df)
stack_r2 = r2_score(y_test, stack_pred)
print(f"  Test R²: {stack_r2:.4f}")

# ============================================================================
# 10. CREATE SCATTER PLOTS
# ============================================================================
print("\n" + "="*70)
print("CREATING SCATTER PLOTS")
print("="*70)

try:
    import matplotlib.pyplot as plt
    
    # Set style
    plt.style.use('default')
    
    # Collect all predictions
    predictions = {
        'Gradient Boosting': gb_pred,
        'CatBoost': cb_pred,
        'XGBoost': xgb_pred,
        'Stacking Ensemble': stack_pred
    }
    
    # Create individual scatter plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    
    colors = ['blue', 'green', 'orange', 'red']
    
    for idx, (model_name, y_pred) in enumerate(predictions.items()):
        ax = axes[idx]
        
        # Scatter plot
        scatter = ax.scatter(y_test, y_pred, alpha=0.6, s=50, 
                           color=colors[idx], edgecolors='k', linewidth=0.5)
        
        # Perfect prediction line
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 
               'r--', linewidth=2, label='Perfect Prediction')
        
        # Calculate metrics
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        
        # Add metrics to plot
        text_str = f'R² = {r2:.4f}\nRMSE = {rmse:.4f}\nMAE = {mae:.4f}'
        ax.text(0.05, 0.95, text_str, transform=ax.transAxes,
               fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Formatting
        ax.set_xlabel('Actual Inhibition Size (mm)', fontsize=11)
        ax.set_ylabel('Predicted Inhibition Size (mm)', fontsize=11)
        ax.set_title(f'{model_name}', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='lower right')
    
    plt.tight_layout()
    plt.savefig('scatter_plots_individual.png', dpi=300, bbox_inches='tight')
    print("✓ Individual scatter plots saved as 'scatter_plots_individual.png'")
    plt.show()
    
    # ========================================================================
    # CREATE COMBINED SCATTER PLOT
    # ========================================================================
    print("\nCreating combined scatter plot...")
    
    fig2, ax2 = plt.subplots(figsize=(12, 10))
    
    markers = ['o', 's', '^', 'D']
    
    for idx, (model_name, y_pred) in enumerate(predictions.items()):
        r2 = r2_score(y_test, y_pred)
        ax2.scatter(y_test, y_pred, alpha=0.5, s=50,
                   color=colors[idx], marker=markers[idx],
                   edgecolors='k', linewidth=0.5,
                   label=f'{model_name} (R²={r2:.3f})')
    
    # Perfect prediction line
    all_preds = list(predictions.values())
    min_val = min(y_test.min(), min([p.min() for p in all_preds]))
    max_val = max(y_test.max(), max([p.max() for p in all_preds]))
    ax2.plot([min_val, max_val], [min_val, max_val], 
            'r--', linewidth=3, label='Perfect Prediction')
    
    # Formatting
    ax2.set_xlabel('Actual Inhibition Size (mm)', fontsize=13)
    ax2.set_ylabel('Predicted Inhibition Size (mm)', fontsize=13)
    ax2.set_title('All Models: Actual vs Predicted Inhibition Size', 
                 fontsize=15, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper left')
    
    plt.tight_layout()
    plt.savefig('scatter_plot_combined.png', dpi=300, bbox_inches='tight')
    print("✓ Combined scatter plot saved as 'scatter_plot_combined.png'")
    plt.show()
    
    # ========================================================================
    # CREATE PERFORMANCE COMPARISON BAR CHART
    # ========================================================================
    print("\nCreating performance comparison chart...")
    
    # Calculate all metrics
    models_metrics = []
    for model_name, y_pred in predictions.items():
        models_metrics.append({
            'Model': model_name,
            'R2_Score': r2_score(y_test, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
            'MAE': mean_absolute_error(y_test, y_pred)
        })
    
    # Create bar chart
    fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(15, 6))
    
    # R² comparison
    model_names = [m['Model'] for m in models_metrics]
    r2_scores = [m['R2_Score'] for m in models_metrics]
    
    bars1 = ax3a.bar(model_names, r2_scores, color=colors[:len(model_names)])
    ax3a.set_ylabel('R² Score', fontsize=12)
    ax3a.set_title('Model Performance (R² Score)', fontsize=14, fontweight='bold')
    ax3a.set_ylim([0, max(1.0, max(r2_scores) + 0.1)])
    ax3a.grid(True, alpha=0.3, axis='y')
    ax3a.tick_params(axis='x', rotation=45)
    
    # Add value labels
    for bar, score in zip(bars1, r2_scores):
        height = bar.get_height()
        ax3a.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f'{score:.3f}', ha='center', va='bottom', fontsize=10)
    
    # RMSE comparison
    rmse_scores = [m['RMSE'] for m in models_metrics]
    
    bars2 = ax3b.bar(model_names, rmse_scores, color=colors[:len(model_names)])
    ax3b.set_ylabel('RMSE (mm)', fontsize=12)
    ax3b.set_title('Model Performance (RMSE)', fontsize=14, fontweight='bold')
    ax3b.grid(True, alpha=0.3, axis='y')
    ax3b.tick_params(axis='x', rotation=45)
    
    # Add value labels
    for bar, score in zip(bars2, rmse_scores):
        height = bar.get_height()
        ax3b.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f'{score:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('performance_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Performance comparison saved as 'performance_comparison.png'")
    plt.show()
    
except ImportError as e:
    print(f"\nMatplotlib import error: {e}")
    print("Install matplotlib with: pip install matplotlib")

# ============================================================================
# 11. SAVE RESULTS
# ============================================================================
print("\n" + "="*70)
print("SAVING RESULTS")
print("="*70)

# Save predictions
results_df = pd.DataFrame({
    'Actual': y_test.values,
    'Gradient_Boosting_Pred': gb_pred,
    'CatBoost_Pred': cb_pred,
    'XGBoost_Pred': xgb_pred,
    'Stacking_Ensemble_Pred': stack_pred
})

results_df.to_csv('model_predictions.csv', index=False)
print("✓ Predictions saved to 'model_predictions.csv'")

# Save metrics
import json
metrics_data = []
for model_name, y_pred in predictions.items():
    metrics_data.append({
        'Model': model_name,
        'R2_Score': float(r2_score(y_test, y_pred)),
        'RMSE': float(np.sqrt(mean_squared_error(y_test, y_pred))),
        'MAE': float(mean_absolute_error(y_test, y_pred))
    })

with open('model_metrics.json', 'w') as f:
    json.dump(metrics_data, f, indent=4)
print("✓ Metrics saved to 'model_metrics.json'")

# Save best parameters
best_params = {
    'GradientBoosting': gb_best_params,
    'CatBoost': cb_best_params,
    'XGBoost': xgb_best_params
}

with open('best_parameters.json', 'w') as f:
    json.dump(best_params, f, indent=4)
print("✓ Best parameters saved to 'best_parameters.json'")

# ============================================================================
# 12. PRINT SUMMARY
# ============================================================================
print("\n" + "="*70)
print("MODEL PERFORMANCE SUMMARY")
print("="*70)

print(f"\n{'Model':<20} {'R² Score':<12} {'RMSE':<12} {'MAE':<12}")
print("-"*60)

for metrics in metrics_data:
    print(f"{metrics['Model']:<20} {metrics['R2_Score']:<12.4f} {metrics['RMSE']:<12.4f} {metrics['MAE']:<12.4f}")

print("-"*60)

# Find best model
best_model = max(metrics_data, key=lambda x: x['R2_Score'])
print(f"\n✨ BEST MODEL: {best_model['Model']}")
print(f"   R² Score: {best_model['R2_Score']:.4f}")
print(f"   RMSE: {best_model['RMSE']:.4f} mm")
print(f"   MAE: {best_model['MAE']:.4f} mm")

# ============================================================================
# 13. FINAL SUMMARY
# ============================================================================
print("\n" + "="*70)
print("PROCESSING COMPLETE")
print("="*70)

print(f"\nDataset: {df.shape}")
print(f"Target column: '{target_column}'")
print(f"Features used: {X_encoded.shape[1]}")
print(f"Numeric columns: {len(numeric_cols)}")
print(f"Categorical columns: {len(categorical_cols)}")
print(f"Text columns dropped: {len(text_cols)}")
print(f"Training samples: {X_train_encoded.shape[0]}")
print(f"Test samples: {X_test_encoded.shape[0]}")

print(f"\nFiles created:")
print("  1. scatter_plots_individual.png")
print("  2. scatter_plot_combined.png")
print("  3. performance_comparison.png")
print("  4. model_predictions.csv")
print("  5. model_metrics.json")
print("  6. best_parameters.json")

print("\n" + "="*70)
print("SCATTER PLOTS CREATED SUCCESSFULLY!")
print("="*70)