# -*- coding: utf-8 -*-
"""
Created on Thu Dec 25 09:32:09 2025

@author: Amare
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor, StackingRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
import warnings
warnings.filterwarnings('ignore')

# Set style for better looking plots
plt.style.use('default')
sns.set_style("whitegrid")

# Color scheme for models
model_colors = {
    'CatBoost': '#FF6B6B',      # Coral Red
    'XGBoost': '#4ECDC4',       # Turquoise
    'GradientBoosting': '#45B7D1',  # Sky Blue
    'Stacking': '#96CEB4'       # Mint Green
}

# Marker styles for scatter plots
marker_styles = {
    'CatBoost': 'o',
    'XGBoost': 's',
    'GradientBoosting': '^',
    'Stacking': 'D'
}

# ---------------------------
# Load CSV - UPDATED PATH
# ---------------------------
print("📊" + "="*60)
print("📊 LOADING AND PREPROCESSING DATA")
print("📊" + "="*60)

# Your specific file path
file_path = r"C:\Users\Amare\Desktop\worku\datato232-8.csv"

try:
    # Try different encodings to handle potential encoding issues
    try:
        df = pd.read_csv(file_path, encoding="utf-8")
    except:
        try:
            df = pd.read_csv(file_path, encoding="cp1252")
        except:
            df = pd.read_csv(file_path, encoding="latin1")
    
    print(f"✓ Successfully loaded data from: {file_path}")
    print(f"Dataset Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Display first few rows to understand the data structure
    print("\nFirst 5 rows of the dataset:")
    print(df.head())
    
    # Display column info
    print("\nDataset Information:")
    print(df.info())
    
except FileNotFoundError:
    print(f"✗ Error: File not found at {file_path}")
    print("Please check the file path and try again.")
    exit()
except Exception as e:
    print(f"✗ Error loading file: {str(e)}")
    exit()

# Clean column names
df.columns = df.columns.str.strip()
print(f"\nCleaned column names: {list(df.columns)}")

# Check if we have the expected target column
target = 'Inhibition size mm'
if target not in df.columns:
    print(f"\n⚠ Warning: Target column '{target}' not found in dataset.")
    print(f"Available columns: {list(df.columns)}")
    
    # Try to find similar column names
    possible_targets = [col for col in df.columns if 'inhibit' in col.lower() or 'size' in col.lower()]
    if possible_targets:
        print(f"Possible target columns: {possible_targets}")
        target = possible_targets[0]
        print(f"Using '{target}' as target variable")
    else:
        print("Please check your dataset and specify the correct target column.")
        exit()

# Define features - adjust based on your actual data
# First, let's see what columns are available
all_columns = df.columns.tolist()
print(f"\nAll available columns ({len(all_columns)}): {all_columns}")

# Remove target from features list
potential_features = [col for col in all_columns if col != target]
print(f"\nPotential features ({len(potential_features)}): {potential_features}")

# Use all available columns except target as features
features = potential_features

print(f"\nUsing {len(features)} features:")
for i, feature in enumerate(features, 1):
    print(f"  {i}. {feature}")

# Keep only necessary columns and drop rows with missing values
print(f"\nData cleaning...")
initial_rows = len(df)
df = df[features + [target]].dropna()
cleaned_rows = len(df)
print(f"Removed {initial_rows - cleaned_rows} rows with missing values")
print(f"Cleaned Dataset Shape: {df.shape}")
print(f"Target variable: '{target}'")

# Check target variable statistics
print(f"\nTarget Statistics:")
print(f"  Mean: {df[target].mean():.4f}")
print(f"  Std: {df[target].std():.4f}")
print(f"  Min: {df[target].min():.4f}")
print(f"  Max: {df[target].max():.4f}")
print(f"  Missing values: {df[target].isnull().sum()}")

# Split X and y
X = df[features]
y = df[target]

# Identify categorical and numeric columns
categorical_cols = X.select_dtypes(include='object').columns.tolist()
numeric_cols = X.select_dtypes(exclude='object').columns.tolist()

print(f"\nData Types Analysis:")
print(f"  Categorical features ({len(categorical_cols)}): {categorical_cols}")
print(f"  Numerical features ({len(numeric_cols)}): {numeric_cols}")

# Display some statistics about categorical columns
if categorical_cols:
    print(f"\nCategorical columns information:")
    for col in categorical_cols:
        unique_vals = X[col].nunique()
        print(f"  {col}: {unique_vals} unique values")
        if unique_vals <= 10:  # Show values if not too many
            print(f"    Values: {X[col].unique().tolist()}")

# Display statistics about numeric columns
if numeric_cols:
    print(f"\nNumerical columns statistics:")
    for col in numeric_cols:
        print(f"  {col}: Mean={X[col].mean():.4f}, Std={X[col].std():.4f}, Min={X[col].min():.4f}, Max={X[col].max():.4f}")

# ---------------------------
# Preprocessor for numeric + categorical encoding
# ---------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("num", "passthrough", numeric_cols)
    ]
)

# ---------------------------
# Train/test split
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nData Splitting:")
print(f"  Training set: {X_train.shape[0]} samples")
print(f"  Testing set: {X_test.shape[0]} samples")

# Display target distribution in train and test
print(f"\nTarget distribution:")
print(f"  Training - Mean: {y_train.mean():.4f}, Std: {y_train.std():.4f}")
print(f"  Testing  - Mean: {y_test.mean():.4f}, Std: {y_test.std():.4f}")

# ---------------------------
# HYPERPARAMETER TUNING FUNCTIONS
# ---------------------------
print("\n" + "="*60)
print("🔧 HYPERPARAMETER TUNING")
print("="*60)

def tune_catboost(X_train, y_train, categorical_features):
    """Perform hyperparameter tuning for CatBoost"""
    print("\nTuning CatBoost hyperparameters...")
    
    # Define parameter grid for CatBoost
    param_grid = {
        'depth': [4, 6, 8, 10],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'iterations': [200, 300, 400, 500],
        'l2_leaf_reg': [1, 3, 5, 7],
        'border_count': [32, 64, 128]
    }
    
    # Create base model
    cat_model = CatBoostRegressor(
        verbose=0,
        random_state=42,
        cat_features=categorical_features
    )
    
    # Use RandomizedSearchCV for faster tuning
    cat_search = RandomizedSearchCV(
        cat_model,
        param_grid,
        n_iter=20,  # Number of parameter settings sampled
        cv=5,
        scoring='r2',
        n_jobs=-1,
        random_state=42,
        verbose=0
    )
    
    cat_search.fit(X_train, y_train)
    
    print(f"  Best parameters: {cat_search.best_params_}")
    print(f"  Best CV R²: {cat_search.best_score_:.4f}")
    
    return cat_search.best_estimator_, cat_search.best_params_

def tune_xgboost(X_train, y_train):
    """Perform hyperparameter tuning for XGBoost"""
    print("\nTuning XGBoost hyperparameters...")
    
    # Define parameter grid for XGBoost
    param_grid = {
        'model__n_estimators': [200, 300, 400, 500],
        'model__learning_rate': [0.01, 0.05, 0.1, 0.2],
        'model__max_depth': [3, 4, 5, 6, 7],
        'model__subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
        'model__colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
        'model__gamma': [0, 0.1, 0.2, 0.3, 0.4],
        'model__reg_alpha': [0, 0.01, 0.1, 1, 10],
        'model__reg_lambda': [0.1, 1, 10, 100]
    }
    
    # Create pipeline for XGBoost
    xgb_pipeline = Pipeline([
        ("prep", preprocessor),
        ("model", XGBRegressor(random_state=42))
    ])
    
    # Use RandomizedSearchCV
    xgb_search = RandomizedSearchCV(
        xgb_pipeline,
        param_grid,
        n_iter=25,
        cv=5,
        scoring='r2',
        n_jobs=-1,
        random_state=42,
        verbose=0
    )
    
    xgb_search.fit(X_train, y_train)
    
    print(f"  Best parameters: {xgb_search.best_params_}")
    print(f"  Best CV R²: {xgb_search.best_score_:.4f}")
    
    return xgb_search.best_estimator_, xgb_search.best_params_

def tune_gradient_boosting(X_train, y_train):
    """Perform hyperparameter tuning for Gradient Boosting"""
    print("\nTuning Gradient Boosting hyperparameters...")
    
    # Define parameter grid for Gradient Boosting
    param_grid = {
        'model__n_estimators': [200, 300, 400, 500],
        'model__learning_rate': [0.01, 0.05, 0.1, 0.2],
        'model__max_depth': [3, 4, 5, 6, 7],
        'model__min_samples_split': [2, 5, 10],
        'model__min_samples_leaf': [1, 2, 4],
        'model__subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
        'model__max_features': ['auto', 'sqrt', 'log2']
    }
    
    # Create pipeline for Gradient Boosting
    gbr_pipeline = Pipeline([
        ("prep", preprocessor),
        ("model", GradientBoostingRegressor(random_state=42))
    ])
    
    # Use RandomizedSearchCV
    gbr_search = RandomizedSearchCV(
        gbr_pipeline,
        param_grid,
        n_iter=20,
        cv=5,
        scoring='r2',
        n_jobs=-1,
        random_state=42,
        verbose=0
    )
    
    gbr_search.fit(X_train, y_train)
    
    print(f"  Best parameters: {gbr_search.best_params_}")
    print(f"  Best CV R²: {gbr_search.best_score_:.4f}")
    
    return gbr_search.best_estimator_, gbr_search.best_params_

# Perform hyperparameter tuning for all models
print("\n" + "="*50)
print("Starting Hyperparameter Tuning...")
print("="*50)

# Tune CatBoost (only if we have categorical columns)
if categorical_cols:
    best_cat_model, best_cat_params = tune_catboost(X_train, y_train, categorical_cols)
else:
    print("\nNo categorical columns found. Skipping CatBoost tuning.")
    # Create a simple CatBoost model without categorical features
    cat_model = CatBoostRegressor(verbose=0, random_state=42)
    cat_model.fit(X_train, y_train)
    best_cat_model = cat_model
    best_cat_params = {"Note": "No categorical features"}

# Tune XGBoost
best_xgb_model, best_xgb_params = tune_xgboost(X_train, y_train)

# Tune Gradient Boosting
best_gbr_model, best_gbr_params = tune_gradient_boosting(X_train, y_train)

# Store best parameters for display
best_params_summary = {
    'CatBoost': best_cat_params,
    'XGBoost': best_xgb_params,
    'GradientBoosting': best_gbr_params
}

print("\n" + "="*50)
print("Hyperparameter Tuning Complete!")
print("="*50)

# ---------------------------
# CREATE TUNED MODELS
# ---------------------------
print("\n" + "="*60)
print("🤖 CREATING TUNED MODELS")
print("="*60)

# Create stacking model with tuned base models
stack_model = StackingRegressor(
    estimators=[
        ("cat", best_cat_model),
        ("xgb", best_xgb_model),
        ("gbr", best_gbr_model)
    ],
    final_estimator=GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )
)

models = {
    "CatBoost": best_cat_model,
    "XGBoost": best_xgb_model,
    "GradientBoosting": best_gbr_model,
    "Stacking": stack_model
}

print("\nModels created with tuned hyperparameters!")

# ---------------------------
# Train, predict, evaluate TUNED MODELS
# ---------------------------
print("\n" + "="*60)
print("🚀 TRAINING TUNED MODELS")
print("="*60)

results = []
all_predictions = {}
all_importances = {}
training_times = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    
    # Train model
    model.fit(X_train, y_train)
    
    # Make predictions
    preds = model.predict(X_test)
    all_predictions[name] = preds

    # Calculate all metrics
    r2 = r2_score(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    
    # Store results
    results.append([name, r2, rmse, mae])
    
    print(f"  R² Score: {r2:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE: {mae:.4f}")

    # COLORFUL Scatter Plot: Predicted vs Actual
    plt.figure(figsize=(8, 7))
    
    # Create gradient color based on error magnitude
    errors = np.abs(preds - y_test)
    
    # Use viridis colormap for error visualization
    scatter = plt.scatter(y_test, preds, 
                         c=errors, 
                         cmap='viridis', 
                         alpha=0.7, 
                         s=80,
                         edgecolors='black',
                         linewidth=0.5)
    
    # Perfect prediction line
    min_val = min(y_test.min(), preds.min())
    max_val = max(y_test.max(), preds.max())
    plt.plot([min_val, max_val], [min_val, max_val], 
            'r--', linewidth=2, alpha=0.8, label='Perfect Prediction')
    
    # Add metrics as text box
    textstr = f'R² = {r2:.4f}\nRMSE = {rmse:.4f}\nMAE = {mae:.4f}'
    props = dict(boxstyle='round', facecolor=model_colors[name], alpha=0.8)
    plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, 
            fontsize=12, verticalalignment='top', bbox=props)
    
    plt.xlabel("Actual Inhibition Size (mm)", fontsize=12, fontweight='bold')
    plt.ylabel("Predicted Inhibition Size (mm)", fontsize=12, fontweight='bold')
    plt.title(f"{name} (Tuned): Predicted vs Actual", fontsize=14, fontweight='bold', pad=15)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='lower right')
    
    # Add colorbar for error
    cbar = plt.colorbar(scatter)
    cbar.set_label('Absolute Error', fontsize=11)
    
    plt.tight_layout()
    plt.show()

    # Feature importance (except stacking)
    if name != "Stacking":
        try:
            if name == "CatBoost":
                importances = model.get_feature_importance()
                feature_names = features
            elif name == "XGBoost":
                # For pipeline models
                if hasattr(model, 'named_steps'):
                    importances = model.named_steps["model"].feature_importances_
                else:
                    importances = model.feature_importances_
                feature_names = features
            else:  # GradientBoosting
                # For pipeline models
                if hasattr(model, 'named_steps'):
                    importances = model.named_steps["model"].feature_importances_
                else:
                    importances = model.feature_importances_
                feature_names = features
            
            # Store for later comparison
            all_importances[name] = importances
            
            # Create colorful feature importance plot
            plt.figure(figsize=(10, 6))
            
            # Sort by importance
            sorted_idx = np.argsort(importances)
            sorted_importances = importances[sorted_idx]
            sorted_features = [feature_names[i] for i in sorted_idx]
            
            # Create colorful bars
            colors = plt.cm.coolwarm(np.linspace(0.2, 0.8, len(sorted_importances)))
            
            bars = plt.barh(range(len(sorted_importances)), sorted_importances, color=colors)
            
            plt.yticks(range(len(sorted_importances)), sorted_features, fontsize=10)
            plt.xlabel("Importance Score", fontsize=12, fontweight='bold')
            plt.title(f"{name} (Tuned): Feature Importance", fontsize=14, fontweight='bold', pad=15)
            plt.grid(True, alpha=0.3, axis='x')
            
            # Add value labels on significant bars
            for i, (bar, val) in enumerate(zip(bars, sorted_importances)):
                width = bar.get_width()
                if width > 0.001:  # Only label significant bars
                    plt.text(width + 0.001, bar.get_y() + bar.get_height()/2, 
                            f'{val:.3f}', va='center', ha='left', fontsize=9)
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"Feature importance not available for {name}: {str(e)[:50]}")

# ---------------------------
# DISPLAY BEST HYPERPARAMETERS
# ---------------------------
print("\n" + "="*70)
print("⚙️  BEST HYPERPARAMETERS FROM TUNING")
print("="*70)

for model_name, params in best_params_summary.items():
    print(f"\n{model_name}:")
    print("-" * 40)
    if isinstance(params, dict):
        for param, value in params.items():
            # Clean up parameter names for display
            clean_param = param.replace('model__', '')
            print(f"  {clean_param}: {value}")
    else:
        print(f"  {params}")

# ---------------------------
# COMPREHENSIVE RESULTS TABLE
# ---------------------------
print("\n" + "="*70)
print("📊 COMPREHENSIVE MODEL PERFORMANCE METRICS (TUNED)")
print("="*70)

# Create results DataFrame
results_df = pd.DataFrame(results, columns=["Model", "R² Score", "RMSE", "MAE"])

# Format the table for better display
styled_df = results_df.copy()
styled_df["R² Score"] = styled_df["R² Score"].apply(lambda x: f"{x:.4f}")
styled_df["RMSE"] = styled_df["RMSE"].apply(lambda x: f"{x:.4f}")
styled_df["MAE"] = styled_df["MAE"].apply(lambda x: f"{x:.4f}")

# Add ranking
results_df_sorted = results_df.sort_values("R² Score", ascending=False)
results_df_sorted["Rank"] = range(1, len(results_df_sorted) + 1)
results_df_sorted["Improvement"] = "✓"  # Mark as tuned

print("\n" + "-"*70)
print(f"{'Model':<20} {'R² Score':<12} {'RMSE':<12} {'MAE':<12} {'Rank':<6} {'Tuned':<8}")
print("-"*70)

for idx, row in results_df_sorted.iterrows():
    print(f"{row['Model']:<20} {row['R² Score']:<12.4f} {row['RMSE']:<12.4f} {row['MAE']:<12.4f} {row['Rank']:<6} {'✓':<8}")

print("-"*70)

# Display formatted table
print("\n📋 Performance Metrics Table (with Hyperparameter Tuning):")
print("="*65)
print(styled_df.to_string(index=False))
print("="*65)

# Find and display best model
best_model_idx = results_df["R² Score"].idxmax()
best_model = results_df.loc[best_model_idx, "Model"]
best_r2 = results_df.loc[best_model_idx, "R² Score"]
best_rmse = results_df.loc[best_model_idx, "RMSE"]

print(f"\n🏆 BEST PERFORMING MODEL (After Tuning): {best_model}")
print(f"   R² Score: {best_r2:.4f}")
print(f"   RMSE: {best_rmse:.4f} mm")
print(f"   MAE: {results_df.loc[best_model_idx, 'MAE']:.4f} mm")

# ---------------------------
# MODEL COMPARISON VISUALIZATION
# ---------------------------
print("\n" + "="*60)
print("📈 MODEL COMPARISON VISUALIZATION (TUNED MODELS)")
print("="*60)

# Create comparison figure
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Model Performance Comparison (After Hyperparameter Tuning)', 
             fontsize=16, fontweight='bold', y=1.05)

# Bar colors for each model
bar_colors = [model_colors[model] for model in results_df["Model"]]

# 1. R² Score Comparison
axes[0].bar(results_df["Model"], results_df["R² Score"], color=bar_colors, 
           edgecolor='black', linewidth=1.5)
axes[0].set_title('R² Score Comparison', fontsize=14, fontweight='bold')
axes[0].set_ylabel('R² Score', fontsize=12)
axes[0].set_ylim([0, 1.05])
axes[0].grid(True, alpha=0.3, axis='y')
axes[0].tick_params(axis='x', rotation=45)

# Add value labels
for i, (model, r2) in enumerate(zip(results_df["Model"], results_df["R² Score"])):
    axes[0].text(i, r2 + 0.02, f'{r2:.3f}', ha='center', va='bottom', 
                fontweight='bold', fontsize=11)

# 2. RMSE Comparison
axes[1].bar(results_df["Model"], results_df["RMSE"], color=bar_colors, 
           edgecolor='black', linewidth=1.5)
axes[1].set_title('RMSE Comparison (Lower is Better)', fontsize=14, fontweight='bold')
axes[1].set_ylabel('RMSE (mm)', fontsize=12)
axes[1].grid(True, alpha=0.3, axis='y')
axes[1].tick_params(axis='x', rotation=45)

# Add value labels
for i, (model, rmse) in enumerate(zip(results_df["Model"], results_df["RMSE"])):
    axes[1].text(i, rmse + 0.02, f'{rmse:.3f}', ha='center', va='bottom', 
                fontweight='bold', fontsize=11)

# 3. MAE Comparison
axes[2].bar(results_df["Model"], results_df["MAE"], color=bar_colors, 
           edgecolor='black', linewidth=1.5)
axes[2].set_title('MAE Comparison (Lower is Better)', fontsize=14, fontweight='bold')
axes[2].set_ylabel('MAE (mm)', fontsize=12)
axes[2].grid(True, alpha=0.3, axis='y')
axes[2].tick_params(axis='x', rotation=45)

# Add value labels
for i, (model, mae) in enumerate(zip(results_df["Model"], results_df["MAE"])):
    axes[2].text(i, mae + 0.02, f'{mae:.3f}', ha='center', va='bottom', 
                fontweight='bold', fontsize=11)

plt.tight_layout()
plt.show()

# ---------------------------
# COMBINED SCATTER PLOT
# ---------------------------
print("\n" + "="*60)
print("🎯 ALL TUNED MODELS: PREDICTED VS ACTUAL")
print("="*60)

plt.figure(figsize=(10, 8))

for name, preds in all_predictions.items():
    plt.scatter(y_test, preds, 
               color=model_colors[name], 
               marker=marker_styles[name],
               alpha=0.6, 
               s=60,
               edgecolors='black',
               linewidth=0.5,
               label=f'{name} (R²={results_df[results_df["Model"]==name]["R² Score"].values[0]:.3f})')

# Perfect prediction line
all_pred_values = np.concatenate(list(all_predictions.values()))
min_val = min(y_test.min(), all_pred_values.min())
max_val = max(y_test.max(), all_pred_values.max())
plt.plot([min_val, max_val], [min_val, max_val], 
        'k--', linewidth=2, alpha=0.7, label='Perfect Prediction')

plt.xlabel("Actual Inhibition Size (mm)", fontsize=13, fontweight='bold')
plt.ylabel("Predicted Inhibition Size (mm)", fontsize=13, fontweight='bold')
plt.title("All Tuned Models: Predicted vs Actual Inhibition Size", 
         fontsize=15, fontweight='bold', pad=15)
plt.grid(True, alpha=0.3)
plt.legend(loc='upper left', fontsize=11)

# Add overall statistics
avg_r2 = results_df["R² Score"].mean()
avg_rmse = results_df["RMSE"].mean()
plt.text(0.05, 0.95, f'Average R²: {avg_r2:.3f}\nAverage RMSE: {avg_rmse:.3f}', 
         transform=plt.gca().transAxes,
         fontsize=11,
         verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.show()

# ---------------------------
# HYPERPARAMETER IMPORTANCE VISUALIZATION
# ---------------------------
print("\n" + "="*60)
print("🔧 HYPERPARAMETER IMPORTANCE SUMMARY")
print("="*60)

# Create a visualization of important hyperparameters
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Key Hyperparameters After Tuning', fontsize=16, fontweight='bold', y=1.05)

# CatBoost parameters
if 'CatBoost' in best_params_summary and isinstance(best_params_summary['CatBoost'], dict):
    cat_params = best_params_summary['CatBoost']
    important_cat_params = {k: v for k, v in cat_params.items() 
                           if k in ['depth', 'learning_rate', 'iterations', 'l2_leaf_reg']}
    
    if important_cat_params:
        axes[0].barh(list(important_cat_params.keys()), list(important_cat_params.values()), 
                    color=model_colors['CatBoost'])
        axes[0].set_title('CatBoost - Key Parameters', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Value')
        axes[0].grid(True, alpha=0.3, axis='x')
    else:
        axes[0].text(0.5, 0.5, 'No categorical\nfeatures', 
                    ha='center', va='center', transform=axes[0].transAxes)
        axes[0].set_title('CatBoost - Key Parameters', fontsize=14, fontweight='bold')

# XGBoost parameters
if 'XGBoost' in best_params_summary:
    xgb_params = best_params_summary['XGBoost']
    # Clean parameter names
    clean_xgb_params = {}
    for k, v in xgb_params.items():
        clean_name = k.replace('model__', '')
        if clean_name in ['n_estimators', 'learning_rate', 'max_depth', 'subsample']:
            clean_xgb_params[clean_name] = v
    
    if clean_xgb_params:
        axes[1].barh(list(clean_xgb_params.keys()), list(clean_xgb_params.values()), 
                    color=model_colors['XGBoost'])
        axes[1].set_title('XGBoost - Key Parameters', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Value')
        axes[1].grid(True, alpha=0.3, axis='x')

# Gradient Boosting parameters
if 'GradientBoosting' in best_params_summary:
    gbr_params = best_params_summary['GradientBoosting']
    # Clean parameter names
    clean_gbr_params = {}
    for k, v in gbr_params.items():
        clean_name = k.replace('model__', '')
        if clean_name in ['n_estimators', 'learning_rate', 'max_depth', 'subsample']:
            clean_gbr_params[clean_name] = v
    
    if clean_gbr_params:
        axes[2].barh(list(clean_gbr_params.keys()), list(clean_gbr_params.values()), 
                    color=model_colors['GradientBoosting'])
        axes[2].set_title('Gradient Boosting - Key Parameters', fontsize=14, fontweight='bold')
        axes[2].set_xlabel('Value')
        axes[2].grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.show()

# ---------------------------
# FINAL SUMMARY
# ---------------------------
print("\n" + "="*80)
print("✅ ANALYSIS COMPLETE - SUMMARY WITH HYPERPARAMETER TUNING")
print("="*80)

print(f"\n📊 Dataset Information:")
print(f"   Original dataset: {df.shape}")
print(f"   Training samples: {X_train.shape[0]}")
print(f"   Testing samples: {X_test.shape[0]}")
print(f"   Features used: {len(features)}")

print(f"\n🎯 Target Statistics (Inhibition Size):")
print(f"   Mean: {y.mean():.4f} mm")
print(f"   Std: {y.std():.4f} mm")
print(f"   Min: {y.min():.4f} mm")
print(f"   Max: {y.max():.4f} mm")

print(f"\n⚙️  Hyperparameter Tuning Summary:")
print(f"   Tuned models: CatBoost, XGBoost, Gradient Boosting")
print(f"   Tuning method: Randomized Search CV (5-fold)")
print(f"   Scoring metric: R² Score")

print(f"\n📈 Model Performance Summary (After Tuning):")
for _, row in results_df_sorted.iterrows():
    print(f"   {row['Rank']}. {row['Model']}: R²={row['R² Score']:.4f}, RMSE={row['RMSE']:.4f} mm")

print(f"\n💡 Key Insights:")
print(f"   1. All models were optimized using hyperparameter tuning")
print(f"   2. Best model found: {best_model} with R²={best_r2:.4f}")
print(f"   3. Stacking ensemble combines the best of individual models")
print(f"   4. Feature importance varies across different algorithms")

print("\n" + "="*80)
print("🎉 All models trained with hyperparameter tuning and evaluated successfully!")
print("="*80)

# Save best parameters to file
import json
with open('best_hyperparameters.json', 'w') as f:
    json.dump(best_params_summary, f, indent=4)
print("\n💾 Best hyperparameters saved to 'best_hyperparameters.json'")

# Save results to CSV
results_df.to_csv('model_performance_results.csv', index=False)
print("💾 Model performance results saved to 'model_performance_results.csv'")