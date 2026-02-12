# -*- coding: utf-8 -*-
"""
Created on Wed Dec 24 11:25:44 2025

@author: Amare
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.cm import get_cmap

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor, StackingRegressor

from xgboost import XGBRegressor
from catboost import CatBoostRegressor

# Set style for colorful plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Color scheme for models
model_colors = {
    'CatBoost': '#FF6B6B',      # Coral Red
    'XGBoost': '#4ECDC4',       # Turquoise
    'GradientBoosting': '#45B7D1',  # Sky Blue
    'Stacking': '#96CEB4'       # Mint Green
}

# ---------------------------
# Load CSV
# ---------------------------
print("📊" + "="*60)
print("📊 LOADING AND PREPROCESSING DATA")
print("📊" + "="*60)

file_path = r"C:\Users\Amare\Desktop\worku\datato232-2.csv"
df = pd.read_csv(file_path, encoding="cp1252")

# Clean column names
df.columns = df.columns.str.strip()

print(f"📁 Dataset Shape: {df.shape}")
print(f"📁 Columns: {list(df.columns)}")

# Define features and target
features = [
    'Name of bacteria',
    'Type of extract',
    'bacteria concentration',
    'AgNP concentration',
    'Resonance  nm',
    'Size nm',
    'shape',
    'Type of bacteria',
    'dispersity'
]

target = 'Inhibition size mm'

# Keep only necessary columns and drop rows with missing values
df = df[features + [target]].dropna()

print(f"✅ Cleaned Dataset Shape: {df.shape}")
print(f"✅ Target variable: '{target}'")

# Split X and y
X = df[features]
y = df[target]

# Identify categorical and numeric columns
categorical_cols = X.select_dtypes(include='object').columns.tolist()
numeric_cols = X.select_dtypes(exclude='object').columns.tolist()

print(f"\n🔍 Data Types Analysis:")
print(f"   Categorical features ({len(categorical_cols)}): {categorical_cols}")
print(f"   Numerical features ({len(numeric_cols)}): {numeric_cols}")

# ---------------------------
# DATA VISUALIZATION - BEFORE TRAINING
# ---------------------------
print("\n🎨" + "="*60)
print("🎨 EXPLORATORY DATA VISUALIZATION")
print("🎨" + "="*60)

# Create figure for initial data exploration
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('📈 Dataset Exploration - Inhibition Size Prediction', fontsize=16, fontweight='bold', y=1.02)

# 1. Target distribution
axes[0, 0].hist(y, bins=30, color='#3498db', edgecolor='black', alpha=0.7)
axes[0, 0].set_title('Target Distribution\n(Inhibition Size mm)', fontweight='bold')
axes[0, 0].set_xlabel('Inhibition Size (mm)')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].grid(True, alpha=0.3)

# 2. Correlation heatmap (numeric features only)
numeric_data = df[numeric_cols + [target]]
corr_matrix = numeric_data.corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
heatmap = axes[0, 1].imshow(corr_matrix.values, cmap='coolwarm', aspect='auto')
axes[0, 1].set_title('Correlation Heatmap\n(Numeric Features)', fontweight='bold')
axes[0, 1].set_xticks(range(len(corr_matrix.columns)))
axes[0, 1].set_yticks(range(len(corr_matrix.columns)))
axes[0, 1].set_xticklabels(corr_matrix.columns, rotation=45, ha='right')
axes[0, 1].set_yticklabels(corr_matrix.columns)
plt.colorbar(heatmap, ax=axes[0, 1])

# Add correlation values
for i in range(len(corr_matrix.columns)):
    for j in range(len(corr_matrix.columns)):
        if i != j:
            axes[0, 1].text(j, i, f'{corr_matrix.iloc[i, j]:.2f}', 
                          ha='center', va='center', color='white' if abs(corr_matrix.iloc[i, j]) > 0.5 else 'black',
                          fontsize=9)

# 3. Categorical feature distribution
if categorical_cols:
    cat_feature = categorical_cols[0]  # Take first categorical feature
    value_counts = X[cat_feature].value_counts().head(10)
    colors = plt.cm.Set3(np.linspace(0, 1, len(value_counts)))
    axes[0, 2].barh(range(len(value_counts)), value_counts.values, color=colors)
    axes[0, 2].set_yticks(range(len(value_counts)))
    axes[0, 2].set_yticklabels(value_counts.index)
    axes[0, 2].set_title(f'Top 10 Categories\n({cat_feature})', fontweight='bold')
    axes[0, 2].set_xlabel('Count')
    axes[0, 2].invert_yaxis()

# 4. Scatter: Inhibition size vs AgNP concentration
if 'AgNP concentration' in numeric_cols:
    scatter = axes[1, 0].scatter(df['AgNP concentration'], y, 
                                 c=y, cmap='viridis', alpha=0.6, s=50)
    axes[1, 0].set_title('Inhibition Size vs AgNP Concentration', fontweight='bold')
    axes[1, 0].set_xlabel('AgNP Concentration')
    axes[1, 0].set_ylabel('Inhibition Size (mm)')
    plt.colorbar(scatter, ax=axes[1, 0]).set_label('Inhibition Size')

# 5. Boxplot for categorical vs target
if categorical_cols:
    cat_for_box = categorical_cols[0] if len(categorical_cols) > 0 else None
    if cat_for_box:
        # Get top 5 categories for clarity
        top_categories = X[cat_for_box].value_counts().head(5).index
        subset = df[df[cat_for_box].isin(top_categories)]
        
        box_data = [subset[subset[cat_for_box] == cat][target].values 
                   for cat in top_categories]
        
        box = axes[1, 1].boxplot(box_data, patch_artist=True)
        
        # Color each box differently
        colors = plt.cm.Pastel1(np.linspace(0, 1, len(top_categories)))
        for patch, color in zip(box['boxes'], colors):
            patch.set_facecolor(color)
        
        axes[1, 1].set_xticklabels(top_categories, rotation=45, ha='right')
        axes[1, 1].set_title(f'Inhibition Size by {cat_for_box}', fontweight='bold')
        axes[1, 1].set_ylabel('Inhibition Size (mm)')
        axes[1, 1].grid(True, alpha=0.3)

# 6. Missing values info
missing_values = df.isnull().sum()
if missing_values.sum() == 0:
    axes[1, 2].text(0.5, 0.5, '✅ No Missing Values', 
                   ha='center', va='center', fontsize=14, 
                   bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
else:
    missing_df = missing_values[missing_values > 0]
    axes[1, 2].barh(range(len(missing_df)), missing_df.values, color='#e74c3c')
    axes[1, 2].set_yticks(range(len(missing_df)))
    axes[1, 2].set_yticklabels(missing_df.index)
    axes[1, 2].set_title('Missing Values by Column', fontweight='bold')
    axes[1, 2].set_xlabel('Number of Missing Values')
    axes[1, 2].invert_yaxis()

plt.tight_layout()
plt.show()

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

print(f"\n🔧" + "="*60)
print(f"🔧 DATA SPLITTING")
print(f"🔧" + "="*60)
print(f"   Training set: {X_train.shape[0]} samples")
print(f"   Test set: {X_test.shape[0]} samples")

# ---------------------------
# Define models
# ---------------------------

# CatBoost can handle categorical directly
cat_model = CatBoostRegressor(
    verbose=0,
    depth=6,
    learning_rate=0.1,
    iterations=400,
    random_state=42,
    cat_features=categorical_cols
)

# XGBoost and GradientBoost need encoded numeric features
xgb_model = Pipeline([
    ("prep", preprocessor),
    ("model", XGBRegressor(
        n_estimators=400,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
        random_state=42
    ))
])

gbr_model = Pipeline([
    ("prep", preprocessor),
    ("model", GradientBoostingRegressor(
        n_estimators=400,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    ))
])

# Stacking model
stack_model = StackingRegressor(
    estimators=[
        ("cat", cat_model),
        ("xgb", xgb_model),
        ("gbr", gbr_model)
    ],
    final_estimator=GradientBoostingRegressor(random_state=42)
)

models = {
    "CatBoost": cat_model,
    "XGBoost": xgb_model,
    "GradientBoosting": gbr_model,
    "Stacking": stack_model
}

# ---------------------------
# Train, predict, evaluate
# ---------------------------
print("\n🤖" + "="*60)
print("🤖 TRAINING MODELS")
print("🤖" + "="*60)

results = []
all_predictions = {}

for name, model in models.items():
    print(f"\n🚀 Training {name}...")
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    all_predictions[name] = preds

    # Calculate metrics
    r2 = r2_score(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    results.append([name, r2, rmse, mae])
    
    print(f"   ✅ R² Score: {r2:.4f}")
    print(f"   ✅ RMSE: {rmse:.4f}")
    print(f"   ✅ MAE: {mae:.4f}")

    # COLORFUL Scatter Plot: Predicted vs Actual
    fig, ax = plt.subplots(figsize=(8, 7))
    
    # Create gradient color based on prediction accuracy
    errors = np.abs(preds - y_test)
    norm_errors = errors / errors.max() if errors.max() > 0 else errors
    
    # Create a colormap from green (good) to red (bad)
    scatter = ax.scatter(y_test, preds, 
                        c=norm_errors, 
                        cmap='RdYlGn_r',  # Red-Yellow-Green reversed
                        alpha=0.7, 
                        s=60, 
                        edgecolors='black', 
                        linewidth=0.5)
    
    # Perfect prediction line
    min_val = min(y_test.min(), preds.min())
    max_val = max(y_test.max(), preds.max())
    ax.plot([min_val, max_val], [min_val, max_val], 
           'k--', linewidth=2, alpha=0.7, label='Perfect Prediction')
    
    # Add R² score as text
    ax.text(0.05, 0.95, f'R² = {r2:.4f}', 
           transform=ax.transAxes, 
           fontsize=14, 
           verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor=model_colors[name], alpha=0.8))
    
    ax.set_xlabel("Actual Inhibition Size (mm)", fontsize=12, fontweight='bold')
    ax.set_ylabel("Predicted Inhibition Size (mm)", fontsize=12, fontweight='bold')
    ax.set_title(f"{name} Model Performance\nPredicted vs Actual", 
                fontsize=14, fontweight='bold', pad=20)
    
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right')
    
    # Add colorbar for error visualization
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Prediction Error (Normalized)', fontsize=10)
    
    plt.tight_layout()
    plt.show()

    # Feature importance (except stacking)
    if name != "Stacking":
        try:
            if name == "CatBoost":
                importances = model.get_feature_importance()
                feature_names = features
            elif name == "XGBoost":
                importances = model.named_steps["model"].feature_importances_
                # Get feature names after preprocessing
                feature_names = []
                for col in categorical_cols:
                    unique_vals = X_train[col].unique()
                    for val in unique_vals:
                        feature_names.append(f"{col}_{val}")
                feature_names.extend(numeric_cols)
                # Truncate if too many features
                if len(feature_names) > len(importances):
                    feature_names = feature_names[:len(importances)]
            else:  # GradientBoosting
                importances = model.named_steps["model"].feature_importances_
                feature_names = features
            
            # Create colorful feature importance plot
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Sort features by importance
            sorted_idx = np.argsort(importances)
            sorted_importances = importances[sorted_idx]
            sorted_features = [feature_names[i] for i in sorted_idx]
            
            # Create color gradient based on importance
            colors = plt.cm.plasma(np.linspace(0.2, 0.8, len(sorted_importances)))
            
            bars = ax.barh(range(len(sorted_importances)), sorted_importances, color=colors)
            
            ax.set_yticks(range(len(sorted_importances)))
            ax.set_yticklabels(sorted_features, fontsize=9)
            ax.set_xlabel("Importance Score", fontsize=12, fontweight='bold')
            ax.set_title(f"{name} - Feature Importance", fontsize=14, fontweight='bold', pad=20)
            ax.grid(True, alpha=0.3, axis='x')
            
            # Add value labels on bars
            for i, (bar, val) in enumerate(zip(bars, sorted_importances)):
                width = bar.get_width()
                if width > 0.001:  # Only label significant bars
                    ax.text(width + 0.001, bar.get_y() + bar.get_height()/2, 
                           f'{val:.3f}', 
                           va='center', ha='left', fontsize=8)
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"⚠ Feature importance not available for {name}: {str(e)[:50]}")

# ---------------------------
# COMPARISON VISUALIZATION
# ---------------------------
print("\n📊" + "="*60)
print("📊 MODEL PERFORMANCE COMPARISON")
print("📊" + "="*60)

# Results table
results_df = pd.DataFrame(results, columns=["Model", "R² Score", "RMSE", "MAE"])
print("\n" + "="*80)
print("FINAL MODEL PERFORMANCE:")
print("="*80)
print(results_df.to_string(index=False))
print("="*80)

# Create comparison visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('🎯 Model Performance Comparison Dashboard', fontsize=16, fontweight='bold', y=1.02)

# 1. R² Score Comparison (Bar Chart)
axes[0, 0].bar(results_df['Model'], results_df['R² Score'], 
               color=[model_colors[m] for m in results_df['Model']], 
               edgecolor='black', linewidth=1.5)
axes[0, 0].set_title('R² Score Comparison', fontsize=14, fontweight='bold')
axes[0, 0].set_ylabel('R² Score', fontsize=12)
axes[0, 0].set_ylim([0, 1])
axes[0, 0].grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for i, (model, r2) in enumerate(zip(results_df['Model'], results_df['R² Score'])):
    axes[0, 0].text(i, r2 + 0.02, f'{r2:.3f}', 
                   ha='center', va='bottom', fontweight='bold')

# 2. RMSE Comparison (Bar Chart)
axes[0, 1].bar(results_df['Model'], results_df['RMSE'], 
               color=[model_colors[m] for m in results_df['Model']], 
               edgecolor='black', linewidth=1.5)
axes[0, 1].set_title('RMSE Comparison (Lower is Better)', fontsize=14, fontweight='bold')
axes[0, 1].set_ylabel('RMSE (mm)', fontsize=12)
axes[0, 1].grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for i, (model, rmse) in enumerate(zip(results_df['Model'], results_df['RMSE'])):
    axes[0, 1].text(i, rmse + 0.02, f'{rmse:.3f}', 
                   ha='center', va='bottom', fontweight='bold')

# 3. MAE Comparison (Bar Chart)
axes[1, 0].bar(results_df['Model'], results_df['MAE'], 
               color=[model_colors[m] for m in results_df['Model']], 
               edgecolor='black', linewidth=1.5)
axes[1, 0].set_title('MAE Comparison (Lower is Better)', fontsize=14, fontweight='bold')
axes[1, 0].set_ylabel('MAE (mm)', fontsize=12)
axes[1, 0].grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for i, (model, mae) in enumerate(zip(results_df['Model'], results_df['MAE'])):
    axes[1, 0].text(i, mae + 0.02, f'{mae:.3f}', 
                   ha='center', va='bottom', fontweight='bold')

# 4. All Models Scatter Plot (combined)
axes[1, 1].set_title('All Models: Predicted vs Actual', fontsize=14, fontweight='bold')
axes[1, 1].set_xlabel('Actual Inhibition Size (mm)', fontsize=12)
axes[1, 1].set_ylabel('Predicted Inhibition Size (mm)', fontsize=12)

markers = ['o', 's', '^', 'D']
for idx, (name, preds) in enumerate(all_predictions.items()):
    axes[1, 1].scatter(y_test, preds, 
                      color=model_colors[name], 
                      marker=markers[idx], 
                      alpha=0.6, 
                      s=50, 
                      edgecolors='black', 
                      linewidth=0.5,
                      label=f'{name} (R²={results_df.loc[idx, "R² Score"]:.3f})')

# Perfect prediction line
min_val = min(y_test.min(), min([p.min() for p in all_predictions.values()]))
max_val = max(y_test.max(), max([p.max() for p in all_predictions.values()]))
axes[1, 1].plot([min_val, max_val], [min_val, max_val], 
               'k--', linewidth=2, alpha=0.5, label='Perfect Prediction')

axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].legend(loc='upper left')

plt.tight_layout()
plt.show()

# ---------------------------
# RESIDUAL ANALYSIS
# ---------------------------
print("\n🔍" + "="*60)
print("🔍 RESIDUAL ANALYSIS")
print("🔍" + "="*60)

fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('📉 Residual Analysis for All Models', fontsize=16, fontweight='bold', y=1.02)

for idx, (name, preds) in enumerate(all_predictions.items()):
    row = idx // 2
    col = idx % 2
    
    residuals = y_test - preds
    
    # Residuals vs Predicted
    scatter = axes[row, col].scatter(preds, residuals, 
                                    color=model_colors[name], 
                                    alpha=0.6, 
                                    s=40)
    axes[row, col].axhline(y=0, color='red', linestyle='--', alpha=0.7)
    axes[row, col].set_title(f'{name} - Residuals', fontsize=13, fontweight='bold')
    axes[row, col].set_xlabel('Predicted Values', fontsize=11)
    axes[row, col].set_ylabel('Residuals', fontsize=11)
    axes[row, col].grid(True, alpha=0.3)
    
    # Add residual statistics
    residual_mean = residuals.mean()
    residual_std = residuals.std()
    axes[row, col].text(0.05, 0.95, 
                       f'Mean: {residual_mean:.3f}\nStd: {residual_std:.3f}', 
                       transform=axes[row, col].transAxes,
                       fontsize=10,
                       verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.show()

# ---------------------------
# FINAL SUMMARY
# ---------------------------
print("\n" + "="*80)
print("🎉 FINAL SUMMARY")
print("="*80)

# Find best model
best_model_idx = results_df['R² Score'].idxmax()
best_model = results_df.loc[best_model_idx, 'Model']
best_r2 = results_df.loc[best_model_idx, 'R² Score']
best_rmse = results_df.loc[best_model_idx, 'RMSE']

print(f"\n🏆 BEST PERFORMING MODEL: {best_model}")
print(f"   R² Score: {best_r2:.4f}")
print(f"   RMSE: {best_rmse:.4f} mm")
print(f"   MAE: {results_df.loc[best_model_idx, 'MAE']:.4f} mm")

print("\n📈 Performance Ranking (by R² Score):")
print("-"*50)
results_sorted = results_df.sort_values('R² Score', ascending=False)
for i, (_, row) in enumerate(results_sorted.iterrows(), 1):
    print(f"{i}. {row['Model']}: R²={row['R² Score']:.4f}, RMSE={row['RMSE']:.4f}, MAE={row['MAE']:.4f}")

print("\n" + "="*80)
print("✅ ANALYSIS COMPLETE")
print("="*80)