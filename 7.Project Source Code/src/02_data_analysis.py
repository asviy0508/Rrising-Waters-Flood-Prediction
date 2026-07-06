"""
EPIC 2: VISUALIZING AND ANALYSING THE DATA
Stories: Import Libraries, Read Dataset, Univariate, Multivariate, Descriptive Analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

# Set visualization style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Create directories for outputs
os.makedirs('outputs/plots', exist_ok=True)
os.makedirs('outputs/data', exist_ok=True)

def load_dataset(file_path='data/flood dataset.xlsx'):
    """Load the dataset"""
    return pd.read_excel(file_path)

def perform_eda(df):
    """Perform Exploratory Data Analysis"""
    
    print("=" * 60)
    print("EPIC 2: VISUALIZING AND ANALYSING THE DATA")
    print("=" * 60)
    
    # ============================================
    # STORY 2: READING THE DATASET
    # ============================================
    print("\n DATASET INFORMATION:")
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # ============================================
    # STORY 5: DESCRIPTIVE ANALYSIS
    # ============================================
    print("\n STATISTICAL SUMMARY:")
    print(df.describe())
    
    print("\n MISSING VALUES:")
    print(df.isnull().sum())
    
    # Target distribution
    print("\n TARGET DISTRIBUTION (flood):")
    print(df['flood'].value_counts())
    print(f"\nPercentages:")
    print(df['flood'].value_counts(normalize=True) * 100)
    
    # ============================================
    # STORY 3: UNIVARIATE ANALYSIS
    # ============================================
    print("\n" + "=" * 60)
    print("UNIVARIATE ANALYSIS")
    print("=" * 60)
    
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Distribution plots
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    axes = axes.flatten()
    
    for i, col in enumerate(numerical_cols):
        if i < len(axes):
            sns.histplot(df[col], kde=True, ax=axes[i], color='skyblue', edgecolor='black')
            axes[i].set_title(f'Distribution of {col}', fontsize=12)
            axes[i].set_xlabel(col)
    
    for i in range(len(numerical_cols), len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('outputs/plots/distribution_plots.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(" Saved: distribution_plots.png")
    
    # Box plots for outliers
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    axes = axes.flatten()
    
    for i, col in enumerate(numerical_cols):
        if i < len(axes):
            sns.boxplot(y=df[col], ax=axes[i], color='lightgreen')
            axes[i].set_title(f'Box Plot of {col}', fontsize=12)
    
    for i in range(len(numerical_cols), len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('outputs/plots/box_plots.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(" Saved: box_plots.png")
    
    # ============================================
    # STORY 4: MULTIVARIATE ANALYSIS
    # ============================================
    print("\n" + "=" * 60)
    print("MULTIVARIATE ANALYSIS")
    print("=" * 60)
    
    # Correlation heatmap
    plt.figure(figsize=(14, 10))
    correlation_matrix = df.corr(numeric_only=True)
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    
    sns.heatmap(correlation_matrix, 
                mask=mask,
                annot=True, 
                cmap='coolwarm', 
                center=0,
                fmt='.2f',
                square=True,
                linewidths=0.5)
    
    plt.title('Correlation Matrix Heatmap', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('outputs/plots/correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: correlation_heatmap.png")
    
    # Top correlations with target
    target_corr = correlation_matrix['flood'].sort_values(ascending=False)
    print(f"\nTop features correlated with 'flood':")
    print(target_corr)
    
    # Save correlation matrix
    correlation_matrix.to_csv('outputs/data/correlation_matrix.csv')
    
    # ============================================
    # SAVE SUMMARY
    # ============================================
    summary = {
        'shape': df.shape,
        'columns': df.columns.tolist(),
        'missing_values': df.isnull().sum().to_dict(),
        'target_distribution': df['flood'].value_counts().to_dict(),
        'correlation_with_target': target_corr.to_dict()
    }
    
    import json
    with open('outputs/data/eda_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\nEDA Summary saved to outputs/data/eda_summary.json")
    print("\nEPIC 2 COMPLETED SUCCESSFULLY!")
    
    return correlation_matrix

if __name__ == "__main__":
    df = load_dataset()
    if df is not None:
        correlation_matrix = perform_eda(df)