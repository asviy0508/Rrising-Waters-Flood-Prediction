"""
EPIC 1: DATA COLLECTION
Story 1: Download and Load the Dataset
"""

import pandas as pd
import numpy as np
import os

# Create data directory if it doesn't exist
os.makedirs('data', exist_ok=True)

def load_dataset(file_path='data/flood dataset.xlsx'):
    """
    Load the flood prediction dataset
    """
    try:
        # Load the dataset
        df = pd.read_excel(file_path)
        
        print("=" * 60)
        print("DATASET LOADED SUCCESSFULLY!")
        print("=" * 60)
        print(f"Dataset Shape: {df.shape[0]} rows × {df.shape[1]} columns")
        print(f"\nColumns: {df.columns.tolist()}")
        print(f"\nFirst 5 rows:")
        print(df.head())
        print(f"\nData Types:")
        print(df.dtypes)
        print(f"\nMissing Values:")
        print(df.isnull().sum())
        
        # Save basic info
        df_info = {
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'dtypes': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict()
        }
        
        return df, df_info
        
    except FileNotFoundError:
        print(f"❌ Error: File not found at {file_path}")
        print("Please make sure the dataset is in the 'data' folder")
        return None, None

if __name__ == "__main__":
    # Run the data collection
    df, info = load_dataset()
    if df is not None:
        print("\n✅ Epic 1 Completed Successfully!")