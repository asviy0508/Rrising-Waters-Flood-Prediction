"""
EPIC 3: DATA PRE-PROCESSING
Complete implementation with all 5 stories

Stories:
1. Handling Missing Values
2. Handling Outliers (IQR with Capping)
3. Handling Categorical Values (Feature Mapping + Label Encoding)
4. Splitting Data into Training and Test Sets
5. Feature Scaling (StandardScaler)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

# ============================================
# STORY 1: HANDLING MISSING VALUES
# ============================================

def handle_missing_values(df):
    """
    Story 1: Identify and handle missing values
    """
    print("\n" + "=" * 70)
    print("📖 STORY 1: HANDLING MISSING VALUES")
    print("=" * 70)
    
    # Check for missing values
    print("\n🔍 Checking for missing values:")
    print("-" * 50)
    
    # Using isnull().any() to check if any column has missing values
    print("isnull().any() - Checking if any column has missing values:")
    print(df.isnull().any())
    
    # Using isnull().sum() to count missing values per column
    print("\nisnull().sum() - Counting missing values per column:")
    print(df.isnull().sum())
    
    total_missing = df.isnull().sum().sum()
    print(f"\n📊 Total missing values in dataset: {total_missing}")
    
    if total_missing > 0:
        print("\n⚠️ Missing values detected! Handling them...")
        
        # Fill numerical missing values with median
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().sum() > 0:
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)
                print(f"   ✅ Filled missing values in '{col}' with median: {median_val:.2f}")
        
        # Fill categorical missing values with mode
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].isnull().sum() > 0:
                mode_val = df[col].mode()[0]
                df[col].fillna(mode_val, inplace=True)
                print(f"   ✅ Filled missing values in '{col}' with mode: {mode_val}")
        
        print("\n✅ Missing values handled successfully!")
        print(f"   Total missing values after handling: {df.isnull().sum().sum()}")
    else:
        print("\n✅ No missing values found in the dataset!")
    
    return df


# ============================================
# STORY 2: HANDLING OUTLIERS
# ============================================

def handle_outliers(df):
    """
    Story 2: Detect and handle outliers using IQR method with capping
    """
    print("\n" + "=" * 70)
    print("📖 STORY 2: HANDLING OUTLIERS")
    print("=" * 70)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    outlier_counts = {}
    
    print("\n🔍 Detecting outliers using IQR method:")
    print("-" * 50)
    
    # Exclude target column from outlier handling
    if 'flood' in numeric_cols:
        numeric_cols = [col for col in numeric_cols if col != 'flood']
    
    for col in numeric_cols:
        # Calculate Q1, Q3, and IQR
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        # Calculate bounds
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Count outliers
        outliers_before = df[(df[col] < lower_bound) | (df[col] > upper_bound)].shape[0]
        outlier_counts[col] = outliers_before
        
        if outliers_before > 0:
            print(f"\n   📊 Column: {col}")
            print(f"      Q1: {Q1:.4f}, Q3: {Q3:.4f}, IQR: {IQR:.4f}")
            print(f"      Lower Bound: {lower_bound:.4f}, Upper Bound: {upper_bound:.4f}")
            print(f"      Outliers detected: {outliers_before}")
            
            # Apply capping technique
            df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
            print(f"      ✅ Capped outliers (replaced with bounds)")
    
    total_outliers = sum(outlier_counts.values())
    print(f"\n📊 Total outliers detected: {total_outliers}")
    
    if total_outliers > 0:
        print("✅ Outliers handled successfully using capping technique!")
    else:
        print("✅ No outliers detected in the dataset!")
    
    return df


# ============================================
# STORY 3: HANDLING CATEGORICAL VALUES
# ============================================

def handle_categorical_values(df):
    """
    Story 3: Convert categorical values to numerical using Feature Mapping and Label Encoding
    """
    print("\n" + "=" * 70)
    print("📖 STORY 3: HANDLING CATEGORICAL VALUES")
    print("=" * 70)
    
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    if len(categorical_cols) == 0:
        print("\n✅ No categorical columns found in the dataset!")
        return df, {}
    
    print(f"\n🔍 Found {len(categorical_cols)} categorical columns:")
    for col in categorical_cols:
        print(f"   - {col} (Unique values: {df[col].nunique()})")
    
    print("\n" + "-" * 50)
    
    # Dictionary to store mapping information
    mapping_info = {}
    
    for col in categorical_cols:
        unique_vals = df[col].unique()
        
        # Method 1: Feature Mapping (for columns with small number of unique values)
        if len(unique_vals) <= 5:
            print(f"\n📌 Column: {col} (Using Feature Mapping)")
            print(f"   Unique values: {unique_vals}")
            
            # Create manual mapping dictionary
            mapping = {val: idx for idx, val in enumerate(sorted(unique_vals))}
            print(f"   Mapping: {mapping}")
            
            # Apply mapping
            df[col] = df[col].map(mapping)
            mapping_info[col] = mapping
            print(f"   ✅ Feature Mapping applied!")
        
        # Method 2: Label Encoding (for columns with many unique values)
        else:
            print(f"\n📌 Column: {col} (Using Label Encoding)")
            print(f"   Unique values: {len(unique_vals)} (too many for manual mapping)")
            
            # Use LabelEncoder from scikit-learn
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            mapping_info[col] = dict(zip(le.classes_, le.transform(le.classes_)))
            print(f"   ✅ Label Encoding applied!")
    
    print("\n✅ All categorical values converted to numerical format!")
    return df, mapping_info


# ============================================
# STORY 4: SPLITTING DATA
# ============================================

def split_data(df, target_col='flood', test_size=0.2, random_state=42):
    """
    Story 4: Split dataset into Independent Variables (X) and Target Variable (y)
    """
    print("\n" + "=" * 70)
    print("📖 STORY 4: SPLITTING DATA INTO TRAINING AND TEST SETS")
    print("=" * 70)
    
    # Separate independent variables (X) and target variable (y)
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    print("\n🔍 Data Split Information:")
    print("-" * 50)
    print(f"   Independent Variables (X): {X.shape[1]} features")
    print(f"   Features: {X.columns.tolist()}")
    print(f"   Target Variable (y): {y.name}")
    print(f"   Target Distribution:")
    print(f"      {y.value_counts().to_dict()}")
    
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"\n📊 Train-Test Split Results:")
    print(f"   Training set: {X_train.shape[0]} samples ({100 * (1 - test_size):.0f}%)")
    print(f"   Test set: {X_test.shape[0]} samples ({test_size * 100:.0f}%)")
    print(f"\n   Target distribution in training:")
    print(f"      {y_train.value_counts().to_dict()}")
    print(f"\n   Target distribution in test:")
    print(f"      {y_test.value_counts().to_dict()}")
    
    print("\n✅ Data split completed successfully!")
    
    return X_train, X_test, y_train, y_test, X.columns.tolist()


# ============================================
# STORY 5: FEATURE SCALING
# ============================================

def scale_features(X_train, X_test):
    """
    Story 5: Apply StandardScaler for feature scaling
    """
    print("\n" + "=" * 70)
    print("📖 STORY 5: FEATURE SCALING (StandardScaler)")
    print("=" * 70)
    
    print("\n🔍 Feature Scaling Information:")
    print("-" * 50)
    print("   Method: StandardScaler")
    print("   Formula: z = (x - μ) / σ")
    print("   - x: original value")
    print("   - μ: mean of the feature")
    print("   - σ: standard deviation of the feature")
    print("   Result: Mean = 0, Standard Deviation = 1")
    
    # Create scaler object
    scaler = StandardScaler()
    
    # Fit on training data and transform
    print("\n📊 Scaling Process:")
    print("   Step 1: Fit scaler on training data (learn μ and σ)")
    X_train_scaled = scaler.fit_transform(X_train)
    print("   Step 2: Transform training data")
    
    # Transform test data using same scaler
    X_test_scaled = scaler.transform(X_test)
    print("   Step 3: Transform test data (using training data's μ and σ)")
    
    print("\n📊 Scaling Results:")
    print(f"   Training data shape: {X_train_scaled.shape}")
    print(f"   Test data shape: {X_test_scaled.shape}")
    print(f"   Training data mean: {X_train_scaled.mean():.4f}")
    print(f"   Training data std: {X_train_scaled.std():.4f}")
    
    print("\n✅ Feature scaling completed successfully!")
    print("   ⚠️  REMEMBER: The scaler must be saved for real-time predictions!")
    
    return X_train_scaled, X_test_scaled, scaler


# ============================================
# MAIN PIPELINE
# ============================================

def load_dataset(file_path='data/flood dataset.xlsx'):
    """Load the flood prediction dataset"""
    try:
        df = pd.read_excel(file_path)
        print(f"\n✅ Dataset loaded successfully!")
        print(f"   Shape: {df.shape[0]} rows × {df.shape[1]} columns")
        return df
    except FileNotFoundError:
        print(f"\n❌ Error: File not found at '{file_path}'")
        print("   Please make sure the dataset is in the 'data' folder")
        return None

def save_preprocessed_data(X_train, X_test, y_train, y_test, scaler, feature_names, mapping_info):
    """Save all preprocessed data and objects"""
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    # Save numpy arrays
    np.save('models/X_train.npy', X_train)
    np.save('models/X_test.npy', X_test)
    np.save('models/y_train.npy', y_train.values)
    np.save('models/y_test.npy', y_test.values)
    
    # Save scaler
    with open('models/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    # Save feature names
    with open('models/feature_names.pkl', 'wb') as f:
        pickle.dump(feature_names, f)
    
    # Save mapping info
    with open('models/mapping_info.pkl', 'wb') as f:
        pickle.dump(mapping_info, f)
    
    print("\n📁 Saved preprocessed data:")
    print("   ✅ models/X_train.npy")
    print("   ✅ models/X_test.npy")
    print("   ✅ models/y_train.npy")
    print("   ✅ models/y_test.npy")
    print("   ✅ models/scaler.pkl")
    print("   ✅ models/feature_names.pkl")
    print("   ✅ models/mapping_info.pkl")

def main():
    """
    Main preprocessing pipeline - Executes all 5 stories
    """
    print("=" * 70)
    print("🌊 RISING WATERS - FLOOD PREDICTION")
    print("EPIC 3: DATA PRE-PROCESSING")
    print("=" * 70)
    
    # Load dataset
    df = load_dataset()
    if df is None:
        return
    
    print(f"\n📊 Initial dataset shape: {df.shape}")
    
    # ============================================
    # STORY 1: Handling Missing Values
    # ============================================
    df = handle_missing_values(df)
    
    # ============================================
    # STORY 2: Handling Outliers
    # ============================================
    df = handle_outliers(df)
    
    # ============================================
    # STORY 3: Handling Categorical Values
    # ============================================
    df, mapping_info = handle_categorical_values(df)
    
    # ============================================
    # STORY 4: Splitting Data
    # ============================================
    X_train, X_test, y_train, y_test, feature_names = split_data(df)
    
    # ============================================
    # STORY 5: Feature Scaling
    # ============================================
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    
    # ============================================
    # Save all preprocessed data
    # ============================================
    save_preprocessed_data(X_train_scaled, X_test_scaled, y_train, y_test, 
                          scaler, feature_names, mapping_info)
    
    print("\n" + "=" * 70)
    print("🎉 EPIC 3: DATA PRE-PROCESSING COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print("\n📋 Summary:")
    print(f"   ✅ Missing values handled")
    print(f"   ✅ Outliers handled (IQR with capping)")
    print(f"   ✅ Categorical values encoded ({len(mapping_info)} columns)")
    print(f"   ✅ Data split (Train: {X_train.shape[0]}, Test: {X_test.shape[0]})")
    print(f"   ✅ Features scaled (StandardScaler)")
    print(f"\n📁 All data saved in 'models/' directory")
    print("\n🚀 Ready for EPIC 4: Model Building!")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_names

if __name__ == "__main__":
    X_train, X_test, y_train, y_test, scaler, feature_names = main()