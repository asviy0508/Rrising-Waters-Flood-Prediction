# test_imports.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb
import flask
import joblib
import openpyxl
import os
import warnings

print("✅ All libraries imported successfully!")
print(f"✅ NumPy version: {np.__version__}")
print(f"✅ Pandas version: {pd.__version__}")
print(f"✅ Scikit-learn version: {sklearn.__version__}")
print(f"✅ Flask version: {flask.__version__}")
print("🎉 Ready to run the project!")