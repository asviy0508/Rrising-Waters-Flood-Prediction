@echo off
echo 🔧 Setting up Rising Waters Project...

REM Create virtual environment
python -m venv venv

REM Activate virtual environment
call venv\Scripts\activate

REM Upgrade pip
pip install --upgrade pip

REM Install all requirements
pip install numpy pandas matplotlib seaborn scikit-learn xgboost flask openpyxl joblib

echo ✅ Setup complete!
echo 🚀 Activate environment with: venv\Scripts\activate