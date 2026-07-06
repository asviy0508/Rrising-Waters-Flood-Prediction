"""
EPIC 5: APPLICATION BUILDING
Complete Flask web application for flood prediction
"""

from flask import Flask, render_template, request, redirect, url_for, jsonify
import numpy as np
import pandas as pd
import os
import warnings
import joblib
from datetime import datetime

warnings.filterwarnings('ignore')

# ============================================
# APPLICATION SETUP AND INITIALIZATION
# ============================================

app = Flask(__name__)
app.secret_key = 'rising_waters_flood_prediction_secret_key_2024'

# Load saved model and scaler
def load_model():
    try:
        model = joblib.load('models/floods.save')
        print("✅ Model loaded successfully: models/floods.save")
        
        scaler = joblib.load('models/transform.save')
        print("✅ Scaler loaded successfully: models/transform.save")
        
        with open('models/feature_names.pkl', 'rb') as f:
            import pickle
            feature_names = pickle.load(f)
        print(f"✅ Feature names loaded: {len(feature_names)} features")
        
        return model, scaler, feature_names
    
    except FileNotFoundError as e:
        print(f"❌ Error loading model files: {e}")
        print("   Please make sure you have run 04_model_building.py first")
        return None, None, None

model, scaler, feature_names = load_model()

# ============================================
# PAGE ROUTING
# ============================================

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict')
def predict_form():
    if model is None:
        return render_template('error.html', 
                             error="Model not loaded. Please check server logs.")
    return render_template('index.html', features=feature_names)

@app.route('/result', methods=['POST'])
def predict_result():
    try:
        input_data = {}
        for feature in feature_names:
            value = request.form.get(feature, '0')
            try:
                input_data[feature] = float(value)
            except ValueError:
                input_data[feature] = 0.0
        
        input_values = np.array([list(input_data.values())])
        input_scaled = scaler.transform(input_values)
        
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0]
        flood_probability = probability[1] * 100
        
        if prediction == 1:
            return render_template('chance.html',
                                 input_data=input_data,
                                 probability=flood_probability,
                                 prediction=prediction)
        else:
            return render_template('no_chance.html',
                                 input_data=input_data,
                                 probability=flood_probability,
                                 prediction=prediction)
    
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return render_template('error.html', error=str(e))

@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        data = request.get_json()
        input_values = []
        for feature in feature_names:
            value = data.get(feature, 0)
            input_values.append(float(value))
        
        input_array = np.array([input_values])
        input_scaled = scaler.transform(input_array)
        
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0].tolist()
        
        return jsonify({
            'success': True,
            'prediction': int(prediction),
            'flood_risk': 'Yes' if prediction == 1 else 'No',
            'probability': {
                'no_flood': probability[0],
                'flood': probability[1]
            },
            'input_data': dict(zip(feature_names, input_values))
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/health')
def health_check():
    return jsonify({
        'status': 'running',
        'model_loaded': model is not None,
        'scaler_loaded': scaler is not None,
        'timestamp': datetime.now().isoformat()
    })

# ============================================
# ERROR HANDLERS
# ============================================

@app.errorhandler(404)
def page_not_found(e):
    return render_template('error.html', error="Page not found (404)"), 404

@app.errorhandler(500)
def internal_server_error(e):
    return render_template('error.html', error="Internal server error (500)"), 500

# ============================================
# APPLICATION LAUNCH - USING PORT 3000
# ============================================

if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("🌊 RISING WATERS - FLOOD PREDICTION APPLICATION")
    print("=" * 70)
    
    if model is not None and scaler is not None:
        print("✅ Model loaded successfully")
        print("✅ Scaler loaded successfully")
        print(f"✅ Features: {len(feature_names)}")
        
        print("\n🚀 Starting Flask server on PORT 3000...")
        print("📍 Access the application at: http://127.0.0.1:3000")
        print("📍 Local URL: http://localhost:3000")
        
        print("\n📋 Available routes:")
        print("   /          - Home page")
        print("   /predict   - Prediction form")
        print("   /result    - Result page (POST)")
        print("   /api/predict - API endpoint")
        print("   /health    - Health check")
        
        print("\n⚠️  Press CTRL+C to stop the server")
        print("=" * 70)
        
        # ⭐ CHANGED: Using port 3000
        app.run(debug=True, host='0.0.0.0', port=3000)
    else:
        print("❌ Failed to load model. Please check the models directory.")
