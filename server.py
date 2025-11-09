from flask import Flask, request, jsonify, render_template
# from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
import os

app = Flask(__name__)
# CORS(app)  # Enable CORS for frontend requests

# Load model once at startup
MODEL_PATH = "xgb_model.joblib"
model = None

def load_model():
    global model
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)
    print(f"✅ Model loaded: {type(model).__name__}")

def haversine_km(a_lat, a_lng, b_lat, b_lng):
    R = 6371.0
    to_rad = np.pi / 180.0
    dlat = (b_lat - a_lat) * to_rad
    dlng = (b_lng - a_lng) * to_rad
    a = np.sin(dlat / 2.0) ** 2 + np.cos(a_lat * to_rad) * np.cos(b_lat * to_rad) * np.sin(dlng / 2.0) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

def build_features(pickup_lat, pickup_lng, drop_lat, drop_lng, passengers=1):
    """
    Build feature DataFrame matching your model's training features.
    IMPORTANT: Update these column names to match your Real_Project.ipynb training data!
    """
    dist_km = haversine_km(pickup_lat, pickup_lng, drop_lat, drop_lng)
    
    # Example features - MODIFY based on your actual model features
    features = pd.DataFrame({
        'passenger_count': passengers,
        'year': 2025, # we can later take input 'year' from users
        'distance': dist_km
    }, index=[0])
    return features

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        # Extract coordinates
        pickup = data.get('pickup', {})
        drop = data.get('drop', {})
        passengers = data.get('passengers', 4)
        vehicle_mult = data.get('vehicle_mult', 1.0)
        
        pickup_lat = float(pickup.get('lat'))
        pickup_lng = float(pickup.get('lng'))
        drop_lat = float(drop.get('lat'))
        drop_lng = float(drop.get('lng'))
        
        # Build features
        X = build_features(pickup_lat, pickup_lng, drop_lat, drop_lng, passengers)
        
        # Predict base fare
        base_prediction = model.predict(X)[0]
        
        # Apply vehicle multiplier
        final_fare = base_prediction * vehicle_mult * 90 
        # 90 => scaling factor to convert USD/INR
        
        # Return prediction
        return jsonify({
            'success': True,
            'fare': float(final_fare),
            'distance': float(X['distance'].iloc[0]),
            'passengers': passengers,
            'vehicle_mult': vehicle_mult
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'error': str(e)
        }), 400

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok', 
        'model_loaded': model is not None,
        'environment': os.environ.get('RAILWAY_ENVIRONMENT', 'local')
    })

@app.route('/')
def home():
    return render_template('index.html')

if __name__ == '__main__':
    load_model()
    # Use PORT environment variable (Railway sets this automatically)
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') != 'production'

    print(f"🚀 Starting Flask server on port {port}")
    print(f"🌐 Environment: {os.environ.get('RAILWAY_ENVIRONMENT', 'local')}")

    app.run(debug=debug, port=port, host='0.0.0.0')