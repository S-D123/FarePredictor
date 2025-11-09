from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import numpy as np
import os
import sys

app = Flask(__name__)

# Load model once at startup
MODEL_PATH = "xgb_model.pkl"
model = None
model_load_error = None

def load_model():
    global model, model_load_error
    try:
        # Debug: Print current directory and files
        print("=" * 60)
        print(f"📂 Current directory: {os.getcwd()}")
        print(f"📂 Files in directory:")
        for f in os.listdir('.'):
            if os.path.isfile(f):
                size = os.path.getsize(f) / 1024 / 1024  # MB
                print(f"   - {f} ({size:.2f} MB)")
            else:
                print(f"   - {f}/ (directory)")
        print("=" * 60)
        
        # Check if model file exists
        if not os.path.exists(MODEL_PATH):
            error_msg = f"Model file not found: {MODEL_PATH}"
            print(f"❌ {error_msg}")
            print(f"📂 Looking in: {os.path.abspath('.')}")
            model_load_error = error_msg
            return False
        
        # Check file size
        file_size = os.path.getsize(MODEL_PATH)
        print(f"✅ Model file found! Size: {file_size / 1024 / 1024:.2f} MB")
        
        # Load model
        model = joblib.load(MODEL_PATH)
        print(f"✅ Model loaded successfully: {type(model).__name__}")
        
        # Debug model features
        if hasattr(model, 'feature_names_in_'):
            print(f"📋 Expected features: {list(model.feature_names_in_)}")
        if hasattr(model, 'n_features_in_'):
            print(f"📊 Number of features: {model.n_features_in_}")
        
        print("=" * 60)
        return True
        
    except Exception as e:
        error_msg = f"Error loading model: {str(e)}"
        print(f"❌ {error_msg}")
        import traceback
        traceback.print_exc()
        model_load_error = error_msg
        return False

def haversine_km(a_lat, a_lng, b_lat, b_lng):
    R = 6371.0
    to_rad = np.pi / 180.0
    dlat = (b_lat - a_lat) * to_rad
    dlng = (b_lng - a_lng) * to_rad
    a = np.sin(dlat / 2.0) ** 2 + np.cos(a_lat * to_rad) * np.cos(b_lat * to_rad) * np.sin(dlng / 2.0) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

def build_features(pickup_lat, pickup_lng, drop_lat, drop_lng, passengers=1):
    """Build features matching the model's training data"""
    dist_km = haversine_km(pickup_lat, pickup_lng, drop_lat, drop_lng)
    
    # IMPORTANT: Match these column names to your model's training features
    features = pd.DataFrame({
        'passenger_count': [int(passengers)],
        'year': [2025],
        'distance': [float(dist_km)]
    })
    
    print(f"🔧 Features created: {features.to_dict('records')[0]}")
    return features

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if model is loaded
        if model is None:
            error_msg = f'Model not loaded. {model_load_error or "Unknown error"}'
            print(f"❌ {error_msg}")
            return jsonify({
                'success': False,
                'error': error_msg
            }), 500
        
        data = request.get_json()
        print("📥 Received payload:", data)
        
        # Extract coordinates
        pickup = data.get('pickup', {})
        drop = data.get('drop', {})
        passengers = data.get('passengers', 4)
        vehicle_mult = data.get('vehicle_mult', 1.0)
        
        pickup_lat = float(pickup.get('lat'))
        pickup_lng = float(pickup.get('lng'))
        drop_lat = float(drop.get('lat'))
        drop_lng = float(drop.get('lng'))
        
        print(f"📍 Pickup: ({pickup_lat}, {pickup_lng})")
        print(f"📍 Drop: ({drop_lat}, {drop_lng})")
        
        # Build features
        X = build_features(pickup_lat, pickup_lng, drop_lat, drop_lng, passengers)
        
        # Predict base fare
        base_prediction = model.predict(X)[0]
        
        # Apply vehicle multiplier and USD to INR conversion
        final_fare = float(base_prediction) * vehicle_mult * 90
        distance = float(X['distance'].iloc[0])
        
        print(f"✅ Prediction successful: ₹{final_fare:.2f}")
        
        # Return prediction
        return jsonify({
            'success': True,
            'fare': round(final_fare, 2),
            'distance': round(distance, 2),
            'passengers': int(passengers),
            'vehicle_mult': float(vehicle_mult)
        })
        
    except Exception as e:
        print(f"❌ Prediction error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok', 
        'model_loaded': model is not None,
        'model_load_error': model_load_error,
        'model_file_exists': os.path.exists(MODEL_PATH),
        'environment': os.environ.get('RAILWAY_ENVIRONMENT', 'local')
    })

@app.route('/debug', methods=['GET'])
def debug():
    """Debug endpoint to check model and file system"""
    model_info = {
        'loaded': model is not None,
        'type': type(model).__name__ if model else None,
        'load_error': model_load_error,
    }
    
    if model:
        if hasattr(model, 'feature_names_in_'):
            model_info['features'] = list(model.feature_names_in_)
        if hasattr(model, 'n_features_in_'):
            model_info['n_features'] = model.n_features_in_
    
    return jsonify({
        'model': model_info,
        'model_file': {
            'exists': os.path.exists(MODEL_PATH),
            'path': os.path.abspath(MODEL_PATH),
            'size_mb': round(os.path.getsize(MODEL_PATH) / 1024 / 1024, 2) if os.path.exists(MODEL_PATH) else 0,
        },
        'filesystem': {
            'cwd': os.getcwd(),
            'files': os.listdir('.'),
        },
        'environment': {
            'railway_env': os.environ.get('RAILWAY_ENVIRONMENT'),
            'port': os.environ.get('PORT'),
            'python_version': sys.version,
        }
    })

@app.route('/')
def home():
    return render_template('index.html')

if __name__ == '__main__':
    print("=" * 60)
    print("🚀 Starting Fare Predictor Server")
    print("=" * 60)
    
    # Load model BEFORE starting the server
    success = load_model()
    
    if not success:
        print("⚠️  WARNING: Model failed to load!")
        print(f"⚠️  Error: {model_load_error}")
        print("⚠️  Server will start but predictions will fail")
        print("=" * 60)
    
    # Get port from environment
    port = int(os.environ.get('PORT', 5000))
    debug_mode = os.environ.get('FLASK_ENV') != 'production'
    
    print(f"🌐 Port: {port}")
    print(f"🌐 Environment: {os.environ.get('RAILWAY_ENVIRONMENT', 'local')}")
    print(f"🐛 Debug mode: {debug_mode}")
    print("=" * 60)
    
    app.run(host='0.0.0.0', port=port, debug=debug_mode)