# 🚖 Fare Predictor - ML-Powered Taxi Fare Estimation

Live taxi fare prediction using XGBoost machine learning model.

## 🚀 Live Demo

[Your Railway URL will be here]

## 📦 Local Development

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/ML_Project.git
cd ML_Project

# Create virtual environment
python -m venv .venv

# Activate (Windows PowerShell)
.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Run locally
python server.py
```

Open http://localhost:5000

## 📁 Project Structure

```
ML_Project/
├── server.py              # Flask backend + API
├── templates/
│   └── index.html        # Frontend UI
├── xgb_model.pkl         # Trained ML model
├── requirements.txt      # Python dependencies
├── Procfile             # Railway/Heroku config
├── runtime.txt          # Python version
└── README.md
```

## 🤖 Tech Stack

- **Backend:** Flask
- **ML Model:** XGBoost (scikit-learn wrapper)
- **Frontend:** HTML/CSS/JavaScript + Leaflet.js
- **Deployment:** Railway (free tier)

## 📊 Model Features

- Passenger count
- Trip distance (Haversine)
- Year (for temporal adjustments)
