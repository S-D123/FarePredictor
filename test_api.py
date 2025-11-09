import requests
import json

payload = {
    "pickup": {"lat": 23.05, "lng": 72.57},
    "drop": {"lat": 23.06, "lng": 72.58},
    "passengers": 4,
    "vehicle_mult": 1.0
}

response = requests.post('http://localhost:5000/predict', json=payload)
print("Status:", response.status_code)
print("Response:", response.json())