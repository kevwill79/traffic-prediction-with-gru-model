# -*- coding: utf-8 -*-
"""
Created on Sun May 4 05:02:14 2025

@author: kwill
"""

from flask import Flask, request, jsonify
import torch
import pandas as pd
from gru import GRURegressor
from sklearn.preprocessing import MinMaxScaler
import os

app = Flask(__name__)

# Paths
main_path = "C:/Users/kwill/Desktop/grad_classes/Master_Project_cpsc_69100"
code_path = os.path.join(main_path, "code", "wk6_updates")
csv = os.path.join(code_path, "aadt_congestion_weather.csv")
model_path = os.path.join(code_path, "gru_regressor.pth")

# Check files
if not os.path.exists(csv):
    raise FileNotFoundError(f"CSV not found: {csv}")
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")

# Load training scaler data
df = pd.read_csv(csv)
required_cols = ["FORECASTED_AADT", "TEMPERATURE", "WINDSPEED"]
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"Column missing in CSV: {col}")

# Load model
model = GRURegressor(input_size=3)
model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
model.eval()

# Fit scaler with training features
scaler = MinMaxScaler()
scaler.fit(df[required_cols])

def classify_congestion(aadt):
    if aadt < 5000:
        return "Low"
    elif aadt <= 15000:
        return "Medium"
    return "High"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        features = data.get("features")
        if not features or len(features) != 5 or len(features[0]) != 3:
            return jsonify({"error": "Input must be shape (5, 3): 5 timesteps with 3 features each"}), 400

        features_scaled = scaler.transform(features)
        input_tensor = torch.tensor(features_scaled, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            prediction = model(input_tensor).item()

        congestion = classify_congestion(prediction)
        return jsonify({
            "forecasted_AADT": round(prediction, 2),
            "congestion_level": congestion
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)