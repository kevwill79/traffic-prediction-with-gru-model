from flask import Flask, request, jsonify, render_template
import torch
import joblib
import os
import pandas as pd
from gru import GRURegressor

app = Flask(__name__, template_folder="templates")

# Paths
main_path = "C:/Users/kwill/Desktop/grad_classes/Master_Project_cpsc_69100"
code_path = os.path.join(main_path, "wk7_update", "traffic_dashboard")
model_dir = os.path.join(code_path, "models")
scaler_dir = os.path.join(code_path, "scalers")

# Helper to classify congestion
def classify_congestion(aadt):
    if aadt < 5000:
        return "Low"
    elif aadt <= 15000:
        return "Medium"
    return "High"

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        site = data.get("site")
        features = data.get("features")  # List of [AADT, TEMP, WIND] values

        if not site or not features or len(features) != 5 or len(features[0]) != 3:
            return jsonify({"error": "Input must include 'site' and 5 timesteps with 3 features each."}), 400

        # Load model and scaler for site
        model_path = os.path.join(model_dir, f"gru_{site}.pth")
        scaler_path = os.path.join(scaler_dir, f"scaler_{site}.pkl")

        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            return jsonify({"error": f"No model or scaler found for site: {site}"}), 404

        model = GRURegressor(input_size=3)
        model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
        model.eval()

        scaler = joblib.load(scaler_path)

        # Scale input
        df_input = pd.DataFrame(features, columns=["AADT", "TEMPERATURE", "WINDSPEED"])
        features_scaled = scaler.transform(df_input)
        input_tensor = torch.tensor(features_scaled, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            pred_scaled = model(input_tensor).item()

        # Use last known temp/wind scaled values
        temp_scaled = features_scaled[-1][1]
        wind_scaled = features_scaled[-1][2]
        inverse_input = [[pred_scaled, temp_scaled, wind_scaled]]
        prediction = scaler.inverse_transform(pd.DataFrame(inverse_input, columns=["AADT", "TEMPERATURE", "WINDSPEED"]))[0][0]

        congestion = classify_congestion(prediction)

        return jsonify({
            "site": site,
            "forecasted_AADT": round(prediction, 2),
            "congestion_level": congestion
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=8080)
