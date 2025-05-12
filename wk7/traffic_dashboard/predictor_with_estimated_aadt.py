import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
import joblib
import os

from get_historical_weather_okaloosa import get_weather_forecast
from gru import GRURegressor

# Paths
main_path = "C:/Users/kwill/Desktop/grad_classes/Master_Project_cpsc_69100"
code_path = os.path.join(main_path, "wk7_update", "traffic_dashboard")
csv = os.path.join(code_path, "cleaned_okaloosa_historical_td.csv")
output_csv = os.path.join(code_path, "predicted_with_estimated_aadt.csv")
model_dir = os.path.join(code_path, "models")
scaler_dir = os.path.join(code_path, "scalers")

os.makedirs(model_dir, exist_ok=True)
os.makedirs(scaler_dir, exist_ok=True)

# Weather for Okaloosa County
lat, lon = 30.64, -86.59
#df_weather = get_weather_forecast(lat, lon)
df_weather = get_weather_forecast()
df_weather["WINDSPEED"] = df_weather["WINDSPEED"]
real_weather_row = df_weather.iloc[0]

# Load historical AADT + weather data
df2 = pd.read_csv(csv)
SEQ_LEN = 5
results = []

# Merge historical data with weather data
df = pd.merge(df2, df_weather, on='YEAR_')

def classify_congestion(aadt):
    if aadt < 5000:
        return "Low"
    elif aadt <= 15000:
        return "Medium"
    return "High"

def estimate_aadt_linear(aadt_values):
    x = np.arange(len(aadt_values))
    coef = np.polyfit(x, aadt_values, 1)
    return round(coef[0] * len(aadt_values) + coef[1], 2)

for site in df["COSITE"].unique():
    df_site = df[df["COSITE"] == site].sort_values("YEAR_")
    if len(df_site) < 4:
        print(f"Skipping {site} — only {len(df_site)} years of data")
        continue  # Require exactly 4 years to simulate the 5th

    df_site = df_site.tail(4)  # Always use most recent 4 years

    aadt_values = df_site["AADT"].values.tolist()
    temp_values = df_site["TEMPERATURE"].values.tolist()
    wind_values = df_site["WINDSPEED"].values.tolist()

    est_aadt = estimate_aadt_linear(aadt_values)
    est_temp = real_weather_row["TEMPERATURE"]
    est_wind = real_weather_row["WINDSPEED"]

    # Build full 5-year input
    full_features = [
        [aadt_values[i], temp_values[i], wind_values[i]] for i in range(4)
    ]
    full_features.append([est_aadt, est_temp, est_wind])

    # Normalize
    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(full_features)

    X = torch.tensor(features_scaled[:-1].reshape(1, SEQ_LEN - 1, 3), dtype=torch.float32)
    y = torch.tensor([features_scaled[-1][0]], dtype=torch.float32).view(1, 1)

    # Train model
    model = GRURegressor(input_size=3)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    for epoch in range(100):
        model.train()
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

    # Predict
    with torch.no_grad():
        input_seq = torch.tensor(features_scaled[-SEQ_LEN:].reshape(1, SEQ_LEN, 3), dtype=torch.float32)
        pred_scaled = model(input_seq).item()
        inverse_input = [[pred_scaled, features_scaled[-1][1], features_scaled[-1][2]]]
        pred_aadt = scaler.inverse_transform(inverse_input)[0][0]

    row = df_site.iloc[-1]
    results.append({
        "COSITE": site,
        "ROADWAY": row["ROADWAY"],
        "FROM": row["DESC_FRM"],
        "TO": row["DESC_TO"],
        "ESTIMATED_CURRENT_AADT": est_aadt,
        "FORECASTED_AADT": round(pred_aadt, 2),
        "Congestion_Level": classify_congestion(pred_aadt),
        "TEMPERATURE": est_temp,
        "WINDSPEED": est_wind#,
    })

    # Save model and scaler
    torch.save(model.state_dict(), os.path.join(model_dir, f"gru_{site}.pth"))
    joblib.dump(scaler, os.path.join(scaler_dir, f"scaler_{site}.pkl"))

# Save predictions
df_result = pd.DataFrame(results)
df_result.to_csv(output_csv, index=False)
print(f"Predictions saved to: {output_csv}")