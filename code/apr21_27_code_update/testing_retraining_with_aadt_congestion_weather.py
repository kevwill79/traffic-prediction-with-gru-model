# -*- coding: utf-8 -*-
"""
Created on Sun Apr 27 08:23:30 2025

@author: kwill
"""
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from get_gru_classifier import GRUClassifier

# Paths
main_path = "C:/Users/kwill/Desktop/grad_classes/Master_Project_cpsc_69100"

code_path = f"{main_path}/code/apr21_27_code_update"
training_csv = f"{code_path}/traffic_pred_weather.csv"

# Load dataset (traffic + weather)
df_retraining = pd.read_csv(training_csv)
'''
# --- Define GRU Regressor ---
class GRURegressor(nn.Module):
    def __init__(self, input_size=3, hidden_size=64):
        super(GRURegressor, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        _, h_n = self.gru(x)
        return self.fc(h_n[-1])
'''
# --- Forecasting for each Site ---
forecast_results = []
SEQ_LEN = 3

for site in df_retraining["COSITE"].unique():
    df_site = df_retraining[df_retraining["COSITE"] == site].sort_values("FORECAST_DATE")

#    if len(df_site) <= SEQ_LEN:
#        continue  # Not enough data

    # Features: AADT, temperature, windspeed
    features = df_site[["FORECASTED_AADT", "TEMPERATURE", "WINDSPEED"]].values
    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(features)

    # Create sequences
    X, y = [], []
    for i in range(len(features_scaled) - SEQ_LEN):
        X.append(features_scaled[i:i+SEQ_LEN])
        y.append(features_scaled[i+SEQ_LEN][0])  # AADT

    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)

    # Model
    #model = GRURegressor()
    model = GRUClassifier()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    for epoch in range(50):
        model.train()
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

    # Predict next year's AADT
    with torch.no_grad():
        last_seq = torch.tensor(features_scaled[-SEQ_LEN:], dtype=torch.float32).unsqueeze(0)

        pred_scaled = model(last_seq).item()
        pred_aadt = scaler.inverse_transform([[pred_scaled, *features[-1, 1:]]])[0][0]

    last_year = int(df_site["FORECAST_DATE"].max())
    forecast_results.append({
        "COSITE": site,
        "ROADWAY": df_site["ROADWAY"].iloc[0],
        "FROM": df_site["FROM"].iloc[0],
        "TO": df_site["TO"].iloc[0],
        "LAST_YEAR": last_year,
        "NEXT_YEAR": last_year + 1,
        "PREDICTED_AADT": round(pred_aadt, 2)
    })

# Save forecasts
df_forecast_weather = pd.DataFrame(forecast_results)
df_forecast_weather.to_csv(f"{code_path}/okaloosa_retrained_predictions_weather.csv", index=False)

print("Retrained forecasts with weather saved!")
print(df_forecast_weather.head())

