# -*- coding: utf-8 -*-
"""
Created on Wed Apr 23 19:15:00 2025

@author: kwill
"""

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler

from get_gru_classifier import GRUClassifier

# Paths
main_path = "C:/Users/kwill/Desktop/grad_classes/Master_Project_cpsc_69100"

code_path = f"{main_path}/code/apr21_27_code_update"
csv_path = f"{code_path}/cleaned_okaloosa_historical_td.csv"

# Number to skip elements in dataframe
num = 0

# --- Step 1: Load cleaned Okaloosa dataset ---
df = pd.read_csv(csv_path)
df_sites = pd.DataFrame()

# --- Step 3: Forecasting for each Site ---
predicted_results = []
SEQ_LEN = 5  # Use last 5 years (2019-2023) to predict next

for site in df["COSITE"].unique():
    df_sites = pd.concat([df_sites, df[df["COSITE"] == site].sort_values("YEAR_")])
    aadt_series = df_sites["AADT"].values.reshape(-1, 1)
    print(df_sites['COSITE'], len(aadt_series))
    if len(aadt_series) <= SEQ_LEN:
        continue  # Not enough data for forecasting

    # Normalize AADT
    scaler = MinMaxScaler()
    aadt_scaled = scaler.fit_transform(aadt_series)

    # Create sequences
    X, y = [], []
    for i in range(len(aadt_scaled) - SEQ_LEN):
        X.append(aadt_scaled[i:i+SEQ_LEN])
        y.append(aadt_scaled[i+SEQ_LEN])
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)

    # Model setup
    model = GRUClassifier()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Train
    for epoch in range(50):
        model.train()
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

    # Predict next AADT
    with torch.no_grad():
        last_seq = torch.tensor(aadt_scaled[-SEQ_LEN:], dtype=torch.float32).unsqueeze(0)
        pred_scaled = model(last_seq).item()
        pred_aadt = scaler.inverse_transform([[pred_scaled]])[0][0]

    # Save forecast
    last_year = int(df_sites["YEAR_"].max())

    predicted_results.append({
        "COSITE": site,
        "ROADWAY": df_sites["ROADWAY"].iloc[num],
        "FROM": df_sites["DESC_FRM"].iloc[num],
        "TO": df_sites["DESC_TO"].iloc[num],
        "LAST_YEAR_RECORDED": last_year,
        "FORECASTED_YEAR": last_year + 1,
        "FORECASTED_AADT": round(pred_aadt, 2)
    })
    num += 5

# --- Step 4: Save All Forecasts ---
df_pred = pd.DataFrame(predicted_results)
df_pred.to_csv(f"{code_path}/okaloosa_aadt_prediction.csv", index=False)

print("Forecasts saved to 'okaloosa_aadt_prediction.csv'")
print(df_pred.head())
