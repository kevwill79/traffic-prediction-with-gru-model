# -*- coding: utf-8 -*-
"""
Created on Wed Apr 23 19:15:00 2025

@author: kwill
"""

import pandas as pd
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler

from weather import get_weather_forecast
from gru import GRURegressor

# Okaloosa County, FL
lat, lon = 30.64, -86.59

# Paths
main_path = "C:/Users/kwill/Desktop/grad_classes/Master_Project_cpsc_69100"
code_path = f"{main_path}/code/wk6_updates"
csv = f"{code_path}/cleaned_okaloosa_historical_td.csv"
output_csv = f"{code_path}/aadt_congestion_weather.csv"

# Load 7-day weather forecast
df_weather = get_weather_forecast(lat, lon)

# Preprocess weather
# Simplify wind speed (remove "mph" text)
df_weather["windSpeed"] = df_weather["windSpeed"].str.extract(r'(\d+)').astype(float)

# Format date correctly
df_weather['Date'] = pd.to_datetime(df_weather['startTime']).dt.date

# Take the first forecast day's weather (today's or tomorrow's)
first_forecast = df_weather.iloc[0]

# Assign the same forecast date to all traffic predictions
today = datetime.today().date()
today = today.strftime('%m-%d-%y')  # Convert today to a string in the same format as the column

#df_aadt_pred['FORECAST_DATE'] = today

# Add weather columns manually
#df_aadt_pred['TEMPERATURE'] = first_forecast['temperature']
#df_aadt_pred['WINDSPEED'] = first_forecast['windSpeed']
#df_aadt_pred['CURRENT_FORECAST'] = first_forecast['shortForecast']

# Load dataset
df = pd.read_csv(csv)
SEQ_LEN = 5
results = []

def classify_congestion(aadt):
    if aadt < 5000:
        return "Low"
    elif aadt <= 15000:
        return "Medium"
    return "High"

# Process each site
for site in df["COSITE"].unique():
    df_site = df[df["COSITE"] == site].sort_values("YEAR_")
    aadt_series = df_site["AADT"].values.reshape(-1, 1)

    if len(aadt_series) != SEQ_LEN:
        continue  # Only include sites with exactly 5 years of data

    # Normalize
    scaler = MinMaxScaler()
    aadt_scaled = scaler.fit_transform(aadt_series)

    # Create training tensors (1 sequence)
    X = torch.tensor(aadt_scaled[:-1].reshape(1, SEQ_LEN - 1, 1), dtype=torch.float32)
    y = torch.tensor(aadt_scaled[-1].reshape(1, 1), dtype=torch.float32)

    # Model
    model = GRURegressor(input_size=1)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Train
    for epoch in range(100):
        model.train()
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

    # Predict next AADT
    with torch.no_grad():
        input_seq = torch.tensor(aadt_scaled[-SEQ_LEN:], dtype=torch.float32).unsqueeze(0)
        pred_scaled = model(input_seq).item()
        pred_aadt = scaler.inverse_transform([[pred_scaled]])[0][0]

    # Gather info
    last_year = int(df_site["YEAR_"].max())
    weather_row = df_site.iloc[-1]

    results.append({
        "COSITE": site,
        "ROADWAY": weather_row["ROADWAY"],
        "FROM": weather_row["DESC_FRM"],
        "TO": weather_row["DESC_TO"],
        "LAST_YEAR_RECORDED": last_year,
        "FORECASTED_YEAR": last_year + 1,
        "FORECASTED_AADT": round(pred_aadt, 2),
        "Congestion_Level": classify_congestion(pred_aadt),
#        "FORECAST_DATE": df_weather[today],
        "FORECAST_DATE": df_weather["Date"][1],
        "TEMPERATURE": first_forecast["temperature"],
        "WINDSPEED": first_forecast["windSpeed"],
        "CURRENT_FORECAST": first_forecast["shortForecast"]
    })

# Save
df_result = pd.DataFrame(results)
df_result.to_csv(output_csv, index=False)

print(f"Forecasts saved to '{output_csv}'")
print(df_result.head())


'''
# Load dataset
df = pd.read_csv(csv)

# Prediction configuration
SEQ_LEN = 5
EPOCHS = 50
LR = 0.01

predicted_results = []

# Loop through each COSITE
for site in df["COSITE"]:
#for site in df["COSITE"].unique():
    df_site = df[df["COSITE"] == site].sort_values("YEAR_")
    aadt_series = df_site["AADT"].values.reshape(-1, 1)

    # Skip if not enough history
    if len(aadt_series) < SEQ_LEN:
        print(site, aadt_series[0])
        continue

    # Normalize
    scaler = MinMaxScaler()
    aadt_scaled = scaler.fit_transform(aadt_series)

    # Prepare sequences
    X, y = [], []
    for i in range(len(aadt_scaled) - SEQ_LEN):
        X.append(aadt_scaled[i:i+SEQ_LEN])
        y.append(aadt_scaled[i+SEQ_LEN])
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)

    # Define GRU model
    model = GRURegressor(input_size=1)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # Train model
    for epoch in range(EPOCHS):
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

    # Store result
    last_year = int(df_site["YEAR_"].max())
    predicted_results.append({
        "COSITE": site,
        "ROADWAY": df_site["ROADWAY"].iloc[0],
        "FROM": df_site["DESC_FRM"].iloc[0],
        "TO": df_site["DESC_TO"].iloc[0],
        "LAST_YEAR_RECORDED": last_year,
        "FORECASTED_YEAR": last_year + 1,
        "FORECASTED_AADT": round(pred_aadt, 2)
    })


# Save predictions
df_pred = pd.DataFrame(predicted_results)
output_path = f"{code_path}/okaloosa_aadt_prediction.csv"
df_pred.to_csv(output_path, index=False)

#print(f"Forecasts saved to '{output_path}'")
#print(df_pred.head())
'''


'''
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler

from gru import GRURegressor

# Paths
main_path = "C:/Users/kwill/Desktop/grad_classes/Master_Project_cpsc_69100"
code_path = f"{main_path}/code/apr21_27_code_update"
csv = f"{code_path}/traffic_pred_weather.csv"

# Number to skip elements in dataframe
num = 0

# --- Step 1: Load cleaned Okaloosa dataset ---
df = pd.read_csv(csv)
df_sites = pd.DataFrame()

# --- Step 3: Forecasting for each Site ---
predicted_results = []
SEQ_LEN = 5  # Use last 5 years (2019-2023) to predict next

for site in df["COSITE"].unique():
    df_sites = df[df["COSITE"] == site].sort_values("YEAR_")
    aadt_series = df_sites["AADT"].values.reshape(-1, 1)
#    df_sites = pd.concat([df_sites, df[df["COSITE"] == site].sort_values("YEAR_")])
#    aadt_series = df_sites["AADT"].values.reshape(-1, 1)
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
    model = GRURegressor()
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
 #       "ROADWAY": df_sites["ROADWAY"].iloc[num],
 #       "FROM": df_sites["DESC_FRM"].iloc[num],
 #       "TO": df_sites["DESC_TO"].iloc[num],
         "ROADWAY": df_sites["ROADWAY"].iloc[0],
         "FROM": df_sites["DESC_FRM"].iloc[0],
         "TO": df_sites["DESC_TO"].iloc[0],
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
'''