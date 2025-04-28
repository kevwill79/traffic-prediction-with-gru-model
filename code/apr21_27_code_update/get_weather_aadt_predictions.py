# -*- coding: utf-8 -*-
"""
Created on Fri Apr 25 22:24:59 2025

@author: kwill
"""
import pandas as pd
import numpy as np
from datetime import datetime

# Paths
main_path = "C:/Users/kwill/Desktop/grad_classes/Master_Project_cpsc_69100"

code_path = f"{main_path}/code/apr21_27_code_update"
pred_congest_csv = f"{code_path}/okaloosa_predictions_congestion.csv"
weather_csv = f"{code_path}/seven_day_forecast.csv"
congestion_weather_output = f"{code_path}/traffic_pred_weather.csv"

# Load your traffic forecast CSV (AADT predictions)
df_aadt_pred = pd.read_csv(pred_congest_csv)

# Load 7-day weather forecast
df_weather = pd.read_csv(weather_csv)

# --- Preprocess weather ---
# Simplify wind speed (remove "mph" text)
df_weather["windSpeed"] = df_weather["windSpeed"].str.extract(r'(\d+)').astype(float)

# Format date correctly
df_weather['Date'] = pd.to_datetime(df_weather['startTime']).dt.date

# Take the first forecast day's weather (today's or tomorrow's)
first_forecast = df_weather.iloc[0]

# --- Assign the same forecast date to all traffic predictions ---
today = datetime.today().date()
df_aadt_pred['FORECAST_DATE'] = today

# --- Add weather columns manually ---
df_aadt_pred['TEMPERATURE'] = first_forecast['temperature']
df_aadt_pred['WINDSPEED'] = first_forecast['windSpeed']
df_aadt_pred['CURRENT_FORECAST'] = first_forecast['shortForecast']

# Save the new merged file
df_aadt_pred.to_csv(congestion_weather_output, index=False)

print("Traffic forecasts updated with today's weather!")
print(df_aadt_pred.head())

'''
# Load your traffic forecast CSV (AADT predictions)
df_aadt_pred = pd.read_csv(pred_congest_csv)

# Load 7-day weather forecast
df_weather = pd.read_csv(weather_csv)

# --- Preprocess weather ---
# Simplify wind speed to numeric (extract number)
df_weather["windSpeed"] = df_weather["windSpeed"].str.extract(r'(\d+)').astype(float)

# Format date nicely
df_weather['Date'] = pd.to_datetime(df_weather['startTime']).dt.date

# Only keep necessary columns
df_weather_daily = df_weather[['Date', 'temperature', 'windSpeed', 'shortForecast']]

# --- Cycle the forecast dates across all sites ---

# Get the unique 7 dates
forecast_dates = df_weather_daily['Date'].unique()

# Assign cyclically
df_aadt_pred['FORECAST_DATE'] = np.resize(forecast_dates, len(df_aadt_pred))

# --- Merge traffic + weather on FORECAST_DATE ---
df_final = pd.merge(df_aadt_pred, df_weather_daily, left_on="FORECAST_DATE", right_on="Date", how="left")

# Save the merged dataset
df_final.to_csv(congestion_weather_output, index=False)

print("Traffic forecasts merged with 7-day weather forecast!")
print(df_final.head())
'''

'''
# Load predicted AADT
df_aadt_pred = pd.read_csv(pred_congest_csv)

# Load annual weather data
df_weather = pd.read_csv(weather_csv)

# Preprocessing weather
df_weather['Date'] = pd.to_datetime(df_weather['startTime']).dt.date

# Aggregate weather by day (first record per day if multiple)
df_weather_daily = df_weather.groupby('Date').first().reset_index()

# Simplify wind speed to numeric (remove "mph" text if exists)
df_weather_daily["windSpeed"] = df_weather_daily["windSpeed"].str.extract(r'(\d+)').astype(float)

# Merge: Add weather to traffic by date matching
# Assume traffic forecast has dates for prediction
df_aadt_pred['FORECAST_DATE'] = pd.date_range(start=pd.Timestamp.today(), periods=len(df_aadt_pred), freq='D').date

# Merge datasets
df_combined = pd.merge(df_aadt_pred, df_weather_daily, left_on='FORECAST_DATE', right_on='Date', how='left')

# Save merged data
df_combined.to_csv(congestion_weather_output, index=False)

print("Weather merged into traffic forecast!")
print(df_combined.head())
'''
