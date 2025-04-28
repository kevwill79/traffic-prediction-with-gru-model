# -*- coding: utf-8 -*-
"""
Created on Sun Apr 13 22:04:41 2025

@author: kwill
"""
import requests
import pandas as pd

# Okaloosa County, FL
lat, lon = 30.64, -86.59

# Use the lat and lon to get forecast for Okaloosa county
weather_url = f"https://api.weather.gov/points/{lat},{lon}"
response = requests.get(weather_url).json()
forecast_url = response['properties']['forecast']

# Forecast
forecast_data = requests.get(forecast_url).json()
periods = forecast_data['properties']['periods']

# Convert to DataFrame
df_weather = pd.DataFrame(periods)[['name', 'startTime', 'temperature', 'shortForecast', 'windSpeed']]
df_weather['startTime'] = pd.to_datetime(df_weather['startTime'])

# Save weather to csv
output_path = "C:/Users/kwill/Desktop/grad_classes/Master_Project_cpsc_69100/code/apr21_27_code_update/seven_day_forecast.csv"

df_weather.to_csv(output_path, index=False)

print(df_weather[:7])
