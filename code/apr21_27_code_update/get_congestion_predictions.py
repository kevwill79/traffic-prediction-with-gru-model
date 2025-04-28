# -*- coding: utf-8 -*-
"""
Created on Thu Apr 24 22:18:33 2025

@author: kwill
"""
import pandas as pd

# Paths
main_path = "C:/Users/kwill/Desktop/grad_classes/Master_Project_cpsc_69100"

code_path = f"{main_path}/code/apr21_27_code_update"
pred_csv = f"{code_path}/okaloosa_aadt_prediction.csv"
pred_congest_csv = f"{code_path}/okaloosa_predictions_congestion.csv"

# Load forecasted AADT
df_forecast = pd.read_csv(pred_csv)

# Define classification function
def classify_congestion(aadt):
    if aadt < 5000:
        return "Low"
    elif 5000 <= aadt <= 15000:
        return "Medium"
    else:
        return "High"

# Apply classification
df_forecast["Congestion_Level"] = df_forecast["FORECASTED_AADT"].apply(classify_congestion)

# Save new CSV with congestion levels
df_forecast.to_csv(pred_congest_csv, index=False)

print("Congestion levels classified and saved!")
print(df_forecast.head())

