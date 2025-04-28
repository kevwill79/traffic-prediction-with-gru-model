# -*- coding: utf-8 -*-
"""
Created on Mon Apr 28 19:13:40 2025

@author: kwill
"""
import pandas as pd
import matplotlib.pyplot as plt

main_path = "C:/Users/kwill/Desktop/grad_classes/Master_Project_cpsc_69100"

code_path = f"{main_path}/code/apr21_27_code_update"
csv_path = f"{code_path}/cleaned_okaloosa_historical_td.csv"
pred_csv = f"{code_path}/okaloosa_aadt_prediction.csv"
plots_path = f"{code_path}/plots.jpg"

# Load cleaned historical AADT
df_actual = pd.read_csv(csv_path)

# Load predicted AADT
df_predicted = pd.read_csv(pred_csv)

# Merge for plotting
# Important: We'll align YEAR_ and FORECAST_YEAR
# Create unified dataframe per site
plots_to_make = df_predicted["COSITE"].unique()[:5]  # Limit to first 5 sites for now

for site in plots_to_make:
    # Actual history
    df_site = df_actual[df_actual["COSITE"] == site].sort_values("YEAR_")
    years = df_site["YEAR_"].tolist()
    aadt_values = df_site["AADT"].tolist()

    # Predict
    forecast_row = df_predicted[df_predicted["COSITE"] == site].iloc[0]
    years.append(forecast_row["FORECAST_YEAR"])
    aadt_values.append(forecast_row["FORECASTED_AADT"])

    # Plot
    plt.figure(figsize=(8,4))
    plt.plot(years[:-1], aadt_values[:-1], marker='o', label="Actual AADT")
    plt.plot(years[-2:], aadt_values[-2:], marker='x', linestyle='--', label="Forecasted AADT", color='red')
    plt.title(f"Site {site} - {forecast_row['ROADWAY']}")
    plt.xlabel("Year")
    plt.ylabel("AADT")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()