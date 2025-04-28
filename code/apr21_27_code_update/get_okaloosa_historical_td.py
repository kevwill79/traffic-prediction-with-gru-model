# -*- coding: utf-8 -*-
"""
Created on TUE Apr 22 18:11:39 2025

@author: kwill
"""
import pandas as pd

# Paths
main_path = "C:/Users/kwill/Desktop/grad_classes/Master_Project_cpsc_69100/"

csv_path = f"{main_path}/fdot_historical_traffic_data/Annual_Average_Daily_Traffic_Historical_TDA.csv"

'''
# Load the new file with safe settings
df_all = pd.read_csv(csv_path, on_bad_lines="skip")

# Filter Okaloosa County
df_okaloosa = df_all[df_all["COUNTY"].str.contains("OKALOOSA", case=False, na=False)].copy()

# Quick preview
print(df_okaloosa.head())
'''

# --- Step 1: Load the dataset safely ---
df_all = pd.read_csv(csv_path, on_bad_lines="skip")

# --- Step 2: Filter for Okaloosa County ---
df_okaloosa = df_all[df_all["COUNTY"].str.contains("OKALOOSA", case=False, na=False)].copy()

# --- Step 3: Select and rename important columns ---
df_okaloosa_clean = df_okaloosa[[
    "YEAR_", "COSITE", "ROADWAY", "DESC_FRM", "DESC_TO", "AADT", "COUNTY"
]].copy()

# --- Step 4: Drop missing or corrupted values ---
df_okaloosa_clean.dropna(subset=["AADT", "YEAR_", "COSITE"], inplace=True)

# --- Step 5: Convert AADT to numeric if it's string type ---
df_okaloosa_clean["AADT"] = pd.to_numeric(df_okaloosa_clean["AADT"], errors="coerce")

# Drop rows where AADT could not be converted
df_okaloosa_clean.dropna(subset=["AADT"], inplace=True)

# --- Step 6: Sort by Site and Year (for Time Series modeling) ---
df_okaloosa_clean = df_okaloosa_clean.sort_values(by=["COSITE", "YEAR_"]).reset_index(drop=True)

# --- Step 7: Save cleaned Okaloosa dataset ---
output_path = "C:/Users/kwill/Desktop/grad_classes/Master_Project_cpsc_69100/code/apr21_27_code_update/cleaned_okaloosa_historical_td.csv"
df_okaloosa_clean.to_csv(output_path, index=False)

print(f"Cleaned Okaloosa data saved at {output_path}")
print(df_okaloosa_clean.head())
