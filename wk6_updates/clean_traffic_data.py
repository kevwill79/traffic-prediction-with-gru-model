# -*- coding: utf-8 -*-
"""
Created on TUE Apr 22 18:11:39 2025

@author: kwill
"""
import pandas as pd

# Paths
main_path = "C:/Users/kwill/Desktop/grad_classes/Master_Project_cpsc_69100/"

csv_path = f"{main_path}/fdot_historical_traffic_data/Annual_Average_Daily_Traffic_Historical_TDA.csv"

# Load the dataset
df_all = pd.read_csv(csv_path, on_bad_lines="skip")

# Filter for Okaloosa County
df_okaloosa = df_all[df_all["COUNTY"].str.contains("OKALOOSA", case=False, na=False)].copy()

# Rename columns
df_okaloosa_clean = df_okaloosa[[

    "YEAR_", "COSITE", "ROADWAY", "DESC_FRM", "DESC_TO", "AADT", "COUNTY"
]].copy()

# Drop missing values
df_okaloosa_clean.dropna(subset=["AADT", "YEAR_", "COSITE"], inplace=True)

# Convert AADT to number
df_okaloosa_clean["AADT"] = pd.to_numeric(df_okaloosa_clean["AADT"], errors="coerce")

# Drop rows where AADT could not be converted
df_okaloosa_clean.dropna(subset=["AADT"], inplace=True)

# Sort by Roadway and Year
df_okaloosa_clean = df_okaloosa_clean.sort_values(by=["ROADWAY", "YEAR_"]).reset_index(drop=True)

# Drop duplicate rows found 05/04/2025
df_okaloosa_clean['DESC_FRM'] = df_okaloosa_clean['DESC_FRM'].fillna('NA')
df_okaloosa_clean['DESC_TO'] = df_okaloosa_clean['DESC_TO'].fillna('NA')
df_okaloosa_clean.drop_duplicates(subset=None, keep='first', inplace=False, ignore_index=False)

# Save cleaned Okaloosa dataset
output_path = "C:/Users/kwill/Desktop/grad_classes/Master_Project_cpsc_69100/code/wk6_updates/cleaned_okaloosa_historical_td.csv"
df_okaloosa_clean.to_csv(output_path, index=False)

print(f"Cleaned Okaloosa data saved at {output_path}")
print(df_okaloosa_clean[20:29])