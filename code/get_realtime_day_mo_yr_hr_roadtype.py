# -*- coding: utf-8 -*-
"""
Created on Thu Apr 17 23:42:20 2025

@author: kwill
"""
import pandas as pd

# Get all data from real-time csv
df = pd.read_csv("C:/Users/kwill/Desktop/grad_classes/Master_Project_cpsc_69100/fdot_realtime_traffic_data/Real_Time_Traffic_Volume_and_Speed_All_Intervals_All_Directions_TDA.csv")

# Filter for Okaloosa county data only
df_okaloosa = df[df["COUNTYNM"].str.contains("OKALOOSA")].copy()

# Filter for year, cosite, roadway, 
df_okaloosa_clean = df_okaloosa[["ROADWAY", "SITE", "COSITE", "LOCALNAM", "COUNTYNM", 
                                 "MONTH_", "MNTHNUM", "DAY_", "YR", "WEEKDAY", 
                                 "HOUR_", "CURAVSPD", "MAXSPEEDR", "MAXSPEEDL"]]