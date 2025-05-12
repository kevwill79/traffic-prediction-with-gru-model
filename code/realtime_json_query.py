# -*- coding: utf-8 -*-
"""
Created on Sat Apr 19 15:25:50 2025

@author: kwill
"""

import requests
import pandas as pd

# ArcGIS Feature Service REST endpoint
#url = "https://services7.arcgis.com/1wkJ0T6GTzZVXtL4/arcgis/rest/services/HourlyPollingDashboard_RealTime/FeatureServer/0/query"
url = "https://services1.arcgis.com/O1JpcwDW8sjYuddV/arcgis/rest/services/Real_Time_Traffic_Volume_and_Speed_Current_All_Directions_TDA/FeatureServer/0/query?where=1%3D1&outFields=ROADWAY,MILE_POST,COUNTY,SITE,COSITE,COUNTYNM,LOCALNAM,MONTH_,MNTHNUM,DAY_,YR,WEEKDAY,HOUR_,MEAS_DT,DIRECTION,CURVOL,HAVGVOL,PCT_DIFF,PCTDIFTX,CURAVSPD,MAXSPEEDR,MAXSPEEDL,SPPCTDF,LATITUDE,LNGITUDE,LBLVAL&outSR=4326&f=json"
#url = "https://services7.arcgis.com/1wkJ0T6GTzZVXtL4/arcgis/rest/services/HourlyPollingDashboard_RealTime/FeatureServer/0/query?where=1&outFields=*&outSR=4326&f=json"

# Parameters for the API query
#params = {
#    "where": "1=1",  # get all data
#    "outFields": "*",  # all fields
#    "f": "json"  # response format
#}

params = {
    "where": "1%3D1",
    "outFields": "ROADWAY,MILE_POST",
    "outSR": "4326",
    "f": "json"
}

#,COUNTY,SITE,COSITE,COUNTYNM,LOCALNAM,MONTH_,MNTHNUM,DAY_,YR,WEEKDAY,HOUR_,MEAS_DT,DIRECTION,CURVOL,HAVGVOL,PCT_DIFF,PCTDIFTX,CURAVSPD,MAXSPEEDR,MAXSPEEDL,SPPCTDF,LATITUDE,LNGITUDE,LBLVAL

# Make the GET request
response = requests.get(url, params=params)

if response.status_code == 200:
    data = response.json()
    features = data.get('features', [])
    records = [feature['attributes'] for feature in features]
    df = pd.DataFrame(records)
    print(df.head())
    print(f"Total records fetched: {len(df)}")
else:
    print(f"Error {response.status_code}: {response.text}")
