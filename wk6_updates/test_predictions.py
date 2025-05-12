# -*- coding: utf-8 -*-
"""
Created on Sun May 4 14:52:23 2025

@author: kwill
"""

import requests

# Server URL
url = "http://127.0.0.1:5000/predict"

# Sample input data (5 years, 3 features: AADT, Temperature, Windspeed)
data = {
    "features": [
        [12000, 70, 5],
        [12500, 72, 6],
        [13000, 68, 4],
        [12800, 75, 5],
        [13500, 73, 4]
    ]
}

# Send the request
response = requests.post(url, json=data)

# Display the result
if response.status_code == 200:
    print("Prediction result:")
    print(response.json())
else:
    print("Error:")
    print(response.status_code, response.text)