# -*- coding: utf-8 -*-
"""
Created on Wed Apr 23 19:15:00 2025

@author: kwill
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from gru import GRURegressor

# Paths
main_path = "C:/Users/kwill/Desktop/grad_classes/Master_Project_cpsc_69100"

code_path = f"{main_path}/code/wk6_updates"
csv = f"{code_path}/aadt_congestion_weather.csv"

# Load your cleaned and merged dataset
df = pd.read_csv(csv)

# Features
df["Congestion"] = pd.cut(df["FORECASTED_AADT"], bins=[0, 5000, 15000, float('inf')], labels=[0, 1, 2])

# Normalize features
features = df[["FORECASTED_AADT", "TEMPERATURE", "WINDSPEED"]]
scaler = MinMaxScaler()
features_scaled = scaler.fit_transform(features)

SEQ_LEN = 5
X, y = [], []
for i in range(len(features_scaled) - SEQ_LEN):
    X.append(features_scaled[i:i+SEQ_LEN])
    y.append(features_scaled[i+SEQ_LEN][0])  # AADT prediction target

X, y = np.array(X), np.array(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to tensor
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

# Initialize model
model = GRURegressor(input_size=3)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Train model
for epoch in range(500):
    model.train()
    optimizer.zero_grad()
    output = model(X_train)
    loss = criterion(output, y_train)
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# Save model
torch.save(model.state_dict(), "gru_regressor.pth")
print("Model trained and saved.")