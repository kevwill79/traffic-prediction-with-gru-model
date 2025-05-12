# -*- coding: utf-8 -*-
"""
Created on Fri Apr 18 19:50:20 2025

@author: kwill
"""
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

df = pd.read_csv("C:/Users/kwill/Desktop/grad_classes/Master_Project_cpsc_69100/fdot_historical_traffic_data/Annual_Average_Daily_Traffic_Historical_TDA.csv")
df_okaloosa = df[df["COUNTY"].str.contains("Okaloosa")].copy()

# Choose one site to model (or loop through multiple later)
#site_id = df_okaloosa["COSITE"].value_counts().index[0]  # Most common site

# Chose a known traffic-heavy site for testing
site_id = 577701
df_site = df_okaloosa[df_okaloosa["COSITE"] == site_id].sort_values("YEAR_")

# Extract AADT time series
aadt_values = df_site["AADT"].values.reshape(-1, 1)

# Normalize
scaler = MinMaxScaler()
aadt_scaled = scaler.fit_transform(aadt_values)

# Create sequences
def create_sequences(data, seq_length=3):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

SEQ_LEN = 3
X, y = create_sequences(aadt_scaled, SEQ_LEN)

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

# Define GRU regressor
class GRURegressor(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, num_layers=1):
        super(GRURegressor, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        _, h_n = self.gru(x)
        return self.fc(h_n[-1])

# Initialize model
model = GRURegressor()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Train model
epochs = 500
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    output = model(X)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()

    if (epoch+1) % 20 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")

# Forecast next AADT value
with torch.no_grad():
    last_seq = torch.tensor(aadt_scaled[-SEQ_LEN:], dtype=torch.float32).unsqueeze(0)
    forecast_scaled = model(last_seq).item()
    forecast = scaler.inverse_transform([[forecast_scaled]])[0][0]
    print(f"Forecasted AADT for next year: {forecast:.2f}")

# Visualize
with torch.no_grad():
    predictions = model(X).numpy()
    predictions = scaler.inverse_transform(predictions).flatten()
    actual = scaler.inverse_transform(y.numpy()).flatten()

plt.plot(actual, label="Actual AADT")
plt.plot(predictions, label="Predicted AADT")
plt.title(f"AADT Forecast for Site {site_id}")
plt.xlabel("Year")
plt.ylabel("AADT")
plt.legend()
plt.tight_layout()
plt.grid(True)
plt.show()