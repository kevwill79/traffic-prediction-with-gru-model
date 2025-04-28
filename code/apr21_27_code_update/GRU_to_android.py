# -*- coding: utf-8 -*-
"""
Created on Fri Apr 25 18:55:45 2025

@author: kwill
"""
import torch
import torch.nn as nn
from get_gru_classifier import GRUClassifier

'''
# Define the same GRU model again (match training)
class GRURegressor(nn.Module):
    def __init__(self, input_size=4, hidden_size=64):
        super(GRURegressor, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        _, h_n = self.gru(x)
        return self.fc(h_n[-1])
'''
# Initialize model with correct settings
input_size = 4  # AADT, Temperature, Precipitation, Wind
model = GRURegressor()

# (IMPORTANT) Load your trained weights
model.load_state_dict(torch.load("/mnt/data/gru_retrained_weather.pth"))
model.eval()

# Create dummy input (shape must match training)
# batch_size = 1, sequence_length = 3, input_features = 4
example_input = torch.rand(1, 3, 4)

# Trace the model
traced_model = torch.jit.trace(model, example_input)

# Save TorchScript model
traced_model.save("/mnt/data/gru_weather_mobile.pt")

print("Mobile-ready model saved as 'gru_weather_mobile.pt'")

