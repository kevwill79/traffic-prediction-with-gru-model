# -*- coding: utf-8 -*-
"""
Created on Tue Apr 1 21:46:27 2025

@author: kwill
"""
import torch
import torch.nn as nn

# The GRU classification model - Right now it's only using historical data
from gru_classifier import GRUClassifier

# Returns a Dataloader object and the number of traffic congestion levels
from fdot_historical_time_series import get_model_training_data

# Model training
dataloader, num_classes = get_model_training_data()

model = GRUClassifier(input_size=1, hidden_size=64, output_size=num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 500
for epoch in range(epochs):
    for seqs, labels in dataloader:
        optimizer.zero_grad()
        output = model(seqs)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# Save the model
torch.save(model.state_dict(), "gru_traffic_congestion_model.pth")