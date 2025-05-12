# -*- coding: utf-8 -*-
"""
Created on Sun Apr 6 05:46:27 2025

@author: kwill
"""
import torch
import torch.nn as nn
from tqdm import tqdm  # For progress bar

# The GRU classification model - Right now it's only using historical data
from gru_classifier import GRUClassifier

# Returns a Dataloader object and the number of traffic congestion levels
from fdot_historical_time_series import get_model_training_data

# Model training
dataloader, num_classes = get_model_training_data()

model = GRUClassifier(input_size=4, hidden_size=64, output_size=num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 500
for epoch in range(epochs):
    model.train() #
    running_loss = 0.0 # 
    for seqs, labels in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"): #
    #for seqs, labels in dataloader:
        optimizer.zero_grad()
        output = model(seqs)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
    avg_loss = running_loss / len(dataloader)
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# TODO Add visualization for congestion levels

# Save the model
torch.save(model.state_dict(), "gru_traffic_congestion_model.pth")