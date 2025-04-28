# -*- coding: utf-8 -*-
"""
Created on Mon Mar 31 19:21:03 2025

@author: kwill
"""
import torch.nn as nn

# Define GRU Classification Model
class GRUClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        super(GRUClassifier, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        out, _ = self.gru(x)
        out = self.fc(out[:, -1, :])
        return out
    