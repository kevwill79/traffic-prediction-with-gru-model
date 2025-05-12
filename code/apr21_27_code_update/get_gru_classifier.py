# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 17:10:43 2025

@author: kwill
"""
import torch.nn as nn

# Define the same GRU model again (match training)
class GRUClassifier(nn.Module):
    def __init__(self, input_size=1, hidden_size=64):
        super(GRUClassifier, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        _, h_n = self.gru(x)
        return self.fc(h_n[-1])