# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 17:19:54 2025

@author: kwill
"""
import torch
from gru_classifier import GRUClassifier

# Load model again (if needed)
model = GRUClassifier(input_size=3, hidden_size=64, output_size=3)
model.load_state_dict(torch.load("C:/Users/kwill/Desktop/grad_classes/Master_Project_cpsc_69100/code/gru_traffic_congestion_model.pth"))
model.eval()

# Create example input (shape: batch=1, seq_len=5, features=3)
example_input = torch.rand(1, 5, 3)
traced_model = torch.jit.trace(model, example_input)

# Save TorchScript model
traced_model.save("gru_classifier_mobile.pt")
print("Saved for mobile!")