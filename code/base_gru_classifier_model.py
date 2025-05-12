# -*- coding: utf-8 -*-
"""
Created on Sun Apr  6 21:33:03 2025

@author: kwill
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Apr  3 19:21:03 2025

@author: kwill
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler, KBinsDiscretizer
import PyPDF2
import re

#from extract_fdot_historical_data import extract_traffic_data
#from fdot_historical_time_series import get_model_training_data

# Extract text from the PDF file
def extract_traffic_data(pdf_path):
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    
    lines = text.split("\n")
    data = []
    for line in lines:
        parts = line.split()
        if len(parts) >= 7 and parts[0].isdigit():  # Ensuring the first part is a site number
            try:
                site = parts[0]
                aadt = int(re.sub(r'\D', '', parts[5]))  # Extracting numeric AADT value
                data.append([site, aadt])
            except ValueError:
                continue  # Skip lines with invalid data
    
    df = pd.DataFrame(data, columns=["Site", "AADT"])
    return df


# Load and preprocess data
pdf_path = "C:/Users/kwill/Desktop/grad_classes/Master_Project_cpsc_69100/3_57_CAADT.pdf"
df = extract_traffic_data(pdf_path)
#df = extract_traffic_data(pdf_path)
df["AADT"] = df["AADT"].astype(float)

# Define congestion levels (Low, Medium, High) 
num_classes = 3  # 3 classes: Low, Medium, High
binner = KBinsDiscretizer(n_bins=num_classes, encode='ordinal', strategy='quantile')
df["Congestion_Level"] = binner.fit_transform(df[["AADT"]]).astype(int)

# Normalize AADT for sequence input
scaler = MinMaxScaler()
df["AADT"] = scaler.fit_transform(df[["AADT"]])

# Prepare dataset for GRU
sequence_length = 5
def create_sequences(data, labels, seq_length):
    sequences, label_seq = [], []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i+seq_length])
        label_seq.append(labels[i+seq_length])
    return np.array(sequences), np.array(label_seq)

X, y = create_sequences(df["AADT"].values, df["Congestion_Level"].values, sequence_length)
X = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)
y = torch.tensor(y, dtype=torch.long)  # Classification labels

dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

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

# Model training
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
print("Congestion classification model trained and saved.")
