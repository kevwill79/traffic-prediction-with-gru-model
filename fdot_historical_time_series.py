# -*- coding: utf-8 -*-
"""
Created on Thu Apr 3 20:06:24 2025

@author: kwill
"""
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler, KBinsDiscretizer

from extract_fdot_historical_data import extract_traffic_data

# Extract historical data from fdot (Florida Dept of Transportation) site
pdf_path = "C:/Users/kwill/Desktop/grad_classes/Master_Project_cpsc_69100/3_57_CAADT.pdf"

def process_data(data_path):
    # Load and preprocess data
    df = extract_traffic_data(data_path)
    df["AADT"] = df["AADT"].astype(float)
    
    # Traffic congestion levels (Low, Medium, High) 
    num_classes = 3  # 3 classes: Low, Medium, High
    binner = KBinsDiscretizer(n_bins=num_classes, encode='ordinal', strategy='quantile')
    df["Congestion_Level"] = binner.fit_transform(df[["AADT"]]).astype(int)
    
    # Normalize AADT for sequence input
    scaler = MinMaxScaler()
    df["AADT"] = scaler.fit_transform(df[["AADT"]])
    
    return df, num_classes

# Prepare dataset for GRU
sequence_length = 5
def create_sequences(data, labels, seq_length):
    sequences, label_seq = [], []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i+seq_length])
        label_seq.append(labels[i+seq_length])
    return np.array(sequences), np.array(label_seq)

def get_model_training_data():
    df, num_classes = process_data(pdf_path)
    
    X, y = create_sequences(df["AADT"].values, df["Congestion_Level"].values, sequence_length)
    X = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)
    y = torch.tensor(y, dtype=torch.long)  # Classification labels
    
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    return dataloader, num_classes