import torch.nn as nn

# Define the same GRU model again (match training)
class GRURegressor(nn.Module):
    def __init__(self, input_size=4, hidden_size=64):
        super(GRURegressor, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        _, h_n = self.gru(x)
        return self.fc(h_n[-1])