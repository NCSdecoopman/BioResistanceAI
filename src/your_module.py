# your_module.py

import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim=100, hidden_dim=200, output_dim=2):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, X):
        return self.model(X)
