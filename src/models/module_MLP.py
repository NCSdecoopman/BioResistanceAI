# module_MLP.py

import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim=2, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x):
        return self.net(x)
