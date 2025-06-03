# src/models/module_MLP.py

import torch
import torch.nn as nn

# Bloc résiduel classique (type ResNet), adapté à un MLP
class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.3):
        super().__init__()

        # Le bloc séquentiel inclut deux couches linéaires, deux batchnorm, GELU et dropout
        self.block = nn.Sequential(
            nn.Linear(dim, dim),         # Première couche linéaire (dim -> dim)
            nn.BatchNorm1d(dim),         # Normalisation de lot
            nn.GELU(),                   # Activation non linéaire
            nn.Dropout(dropout),         # Dropout régularisant
            nn.Linear(dim, dim),         # Deuxième couche linéaire
            nn.BatchNorm1d(dim)          # BatchNorm de sortie
        )
        self.activation = nn.GELU()      # Activation finale (appliquée après addition résiduelle)

    def forward(self, x):
        # Ajoute la sortie du bloc à l'entrée, puis applique l'activation
        return self.activation(x + self.block(x))

# MLP classique avec deux blocs résiduels pour le feature learning
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim=2, hidden_dim=128, dropout=0.3):
        super().__init__()
        # Réseau séquentiel :
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),       # Entrée → caché
            nn.BatchNorm1d(hidden_dim),             # Normalisation
            nn.GELU(),                             # Activation
            nn.Dropout(dropout),                   # Dropout
            ResidualBlock(hidden_dim, dropout),    # Premier bloc résiduel
            ResidualBlock(hidden_dim, dropout),    # Deuxième bloc résiduel
            nn.Linear(hidden_dim, output_dim)      # Sortie finale (classif ou régression)
        )

    def forward(self, x):
        return self.net(x)
