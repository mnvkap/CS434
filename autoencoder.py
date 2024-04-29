import torch

import torch.nn as nn
import torch.optim as optim


# Define your neural network architecture
class Autoencoder(nn.Module):
    def __init__(self, input_size, hidden_size, input_layers, output_layers):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(input_size, hidden_size), nn.ReLU())
        self.decoder = nn.Sequential(nn.Linear(hidden_size, input_size), nn.Sigmoid())

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
