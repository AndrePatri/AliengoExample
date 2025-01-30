import torch
import torch.nn as nn
import torch.nn.functional as F

class RNDNetwork(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, 
                 layer_width: int = 128, n_hidden_layers: int = 2, target: bool = False):
        """
        Random Network Distillation (RND) model.

        :param input_dim: Dimension of the input (obs + action)
        :param output_dim: Dimension of the output feature space
        :param layer_width: Number of neurons per hidden layer
        :param n_hidden_layers: Number of hidden layers
        :param target: If True, this network acts as the fixed target network (no gradient updates)
        """
        super().__init__()
        
        layers = [nn.Linear(input_dim, layer_width), nn.ReLU()]
        
        for _ in range(n_hidden_layers):
            layers.append(nn.Linear(layer_width, layer_width))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(layer_width, output_dim))

        self.network = nn.Sequential(*layers)

        if target:
            for param in self.parameters():
                param.requires_grad = False  # Freeze target network

    def forward(self, x):
        return self.network(x)
