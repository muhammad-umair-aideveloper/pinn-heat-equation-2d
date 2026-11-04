"""Physics-Informed Neural Network (PINN) architecture for 2D Heat Equation.

Defines a Multi-Layer Perceptron (MLP) taking spatio-temporal coordinates (x, y, t)
and predicting temperature T(x, y, t).
"""

import torch
import torch.nn as nn


class HeatPINN(nn.Module):
    """Multi-Layer Perceptron for solving 2D transient heat conduction.

    Architecture:
        - Input: 3 features (x, y, t)
        - 4 hidden layers of 64 units each with tanh activation
        - Output: 1 feature T(x, y, t)
        - Xavier uniform initialization on weights, zeros on biases
    """

    def __init__(
        self,
        input_dim: int = 3,
        hidden_dim: int = 64,
        num_hidden_layers: int = 4,
        output_dim: int = 1,
    ) -> None:
        """Initialize HeatPINN layers and apply Xavier initialization.

        Args:
            input_dim: Dimension of input space (default 3: x, y, t).
            hidden_dim: Number of neurons per hidden layer (default 64).
            num_hidden_layers: Number of fully connected hidden layers (default 4).
            output_dim: Dimension of output (default 1: temperature T).
        """
        super().__init__()
        layers = []

        # First hidden layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.Tanh())

        # Intermediate hidden layers
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())

        # Output linear layer (no activation function to allow any real value)
        layers.append(nn.Linear(hidden_dim, output_dim))

        self.network = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self) -> None:
        """Apply Xavier uniform initialization to weights and zero out biases."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, xyt: torch.Tensor) -> torch.Tensor:
        """Forward pass predicting temperature field.

        Args:
            xyt: Tensor of shape (N, 3) containing coordinates [x, y, t].

        Returns:
            Tensor of shape (N, 1) representing predicted temperature T.
        """
        return self.network(xyt)

