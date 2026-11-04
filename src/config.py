"""Configuration module for 2D Heat Equation PINN.

Defines all physics constants, neural network hyperparameters, training settings,
and filesystem paths. Centralizing these prevents hardcoded magic numbers.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class PINNConfig:
    """Hyperparameters and configuration settings for PINN training and evaluation."""

    # Physics parameters
    alpha: float = 0.01  # Thermal diffusivity coefficient (m^2/s)
    x_min: float = 0.0
    x_max: float = 1.0
    y_min: float = 0.0
    y_max: float = 1.0
    t_min: float = 0.0
    t_max: float = 1.0

    # Architecture
    input_dim: int = 3   # (x, y, t)
    hidden_dim: int = 64
    num_hidden_layers: int = 4
    output_dim: int = 1  # T(x, y, t)

    # Sampling per epoch
    n_collocation_pde: int = 5000  # Resampled interior points
    n_collocation_ic: int = 1000   # Initial condition points at t=0
    n_collocation_bc: int = 1000   # Boundary points across all 4 edges

    # Training parameters
    epochs: int = 5000
    learning_rate: float = 1e-3
    log_frequency: int = 100
    seed: int = 42
    device: str = "cpu"

    # Loss term weights (weighted sum in total loss)
    weight_pde: float = 1.0
    weight_ic: float = 20.0
    weight_bc: float = 20.0

    # Evaluation settings
    eval_grid_size: int = 50  # 50x50 spatial grid
    eval_times: List[float] = field(default_factory=lambda: [0.0, 0.25, 0.5, 1.0])

    # Visualization settings
    gif_frames: int = 40
    gif_fps: int = 10

    # File paths (relative to repository root)
    results_dir: Path = Path("results")
    model_path: Path = Path("results/pinn.pt")
    loss_curve_path: Path = Path("results/loss_curve.png")
    metrics_path: Path = Path("results/metrics.md")
    gif_path: Path = Path("results/heat_evolution.gif")
