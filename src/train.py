"""Training script for 2D Heat Equation Physics-Informed Neural Network (PINN).

Features:
- Adam optimizer with learning rate scheduler.
- Dynamic resampling of collocation points every epoch.
- Periodic logging of individual loss terms (PDE residual, IC, BC, Total).
- Checkpointing model weights to results/pinn.pt.
- Generating training loss convergence curve to results/loss_curve.png.
"""

import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.optim as optim

# Ensure project root is in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import PINNConfig
from src.model import HeatPINN
from src.physics_loss import compute_total_loss, sample_collocation_points
from src.visualize import plot_loss_curve


def train_pinn(config: PINNConfig) -> Dict[str, List[float]]:
    """Train the PINN model according to the provided configuration.

    Args:
        config: Dataclass instance containing all hyperparameters.

    Returns:
        Dictionary containing recorded loss history per logged interval.
    """
    # 1. Reproducibility
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    device = torch.device(config.device)
    print(f"Starting training on device: {device} with random seed: {config.seed}")
    print(
        f"Hyperparameters: {config.epochs} epochs, lr={config.learning_rate}, "
        f"alpha={config.alpha}, n_collocation_pde={config.n_collocation_pde}"
    )

    # 2. Instantiate Model and Optimizer
    model = HeatPINN(
        input_dim=config.input_dim,
        hidden_dim=config.hidden_dim,
        num_hidden_layers=config.num_hidden_layers,
        output_dim=config.output_dim,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    # Cosine annealing scheduler helps fine-tune gradients towards target relative L2 error
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs, eta_min=1e-5
    )

    loss_history: Dict[str, List[float]] = {
        "epochs": [],
        "loss_total": [],
        "loss_pde": [],
        "loss_ic": [],
        "loss_bc": [],
    }

    start_time = time.time()
    config.results_dir.mkdir(parents=True, exist_ok=True)

    # 3. Main Optimization Loop
    for epoch in range(1, config.epochs + 1):
        model.train()
        optimizer.zero_grad()

        # Resample collocation points fresh every epoch
        x_pde, y_pde, t_pde, ic_pts, bc_pts = sample_collocation_points(config, device)

        # Compute multi-term physics loss
        total_loss, loss_dict = compute_total_loss(
            model=model,
            x_pde=x_pde,
            y_pde=y_pde,
            t_pde=t_pde,
            ic_pts=ic_pts,
            bc_pts=bc_pts,
            config=config,
        )

        # Backpropagation
        total_loss.backward()
        optimizer.step()
        scheduler.step()

        # Periodic logging
        if epoch == 1 or epoch % config.log_frequency == 0 or epoch == config.epochs:
            loss_history["epochs"].append(epoch)
            loss_history["loss_total"].append(loss_dict["loss_total"])
            loss_history["loss_pde"].append(loss_dict["loss_pde"])
            loss_history["loss_ic"].append(loss_dict["loss_ic"])
            loss_history["loss_bc"].append(loss_dict["loss_bc"])

            current_lr = scheduler.get_last_lr()[0]
            elapsed = time.time() - start_time
            print(
                f"[Epoch {epoch:5d}/{config.epochs:5d}] "
                f"Total: {loss_dict['loss_total']:.6f} | "
                f"PDE: {loss_dict['loss_pde']:.6f} | "
                f"IC: {loss_dict['loss_ic']:.6f} | "
                f"BC: {loss_dict['loss_bc']:.6f} | "
                f"lr: {current_lr:.2e} | "
                f"Time: {elapsed:.1f}s"
            )

    total_training_time = time.time() - start_time
    print(f"\nTraining completed in {total_training_time:.2f} seconds.")

    # 4. Save Model Checkpoint
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config,
            "final_losses": {
                k: loss_history[k][-1] for k in ["loss_total", "loss_pde", "loss_ic", "loss_bc"]
            },
            "training_time_seconds": total_training_time,
        },
        config.model_path,
    )
    print(f"Model saved to: {config.model_path}")

    # 5. Plot Loss Curves
    plot_loss_curve(loss_history, config.loss_curve_path)

    return loss_history


if __name__ == "__main__":
    cfg = PINNConfig()
    train_pinn(cfg)

