"""Visualization utilities for 2D Heat Equation PINN.

Provides functions to:
1. Plot and save multi-term training loss curves.
2. Generate a 3-panel GIF showing the PINN prediction, exact analytical solution,
   and absolute error map over time.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server/script execution
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import torch
import torch.nn as nn

# Ensure project root is in sys.path for direct script execution
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import PINNConfig


def exact_solution_np(x: np.ndarray, y: np.ndarray, t: float, alpha: float) -> np.ndarray:
    """Compute exact analytical solution: sin(pi*x) * sin(pi*y) * exp(-2*pi^2*alpha*t).

    Args:
        x: 2D array of x coordinates.
        y: 2D array of y coordinates.
        t: Time value.
        alpha: Thermal diffusivity.

    Returns:
        2D array of exact temperatures.
    """
    decay = np.exp(-2.0 * (np.pi ** 2) * alpha * t)
    return np.sin(np.pi * x) * np.sin(np.pi * y) * decay


def plot_loss_curve(
    loss_history: Dict[str, List[float]],
    save_path: Path,
) -> None:
    """Plot and save training loss curves on a logarithmic scale.

    Args:
        loss_history: Dictionary containing lists of losses per logged epoch
                      ('epochs', 'loss_total', 'loss_pde', 'loss_ic', 'loss_bc').
        save_path: Destination image path (e.g., results/loss_curve.png).
    """
    epochs = loss_history["epochs"]
    plt.figure(figsize=(9, 6), dpi=150)

    plt.semilogy(epochs, loss_history["loss_total"], label="Total Loss", color="#1f77b4", linewidth=2.0)
    plt.semilogy(epochs, loss_history["loss_pde"], label="PDE Residual Loss", color="#ff7f0e", linestyle="--")
    plt.semilogy(epochs, loss_history["loss_ic"], label="Initial Condition Loss", color="#2ca02c", linestyle="-.")
    plt.semilogy(epochs, loss_history["loss_bc"], label="Boundary Condition Loss", color="#d62728", linestyle=":")

    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss (MSE, log scale)", fontsize=12)
    plt.title("2D Heat Equation PINN - Training Convergence", fontsize=14, fontweight="bold")
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.legend(fontsize=11, loc="upper right")
    plt.tight_layout()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Loss curve saved to: {save_path}")


def generate_heat_evolution_gif(
    model: nn.Module,
    config: PINNConfig,
    device: torch.device,
    save_path: Path,
    n_grid: int = 100,
    n_frames: int = 40,
    fps: int = 10,
) -> None:
    """Generate side-by-side animated GIF of PINN vs Exact vs Error.

    Three side-by-side panels:
        1. PINN Predicted Temperature T_pred(x, y, t)
        2. Exact Analytical Solution T_exact(x, y, t)
        3. Absolute Error |T_pred - T_exact|

    Features:
        - Fixed inferno colormap and shared colorbar for PINN and exact [0, 1]
        - Viridis colormap for absolute error
        - Current time displayed prominently in the figure title

    Args:
        model: Trained PINN model.
        config: Problem configuration (for alpha and domain bounds).
        device: Torch device model resides on.
        save_path: Path to output .gif file.
        n_grid: Spatial resolution per dimension (n_grid x n_grid).
        n_frames: Number of temporal frames over [t_min, t_max].
        fps: Frames per second for the animation.
    """
    model.eval()

    # Create spatial grid
    x_vals = np.linspace(config.x_min, config.x_max, n_grid)
    y_vals = np.linspace(config.y_min, config.y_max, n_grid)
    X, Y = np.meshgrid(x_vals, y_vals)
    t_vals = np.linspace(config.t_min, config.t_max, n_frames)

    frames = []

    # Flatten spatial grid for torch model evaluation
    x_flat = torch.from_numpy(X.flatten().astype(np.float32)).unsqueeze(1).to(device)
    y_flat = torch.from_numpy(Y.flatten().astype(np.float32)).unsqueeze(1).to(device)

    print(f"Generating {n_frames} animation frames for {save_path}...")

    with torch.no_grad():
        for i, t in enumerate(t_vals):
            t_flat = torch.full_like(x_flat, float(t))
            xyt = torch.cat([x_flat, y_flat, t_flat], dim=1)

            # Predict PINN solution
            pred_flat = model(xyt).cpu().numpy().flatten()
            T_pinn = pred_flat.reshape((n_grid, n_grid))

            # Exact solution
            T_exact = exact_solution_np(X, Y, t, config.alpha)

            # Absolute error
            abs_error = np.abs(T_pinn - T_exact)

            # Render frame
            fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=100)

            # 1. PINN Prediction (fixed [0, 1] inferno)
            im0 = axes[0].imshow(
                T_pinn,
                extent=[config.x_min, config.x_max, config.y_min, config.y_max],
                origin="lower",
                cmap="inferno",
                vmin=0.0,
                vmax=1.0,
            )
            axes[0].set_title("PINN Prediction $T_{pred}$", fontsize=11, fontweight="bold")
            axes[0].set_xlabel("x")
            axes[0].set_ylabel("y")
            fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

            # 2. Exact Analytical Solution (fixed [0, 1] inferno)
            im1 = axes[1].imshow(
                T_exact,
                extent=[config.x_min, config.x_max, config.y_min, config.y_max],
                origin="lower",
                cmap="inferno",
                vmin=0.0,
                vmax=1.0,
            )
            axes[1].set_title("Exact Solution $T_{exact}$", fontsize=11, fontweight="bold")
            axes[1].set_xlabel("x")
            axes[1].set_ylabel("y")
            fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

            # 3. Absolute Error
            max_err_frame = max(float(np.max(abs_error)), 0.01)
            im2 = axes[2].imshow(
                abs_error,
                extent=[config.x_min, config.x_max, config.y_min, config.y_max],
                origin="lower",
                cmap="viridis",
                vmin=0.0,
                vmax=max_err_frame,
            )
            axes[2].set_title(f"Absolute Error (max: {np.max(abs_error):.2e})", fontsize=11, fontweight="bold")
            axes[2].set_xlabel("x")
            axes[2].set_ylabel("y")
            fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

            fig.suptitle(f"2D Heat Conduction Evolution | Time t = {t:.3f} s (frame {i+1}/{n_frames})", fontsize=13, fontweight="bold")
            plt.tight_layout()

            # Canvas to numpy RGB array
            fig.canvas.draw()
            rgba = np.asarray(fig.canvas.buffer_rgba())
            rgb = rgba[:, :, :3]
            frames.append(rgb)
            plt.close(fig)

    # Save animated GIF
    save_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(save_path, frames, fps=fps, loop=0)
    print(f"Heat evolution GIF saved successfully to: {save_path}")


if __name__ == "__main__":
    from src.model import HeatPINN

    cfg = PINNConfig()
    dev = torch.device(cfg.device)

    if not cfg.model_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at {cfg.model_path}. Train the model first via src/train.py"
        )

    checkpoint = torch.load(cfg.model_path, map_location=dev, weights_only=False)
    net = HeatPINN(
        input_dim=cfg.input_dim,
        hidden_dim=cfg.hidden_dim,
        num_hidden_layers=cfg.num_hidden_layers,
        output_dim=cfg.output_dim,
    ).to(dev)
    net.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded checkpoint from: {cfg.model_path}")

    generate_heat_evolution_gif(
        model=net,
        config=cfg,
        device=dev,
        save_path=cfg.gif_path,
        n_grid=50,
        n_frames=cfg.gif_frames,
        fps=cfg.gif_fps,
    )
