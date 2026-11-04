"""Physics-informed loss calculation for 2D Heat Equation.

Computes PDE residual using PyTorch autograd (with create_graph=True),
along with initial condition (IC) and Dirichlet boundary condition (BC) losses.
"""

import math
import sys
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn

# Ensure project root is in sys.path for direct script execution
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import PINNConfig


def sample_collocation_points(
    config: PINNConfig,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample collocation points for PDE interior, initial condition, and boundary condition.

    Args:
        config: Configuration containing sample counts and domain bounds.
        device: Target torch device (e.g., 'cpu').

    Returns:
        Tuple of:
            - x_pde, y_pde, t_pde: Interior coordinates with requires_grad=True
            - ic_pts: Shape (N_ic, 3) at t=0
            - bc_pts: Shape (N_bc, 3) along the 4 spatial boundaries
    """
    n_pde = config.n_collocation_pde
    n_ic = config.n_collocation_ic
    n_bc = config.n_collocation_bc

    # 1. Interior collocation points (x, y, t) in [0, 1]^3
    x_pde = torch.rand(n_pde, 1, device=device, requires_grad=True)
    y_pde = torch.rand(n_pde, 1, device=device, requires_grad=True)
    t_pde = torch.rand(n_pde, 1, device=device, requires_grad=True)

    # 2. Initial condition points (x, y in [0, 1], t = 0)
    x_ic = torch.rand(n_ic, 1, device=device)
    y_ic = torch.rand(n_ic, 1, device=device)
    t_ic = torch.zeros(n_ic, 1, device=device)
    ic_pts = torch.cat([x_ic, y_ic, t_ic], dim=1)

    # 3. Boundary condition points (4 edges, each gets n_bc // 4 points)
    n_edge = n_bc // 4
    # Edge 1: x = 0, y in [0, 1], t in [0, 1]
    e1 = torch.cat([
        torch.zeros(n_edge, 1, device=device),
        torch.rand(n_edge, 1, device=device),
        torch.rand(n_edge, 1, device=device),
    ], dim=1)
    # Edge 2: x = 1, y in [0, 1], t in [0, 1]
    e2 = torch.cat([
        torch.ones(n_edge, 1, device=device),
        torch.rand(n_edge, 1, device=device),
        torch.rand(n_edge, 1, device=device),
    ], dim=1)
    # Edge 3: y = 0, x in [0, 1], t in [0, 1]
    e3 = torch.cat([
        torch.rand(n_edge, 1, device=device),
        torch.zeros(n_edge, 1, device=device),
        torch.rand(n_edge, 1, device=device),
    ], dim=1)
    # Edge 4: y = 1, x in [0, 1], t in [0, 1]
    e4 = torch.cat([
        torch.rand(n_edge, 1, device=device),
        torch.ones(n_edge, 1, device=device),
        torch.rand(n_edge, 1, device=device),
    ], dim=1)

    bc_pts = torch.cat([e1, e2, e3, e4], dim=0)

    return x_pde, y_pde, t_pde, ic_pts, bc_pts


def compute_pde_residual_loss(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    t: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    """Compute mean squared error of the PDE residual: dT/dt - alpha * (d2T/dx2 + d2T/dy2).

    Uses torch.autograd.grad with create_graph=True to maintain the computational
    graph for higher-order derivatives and backpropagation.

    Args:
        model: PINN network predicting T given [x, y, t].
        x: Spatial x coordinates with requires_grad=True (N, 1).
        y: Spatial y coordinates with requires_grad=True (N, 1).
        t: Temporal coordinates with requires_grad=True (N, 1).
        alpha: Thermal diffusivity coefficient.

    Returns:
        Scalar MSE loss of PDE residual.
    """
    xyt = torch.cat([x, y, t], dim=1)
    T = model(xyt)

    # First derivatives computed in a single graph traversal
    grad_outputs = torch.ones_like(T)
    T_x, T_y, T_t = torch.autograd.grad(
        T, (x, y, t), grad_outputs=grad_outputs, create_graph=True
    )

    # Second spatial derivatives
    T_xx = torch.autograd.grad(T_x, x, grad_outputs=torch.ones_like(T_x), create_graph=True)[0]
    T_yy = torch.autograd.grad(T_y, y, grad_outputs=torch.ones_like(T_y), create_graph=True)[0]

    # Heat equation residual: R = dT/dt - alpha * (d2T/dx2 + d2T/dy2)
    residual = T_t - alpha * (T_xx + T_yy)

    return torch.mean(residual ** 2)


def compute_ic_loss(
    model: nn.Module,
    ic_pts: torch.Tensor,
) -> torch.Tensor:
    """Compute initial condition loss at t = 0: T(x, y, 0) = sin(pi*x) * sin(pi*y).

    Args:
        model: PINN network.
        ic_pts: Coordinates [x, y, t=0] of shape (N, 3).

    Returns:
        Scalar MSE loss between predicted T and exact IC.
    """
    T_pred = model(ic_pts)
    x = ic_pts[:, 0:1]
    y = ic_pts[:, 1:2]
    T_exact_ic = torch.sin(math.pi * x) * torch.sin(math.pi * y)

    return torch.mean((T_pred - T_exact_ic) ** 2)


def compute_bc_loss(
    model: nn.Module,
    bc_pts: torch.Tensor,
) -> torch.Tensor:
    """Compute Dirichlet boundary condition loss on all four edges: T = 0.

    Args:
        model: PINN network.
        bc_pts: Coordinates along boundary edges of shape (N, 3).

    Returns:
        Scalar MSE loss penalizing deviations from zero at boundaries.
    """
    T_pred_bc = model(bc_pts)
    return torch.mean(T_pred_bc ** 2)


def compute_total_loss(
    model: nn.Module,
    x_pde: torch.Tensor,
    y_pde: torch.Tensor,
    t_pde: torch.Tensor,
    ic_pts: torch.Tensor,
    bc_pts: torch.Tensor,
    config: PINNConfig,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute total weighted loss: w_pde * L_pde + w_ic * L_ic + w_bc * L_bc.

    Args:
        model: PINN network.
        x_pde, y_pde, t_pde: Interior coordinates.
        ic_pts: Initial condition coordinates.
        bc_pts: Boundary condition coordinates.
        config: Configuration containing weights and alpha.

    Returns:
        Tuple of (total_loss_tensor, loss_dict_with_float_values).
    """
    loss_pde = compute_pde_residual_loss(model, x_pde, y_pde, t_pde, config.alpha)
    loss_ic = compute_ic_loss(model, ic_pts)
    loss_bc = compute_bc_loss(model, bc_pts)

    total_loss = (
        config.weight_pde * loss_pde
        + config.weight_ic * loss_ic
        + config.weight_bc * loss_bc
    )

    losses_dict = {
        "loss_total": total_loss.item(),
        "loss_pde": loss_pde.item(),
        "loss_ic": loss_ic.item(),
        "loss_bc": loss_bc.item(),
    }

    return total_loss, losses_dict

