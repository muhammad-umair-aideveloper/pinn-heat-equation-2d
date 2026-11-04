"""Unit tests for 2D Heat Equation PINN physics loss and model architecture.

Tests:
    Test A: Verify PDE residual equals zero (within float64 precision < 1e-5) for the exact solution.
    Test B: Verify initial condition loss equals zero for the exact initial temperature profile.
    Test C: Verify boundary condition loss equals zero when boundary temperatures vanish.
    Test D: Verify model output shape (N, 1) and finite loss outputs (no NaN or Inf).
"""

import math
import sys
from pathlib import Path

import torch
import torch.nn as nn

# Ensure project root is in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import PINNConfig
from src.model import HeatPINN
from src.physics_loss import (
    compute_bc_loss,
    compute_ic_loss,
    compute_pde_residual_loss,
    compute_total_loss,
    sample_collocation_points,
)


class ExactHeatSolutionModule(nn.Module):
    """Wrapper module implementing the exact analytical heat equation solution."""

    def __init__(self, alpha: float) -> None:
        super().__init__()
        self.alpha = alpha

    def forward(self, xyt: torch.Tensor) -> torch.Tensor:
        x = xyt[:, 0:1]
        y = xyt[:, 1:2]
        t = xyt[:, 2:3]
        pi = math.pi
        decay = torch.exp(-2.0 * (pi ** 2) * self.alpha * t)
        return torch.sin(pi * x) * torch.sin(pi * y) * decay


def test_a_exact_solution_pde_residual_zero() -> None:
    """Test A: PDE residual on exact solution must be < 1e-5 in float64."""
    alpha = 0.01
    exact_mod = ExactHeatSolutionModule(alpha=alpha).to(torch.float64)

    # Random test coordinates in double precision
    torch.manual_seed(123)
    n_pts = 100
    x = torch.rand(n_pts, 1, dtype=torch.float64, requires_grad=True)
    y = torch.rand(n_pts, 1, dtype=torch.float64, requires_grad=True)
    t = torch.rand(n_pts, 1, dtype=torch.float64, requires_grad=True)

    residual_loss = compute_pde_residual_loss(exact_mod, x, y, t, alpha=alpha)
    res_val = residual_loss.item()
    print(f"[Test A] Exact solution PDE residual MSE: {res_val:.2e}")
    assert res_val < 1e-5, f"PDE residual of exact solution exceeds 1e-5: {res_val}"


def test_b_initial_condition_exact_zero() -> None:
    """Test B: IC loss must be ~0 when model output equals exact IC at t=0."""
    alpha = 0.01
    exact_mod = ExactHeatSolutionModule(alpha=alpha).to(torch.float64)

    torch.manual_seed(456)
    n_pts = 100
    x_ic = torch.rand(n_pts, 1, dtype=torch.float64)
    y_ic = torch.rand(n_pts, 1, dtype=torch.float64)
    t_ic = torch.zeros(n_pts, 1, dtype=torch.float64)
    ic_pts = torch.cat([x_ic, y_ic, t_ic], dim=1)

    ic_loss = compute_ic_loss(exact_mod, ic_pts)
    ic_val = ic_loss.item()
    print(f"[Test B] Exact initial condition loss: {ic_val:.2e}")
    assert ic_val < 1e-12, f"IC loss of exact IC profile exceeds 1e-12: {ic_val}"


def test_c_boundary_condition_exact_zero() -> None:
    """Test C: Boundary loss must be ~0 when evaluated on exact solution edges."""
    alpha = 0.01
    exact_mod = ExactHeatSolutionModule(alpha=alpha).to(torch.float64)

    n_edge = 25
    # Boundaries: x=0, x=1, y=0, y=1
    e1 = torch.cat([torch.zeros(n_edge, 1, dtype=torch.float64), torch.rand(n_edge, 1, dtype=torch.float64), torch.rand(n_edge, 1, dtype=torch.float64)], dim=1)
    e2 = torch.cat([torch.ones(n_edge, 1, dtype=torch.float64), torch.rand(n_edge, 1, dtype=torch.float64), torch.rand(n_edge, 1, dtype=torch.float64)], dim=1)
    e3 = torch.cat([torch.rand(n_edge, 1, dtype=torch.float64), torch.zeros(n_edge, 1, dtype=torch.float64), torch.rand(n_edge, 1, dtype=torch.float64)], dim=1)
    e4 = torch.cat([torch.rand(n_edge, 1, dtype=torch.float64), torch.ones(n_edge, 1, dtype=torch.float64), torch.rand(n_edge, 1, dtype=torch.float64)], dim=1)
    bc_pts = torch.cat([e1, e2, e3, e4], dim=0)

    bc_loss = compute_bc_loss(exact_mod, bc_pts)
    bc_val = bc_loss.item()
    print(f"[Test C] Exact boundary condition loss: {bc_val:.2e}")
    assert bc_val < 1e-12, f"BC loss on zero boundaries exceeds 1e-12: {bc_val}"


def test_d_model_shape_and_finite_loss() -> None:
    """Test D: Model output shape is (N, 1) for input (N, 3), and losses are finite."""
    cfg = PINNConfig(
        n_collocation_pde=64,
        n_collocation_ic=32,
        n_collocation_bc=32,
    )
    dev = torch.device("cpu")
    model = HeatPINN(
        input_dim=cfg.input_dim,
        hidden_dim=cfg.hidden_dim,
        num_hidden_layers=cfg.num_hidden_layers,
        output_dim=cfg.output_dim,
    ).to(dev)

    # Check forward shape
    test_in = torch.randn(50, 3, device=dev)
    test_out = model(test_in)
    assert test_out.shape == (50, 1), f"Expected shape (50, 1), got {test_out.shape}"

    # Check sample points
    x_pde, y_pde, t_pde, ic_pts, bc_pts = sample_collocation_points(cfg, dev)
    assert x_pde.shape == (64, 1) and x_pde.requires_grad
    assert y_pde.shape == (64, 1) and y_pde.requires_grad
    assert t_pde.shape == (64, 1) and t_pde.requires_grad
    assert ic_pts.shape == (32, 3)
    assert bc_pts.shape == (32, 3)

    # Check finite losses
    tot_loss, loss_dict = compute_total_loss(model, x_pde, y_pde, t_pde, ic_pts, bc_pts, cfg)
    for name, val in loss_dict.items():
        assert not math.isnan(val), f"Loss {name} is NaN"
        assert not math.isinf(val), f"Loss {name} is Inf"
        assert val >= 0.0, f"Loss {name} must be non-negative, got {val}"

    # Test backward pass
    tot_loss.backward()
    for param in model.parameters():
        assert param.grad is not None, "Model parameter missing gradient after backward()"
        assert torch.isfinite(param.grad).all(), "Non-finite gradient detected"

    print(f"[Test D] Forward shape, finite losses, and backward grads verified. Total loss: {loss_dict['loss_total']:.4f}")


def run_all_tests() -> None:
    """Run all unit tests."""
    print("Running Physics Loss and PINN Model Unit Tests...")
    test_a_exact_solution_pde_residual_zero()
    test_b_initial_condition_exact_zero()
    test_c_boundary_condition_exact_zero()
    test_d_model_shape_and_finite_loss()
    print("ALL TESTS PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    run_all_tests()

