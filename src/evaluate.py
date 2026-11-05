"""Evaluation script for 2D Heat Equation PINN.

Loads the trained model weights from results/pinn.pt, evaluates the solution on a
50x50 spatial grid at t = 0.0, 0.25, 0.5, and 1.0, and computes the relative L2 error
against the exact analytical solution:
    T(x, y, t) = sin(pi * x) * sin(pi * y) * exp(-2 * pi^2 * alpha * t)

Outputs the results to stdout and saves a Markdown summary table into results/metrics.md.
"""

import platform
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

# Ensure project root is in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import PINNConfig
from src.model import HeatPINN
from src.visualize import exact_solution_np


def evaluate_relative_l2_errors(
    model: torch.nn.Module,
    config: PINNConfig,
    device: torch.device,
) -> Dict[float, Dict[str, float]]:
    """Compute relative L2 errors across specified evaluation times on an NxN grid.

    Relative L2 error formula:
        ||T_pred - T_exact||_2 / ||T_exact||_2

    Args:
        model: Trained HeatPINN model.
        config: Configuration containing grid size, bounds, alpha, and eval times.
        device: Device where model resides.

    Returns:
        Dictionary mapping each evaluation time t to its error metrics.
    """
    model.eval()
    n_grid = config.eval_grid_size
    x_vals = np.linspace(config.x_min, config.x_max, n_grid)
    y_vals = np.linspace(config.y_min, config.y_max, n_grid)
    X, Y = np.meshgrid(x_vals, y_vals)

    x_flat = torch.from_numpy(X.flatten().astype(np.float32)).unsqueeze(1).to(device)
    y_flat = torch.from_numpy(Y.flatten().astype(np.float32)).unsqueeze(1).to(device)

    results: Dict[float, Dict[str, float]] = {}

    with torch.no_grad():
        for t in config.eval_times:
            t_flat = torch.full_like(x_flat, float(t))
            xyt = torch.cat([x_flat, y_flat, t_flat], dim=1)

            pred_flat = model(xyt).cpu().numpy().flatten()
            T_pred = pred_flat.reshape((n_grid, n_grid))
            T_exact = exact_solution_np(X, Y, t, config.alpha)

            diff = T_pred - T_exact
            l2_error = float(np.linalg.norm(diff))
            l2_norm_exact = float(np.linalg.norm(T_exact))

            if l2_norm_exact > 0:
                rel_l2_error = l2_error / l2_norm_exact
            else:
                rel_l2_error = l2_error

            max_abs_error = float(np.max(np.abs(diff)))
            mean_abs_error = float(np.mean(np.abs(diff)))

            results[t] = {
                "rel_l2_error": rel_l2_error,
                "rel_l2_error_pct": rel_l2_error * 100.0,
                "max_abs_error": max_abs_error,
                "mean_abs_error": mean_abs_error,
                "l2_exact": l2_norm_exact,
            }

    return results


def write_metrics_markdown(
    results: Dict[float, Dict[str, float]],
    config: PINNConfig,
    checkpoint_meta: Dict,
    save_path: Path,
) -> None:
    """Format and save evaluation metrics as a GitHub-flavored Markdown table.

    Args:
        results: Dictionary of computed error metrics per time.
        config: PINNConfig instance.
        checkpoint_meta: Metadata from saved pinn.pt checkpoint.
        save_path: File path to save results/metrics.md.
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)

    cpu_name = platform.processor() or platform.machine()
    python_ver = platform.python_version()
    torch_ver = torch.__version__
    train_time = checkpoint_meta.get("training_time_seconds", 0.0)

    lines: List[str] = [
        "# PINN Heat Equation 2D - Evaluation Metrics",
        "",
        "## Experimental Setup",
        f"- **Hardware (CPU)**: {cpu_name}",
        f"- **Operating System**: {platform.system()} {platform.release()}",
        f"- **Python Version**: {python_ver}",
        f"- **PyTorch Version**: {torch_ver}",
        f"- **Spatial Resolution**: {config.eval_grid_size} x {config.eval_grid_size} uniform grid",
        f"- **Thermal Diffusivity (alpha)**: {config.alpha}",
        f"- **Training Epochs**: {config.epochs}",
        f"- **Collocation Points**: {config.n_collocation_pde} interior, {config.n_collocation_ic} IC, {config.n_collocation_bc} BC",
        f"- **Training Time**: {train_time:.2f} s",
        "",
        "## Relative $L_2$ Error vs Exact Solution",
        "",
        "Exact solution: $T(x, y, t) = \\sin(\\pi x) \\sin(\\pi y) \\exp(-2 \\pi^2 \\alpha t)$",
        "",
        "| Time ($t$) | Relative $L_2$ Error (%) | Max Absolute Error | Mean Absolute Error | Status |",
        "|:---:|:---:|:---:|:---:|:---:|",
    ]

    all_below_target = True
    for t, m in results.items():
        pct = m["rel_l2_error_pct"]
        status = "PASSED (< 2%)" if pct < 2.0 else "ABOVE TARGET"
        if pct >= 2.0:
            all_below_target = False
        lines.append(
            f"| {t:.2f} | {pct:.4f}% | {m['max_abs_error']:.6f} | {m['mean_abs_error']:.6f} | {status} |"
        )

    lines.append("")
    mean_rel_error = np.mean([m["rel_l2_error_pct"] for m in results.values()])
    lines.append(f"**Mean Relative $L_2$ Error across all snapshots**: **{mean_rel_error:.4f}%**")
    lines.append("")
    if all_below_target:
        lines.append("> [!NOTE]\n> All evaluation snapshots met the scholarship target of relative $L_2$ error < 2.0%.")
    else:
        lines.append("> [!WARNING]\n> Some evaluation snapshots exceed 2.0%. Fine-tuning or extended training recommended.")

    with open(save_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"\nEvaluation metrics successfully written to: {save_path}")


def run_evaluation(config: Optional[PINNConfig] = None) -> Dict[float, Dict[str, float]]:
    """Load trained checkpoint and perform evaluation."""
    if config is None:
        config = PINNConfig()

    device = torch.device(config.device)
    if not config.model_path.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found at {config.model_path}. Run training first via src/train.py"
        )

    checkpoint = torch.load(config.model_path, map_location=device, weights_only=False)
    model = HeatPINN(
        input_dim=config.input_dim,
        hidden_dim=config.hidden_dim,
        num_hidden_layers=config.num_hidden_layers,
        output_dim=config.output_dim,
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded trained checkpoint from: {config.model_path}")

    results = evaluate_relative_l2_errors(model, config, device)

    print("\n================ EVALUATION SUMMARY ================")
    print(f"{'Time (t)':>10} | {'Rel L2 Error (%)':>18} | {'Max Abs Error':>15} | {'Status':>12}")
    print("-" * 65)
    for t, m in results.items():
        status = "PASSED" if m["rel_l2_error_pct"] < 2.0 else "FAILED"
        print(f"{t:10.2f} | {m['rel_l2_error_pct']:17.4f}% | {m['max_abs_error']:15.6f} | {status:>12}")
    print("====================================================\n")

    write_metrics_markdown(results, config, checkpoint, config.metrics_path)
    return results


if __name__ == "__main__":
    run_evaluation()

