# Project Structure: `pinn-heat-equation-2d`

Physics-Informed Neural Network (PINN) that solves the 2D heat equation
without any dataset.

PDE: dT/dt = alpha * (d2T/dx2 + d2T/dy2), on x, y in [0,1], t in [0,1], alpha = 0.01

## Folder layout

```
pinn-heat-equation-2d/
├── README.md              # Project overview, results table, how to run
├── AGENTS.md              # Rules for the AI coding agent
├── STRUCTURE.md           # This file
├── requirements.txt       # Python dependencies
├── .gitignore             # Files git must ignore
├── LICENSE                # MIT license
├── src/
│   ├── config.py          # All hyperparameters in one place
│   ├── model.py           # PINN network: (x, y, t) -> T
│   ├── physics_loss.py    # PDE + initial + boundary losses (autograd)
│   ├── train.py           # Training loop, saves model and loss curve
│   ├── evaluate.py        # Relative L2 error vs exact solution
│   └── visualize.py       # Creates the heat evolution GIF
└── results/
    ├── pinn.pt            # Trained model weights
    ├── loss_curve.png     # Loss of each term vs epoch
    ├── metrics.md         # Relative L2 error table (real numbers only)
    └── heat_evolution.gif # PINN vs exact vs error map
```

## What each file does

| File | Responsibility |
|---|---|
| `src/config.py` | Alpha, epochs, learning rate, number of collocation points, loss weights, random seed, grid size. No magic numbers elsewhere. |
| `src/model.py` | MLP with input (x, y, t), 4 hidden layers x 64 units, `tanh` activation, output T. Xavier initialization. |
| `src/physics_loss.py` | Computes dT/dt, d2T/dx2, d2T/dy2 with `torch.autograd.grad(create_graph=True)`. Returns the PDE residual loss, initial condition loss (T = sin(pi x) sin(pi y) at t=0) and boundary loss (T = 0 on all 4 edges), then the weighted total. |
| `src/train.py` | Adam optimizer, new random collocation points every epoch, logs each loss term every 100 epochs, saves `results/pinn.pt` and `results/loss_curve.png`. |
| `src/evaluate.py` | Compares the PINN with the exact solution `sin(pi x) sin(pi y) exp(-2 pi^2 alpha t)` on a 50x50 grid at t = 0, 0.25, 0.5, 1.0 and writes the relative L2 error to `results/metrics.md`. |
| `src/visualize.py` | Builds `results/heat_evolution.gif`: 3 panels (PINN, exact, absolute error), shared colorbar, 40 frames, time in the title. |

## Data flow

```
config.py ──► train.py ──► model.py + physics_loss.py ──► results/pinn.pt
                                                              │
                                      ┌───────────────────────┴──────────┐
                                      ▼                                  ▼
                               evaluate.py                        visualize.py
                          results/metrics.md                results/heat_evolution.gif
```

## How to run (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

python src/train.py
python src/evaluate.py
python src/visualize.py
```

## Rules for this structure

- Do not add files or folders outside this layout without a reason.
- Keep paths relative to the repo root (use `pathlib.Path`).
- Keep `results/` in git for small outputs (plots, metrics, GIF). Model weights are small, so commit `pinn.pt` too.
- Every change to any file gets its own conventional commit (see `AGENTS.md`).
