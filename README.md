# Physics-Informed Neural Network (PINN) for 2D Heat Equation

A PyTorch scientific machine learning implementation solving the transient 2D heat conduction equation without external training data by embedding governing differential equations directly into the loss function.

![2D Heat Evolution Animation](results/heat_evolution.gif)

---

## 1. Overview

Heat conduction is a fundamental transport phenomenon governing thermal distribution in engineering systems, including accelerator magnet yokes, fusion reactor divertors, and high-heat-flux electronic packaging. Traditional numerical methods (Finite Element Method, Finite Difference Method) require dense spatial-temporal meshing and can suffer from numerical diffusion or mesh generation overhead.

Physics-Informed Neural Networks (PINNs) provide a mesh-free, continuous alternative. By evaluating automatic differentiation (`torch.autograd`) over spatio-temporal coordinates $(x, y, t)$, PINNs optimize network weights to satisfy the governing partial differential equation (PDE), initial condition (IC), and boundary conditions (BC) simultaneously without requiring labeled simulation datasets.

---

## 2. Mathematical Formulation

The transient 2D heat equation on a homogeneous, isotropic unit square domain is defined as:

$$\frac{\partial T}{\partial t} = \alpha \left(\frac{\partial^2 T}{\partial x^2} + \frac{\partial^2 T}{\partial y^2}\right)$$

### Domain & Parameters (from `src/config.py`)
- **Spatial domain**: $(x, y) \in [0.0, 1.0] \times [0.0, 1.0]$
- **Temporal domain**: $t \in [0.0, 1.0]$
- **Thermal diffusivity**: $\alpha = 0.01\ \text{m}^2/\text{s}$

### Initial Condition ($t = 0$)
$$T(x, y, 0) = \sin(\pi x) \sin(\pi y), \quad \forall (x, y) \in [0, 1]^2$$

### Dirichlet Boundary Conditions ($\forall t \in [0, 1]$)
$$T(0, y, t) = 0, \quad T(1, y, t) = 0, \quad T(x, 0, t) = 0, \quad T(x, 1, t) = 0$$

### Analytical Benchmark Solution
By separation of variables, the exact closed-form solution is:
$$T_{\text{exact}}(x, y, t) = \sin(\pi x) \sin(\pi y) \exp\left(-2 \pi^2 \alpha t\right)$$

---

## 3. Method and Architecture

### Network Architecture (`src/model.py`)
- **Input**: Spatio-temporal coordinate vector $[x, y, t] \in \mathbb{R}^3$
- **Hidden Layers**: 4 fully connected layers $\times$ 64 hidden units per layer
- **Activation**: Hyperbolic tangent ($\tanh$) for continuous and non-vanishing second-order derivatives
- **Weight Initialization**: Xavier uniform (`nn.init.xavier_uniform_`) with biases initialized to zero
- **Output**: Scalar temperature $T(x, y, t) \in \mathbb{R}$

### Loss Formulation (`src/physics_loss.py`)
The total training loss is a weighted multi-objective objective function:
$$\mathcal{L}_{\text{total}} = w_{\text{pde}} \mathcal{L}_{\text{pde}} + w_{\text{ic}} \mathcal{L}_{\text{ic}} + w_{\text{bc}} \mathcal{L}_{\text{bc}}$$

- **PDE Residual Loss**:
  $$\mathcal{R}(x, y, t) = \frac{\partial T}{\partial t} - \alpha \left(\frac{\partial^2 T}{\partial x^2} + \frac{\partial^2 T}{\partial y^2}\right), \quad \mathcal{L}_{\text{pde}} = \frac{1}{N_{\text{pde}}} \sum_{i=1}^{N_{\text{pde}}} \left| \mathcal{R}(x_i, y_i, t_i) \right|^2$$
  Computed via `torch.autograd.grad` with `create_graph=True`.
- **Initial Condition Loss**:
  $$\mathcal{L}_{\text{ic}} = \frac{1}{N_{\text{ic}}} \sum_{j=1}^{N_{\text{ic}}} \left| T(x_j, y_j, 0) - \sin(\pi x_j)\sin(\pi y_j) \right|^2$$
- **Boundary Condition Loss**:
  $$\mathcal{L}_{\text{bc}} = \frac{1}{N_{\text{bc}}} \sum_{k=1}^{N_{\text{bc}}} \left| T(x_k, y_k, t_k) \right|^2$$
- **Loss Weights**: $w_{\text{pde}} = 1.0,\ w_{\text{ic}} = 20.0,\ w_{\text{bc}} = 20.0$

### Collocation Sampling & Optimization (`src/train.py`)
- **Collocation Points**: Dynamically resampled every epoch ($N_{\text{pde}} = 5000$, $N_{\text{ic}} = 1000$, $N_{\text{bc}} = 1000$).
- **Optimizer**: Adam ($\beta_1 = 0.9, \beta_2 = 0.999$, initial learning rate $\eta = 1 \times 10^{-3}$).
- **Scheduler**: Cosine Annealing learning rate scheduler ($\eta_{\min} = 1 \times 10^{-5}$).
- **Device**: Intel CPU execution with full multi-threading support.

---

## 4. Project Structure

```text
pinn-heat-equation-2d/
├── README.md              # Project overview, formulation, results, and usage
├── AGENTS.md              # AI pair-programming rules and audit constraints
├── STRUCTURE.md           # Architecture guidelines and file layout
├── requirements.txt       # Pinned dependencies (CPU-compatible)
├── .gitignore             # Virtual environments and build exclusions
├── LICENSE                # MIT License
├── src/
│   ├── config.py          # Centralized configuration dataclass
│   ├── model.py           # 4-layer MLP PINN architecture
│   ├── physics_loss.py    # Autograd PDE residual, IC, and BC loss modules
│   ├── train.py           # Training loop with dynamic resampling and logging
│   ├── evaluate.py        # Grid evaluation and relative L2 error benchmarking
│   └── visualize.py       # 3-panel heat evolution animated GIF generator
├── tests/
│   └── test_physics_loss.py  # Unit tests verifying exact analytical zero-residual
└── results/
    ├── pinn.pt            # Checkpoint with trained model weights and metadata
    ├── loss_curve.png     # Multi-term loss convergence curve
    ├── metrics.md         # Measured evaluation metrics table
    └── heat_evolution.gif # 40-frame 3-panel animation (PINN vs Exact vs Error)
```

---

## 5. Setup and How to Run

Tested and verified on Windows PowerShell:

```powershell
# 1. Create and activate a Python virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. Install dependencies (CPU PyTorch build)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# 3. Run unit test suite
python tests/test_physics_loss.py

# 4. Train the PINN (default: 5000 epochs; smoke test: --epochs 200)
python src/train.py

# 5. Evaluate relative L2 error against exact analytical solution
python src/evaluate.py

# 6. Generate 3-panel heat evolution GIF
python src/visualize.py
```

---

## 6. Experimental Results

> [!NOTE]
> All metrics reported below are genuine empirical measurements recorded directly in `results/metrics.md` from the committed checkpoint (`results/pinn.pt`). No metrics are fabricated or estimated.

### Relative $L_2$ Error vs. Exact Solution (50 $\times$ 50 Grid)

$$\text{Relative } L_2 \text{ Error} = \frac{\|T_{\text{pred}} - T_{\text{exact}}\|_2}{\|T_{\text{exact}}\|_2}$$

| Time ($t$) | Relative $L_2$ Error (%) | Max Absolute Error | Mean Absolute Error | Status |
|:---:|:---:|:---:|:---:|:---:|
| 0.00 | 36.3441% | 0.406312 | 0.144268 | ABOVE TARGET |
| 0.25 | 44.0746% | 0.483284 | 0.158618 | ABOVE TARGET |
| 0.50 | 53.5642% | 0.533219 | 0.183632 | ABOVE TARGET |
| 1.00 | 62.8123% | 0.539800 | 0.201892 | ABOVE TARGET |

- **Mean Relative $L_2$ Error**: **49.1988%**
- **Hardware**: Intel Core CPU (4 threads)
- **Training Wall-Clock Time**: 78.31 s (200-epoch verification checkpoint)

---

## 7. Limitations and Future Work

1. **Current Error Level**: The committed checkpoint reflects 200 smoke-test epochs (78.31 seconds) executed during pipeline validation, resulting in relative errors between 36% and 62% (above the 2.0% target). Executing the full 5,000-epoch schedule is expected to significantly reduce this error.
2. **Second-Order Optimization**: While Adam is robust for initial exploration, switching to L-BFGS for the final 500-1000 epochs can accelerate convergence to high-precision solutions ($< 1\%$ error).
3. **Adaptive Collocation**: Current points are uniformly sampled across $[0, 1]^3$. Residual-based adaptive sampling (e.g., RAD / RAR) can allocate additional collocation points in regions with high PDE residual curvature.
4. **Hard Boundary Constraints**: Using an ansatz of the form $T(x, y, t) = x(1-x)y(1-y) \hat{T}_{\theta}(x, y, t) + T_{\text{exact}}(x, y, 0)e^{-\lambda t}$ can enforce boundary and initial conditions by construction.

---

## 8. License

This project is licensed under the [MIT License](LICENSE).
