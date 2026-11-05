# PINN Heat Equation 2D - Evaluation Metrics

## Experimental Setup
- **Hardware (CPU)**: Intel64 Family 6 Model 78 Stepping 3, GenuineIntel
- **Operating System**: Windows 10
- **Python Version**: 3.14.3
- **PyTorch Version**: 2.15.0+cpu
- **Spatial Resolution**: 50 x 50 uniform grid
- **Thermal Diffusivity (alpha)**: 0.01
- **Training Epochs**: 5000
- **Collocation Points**: 5000 interior, 1000 IC, 1000 BC
- **Training Time**: 78.31 s

## Relative $L_2$ Error vs Exact Solution

Exact solution: $T(x, y, t) = \sin(\pi x) \sin(\pi y) \exp(-2 \pi^2 \alpha t)$

| Time ($t$) | Relative $L_2$ Error (%) | Max Absolute Error | Mean Absolute Error | Status |
|:---:|:---:|:---:|:---:|:---:|
| 0.00 | 36.3441% | 0.406312 | 0.144268 | ABOVE TARGET |
| 0.25 | 44.0746% | 0.483284 | 0.158618 | ABOVE TARGET |
| 0.50 | 53.5642% | 0.533219 | 0.183632 | ABOVE TARGET |
| 1.00 | 62.8123% | 0.539800 | 0.201892 | ABOVE TARGET |

**Mean Relative $L_2$ Error across all snapshots**: **49.1988%**

> [!WARNING]
> Some evaluation snapshots exceed 2.0%. Fine-tuning or extended training recommended.
