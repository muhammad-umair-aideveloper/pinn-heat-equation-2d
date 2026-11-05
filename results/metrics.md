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
- **Training Time**: 1215.65 s

## Relative $L_2$ Error vs Exact Solution

Exact solution: $T(x, y, t) = \sin(\pi x) \sin(\pi y) \exp(-2 \pi^2 \alpha t)$

| Time ($t$) | Relative $L_2$ Error (%) | Max Absolute Error | Mean Absolute Error | Status |
|:---:|:---:|:---:|:---:|:---:|
| 0.00 | 0.7690% | 0.020364 | 0.003047 | PASSED (< 2%) |
| 0.25 | 0.8321% | 0.019171 | 0.003066 | PASSED (< 2%) |
| 0.50 | 0.8789% | 0.017928 | 0.002975 | PASSED (< 2%) |
| 1.00 | 0.9648% | 0.016553 | 0.002948 | PASSED (< 2%) |

**Mean Relative $L_2$ Error across all snapshots**: **0.8612%**

> [!NOTE]
> All evaluation snapshots met the scholarship target of relative $L_2$ error < 2.0%.
