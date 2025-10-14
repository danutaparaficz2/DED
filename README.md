# DED - Physics Informed Neural Networks

A comprehensive implementation of Physics Informed Neural Networks (PINNs) for solving partial differential equations (PDEs).

## Overview

This repository provides a complete framework for training and testing Physics Informed Neural Networks to solve various types of partial differential equations. PINNs combine the power of neural networks with the physics of the underlying differential equations, enabling the solution of PDEs without requiring large datasets.

## Features

- **Modular PINN Architecture**: Flexible neural network implementation with customizable layers and activations
- **Multiple PDE Problems**: Pre-implemented examples including Heat, Wave, and Burgers equations
- **Automatic Differentiation**: Built-in computation of derivatives up to second order
- **Comprehensive Training**: Physics-informed loss functions with boundary and initial conditions
- **Visualization Tools**: Built-in plotting functions for solutions and training progress
- **Command-Line Interface**: Easy-to-use scripts for training different PDE problems
- **Error Analysis**: Computation of error metrics when exact solutions are available

## Installation

1. Clone the repository:
```bash
git clone https://github.com/danutaparaficz2/DED.git
cd DED
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### Using the Command-Line Interface

Train a PINN to solve the heat equation:
```bash
python train_pinn.py --problem heat --epochs 5000 --lr 0.001
```

Train a PINN to solve the 3D heat equation:
```bash
python train_pinn.py --problem heat3d --epochs 8000 --lr 0.001
```

Train a PINN to solve the wave equation:
```bash
python train_pinn.py --problem wave --epochs 8000 --hidden_layers 64 64 64 64
```

Train a PINN to solve Burgers' equation:
```bash
python train_pinn.py --problem burgers --epochs 10000 --lr 0.0005
```

### Using the Examples

Run pre-configured examples:
```bash
# Heat equation example
python examples/heat_equation_example.py

# 3D Heat equation example
python examples/heat_equation_3d_example.py

# Wave equation example
python examples/wave_equation_example.py

# Burgers equation example
python examples/burgers_equation_example.py
```

### Custom Implementation

```python
import torch
from src.pinn import PINN
from src.pde_problems import HeatEquation
from src.utils import save_results

# Create a PDE problem
problem = HeatEquation(alpha=0.1, L=1.0, T=1.0)

# Create PINN model
model = PINN(
    input_dim=2,
    hidden_layers=[50, 50, 50, 50],
    output_dim=1,
    activation='tanh'
)

# Train the model
losses = model.train_model(
    pde_loss_fn=problem.pde_loss,
    boundary_loss_fn=problem.boundary_loss,
    initial_loss_fn=problem.initial_loss,
    num_epochs=5000,
    lr=1e-3,
    lambda_pde=1.0,
    lambda_bc=100.0,
    lambda_ic=100.0
)

# Save results
save_results(model, problem, losses)
```

## Implemented PDE Problems

### 1. Heat Equation
**Equation**: `u_t = α * u_xx`

- **Domain**: x ∈ [0, L], t ∈ [0, T]
- **Boundary Conditions**: u(0, t) = u(L, t) = 0
- **Initial Condition**: u(x, 0) = sin(π*x/L)
- **Exact Solution**: Available for validation

### 2. 3D Heat Equation
**Equation**: `∂T/∂t = α * (∂²T/∂x² + ∂²T/∂y² + ∂²T/∂z²)`

- **Domain**: x,y,z ∈ [0, L], t ∈ [0, T]
- **Boundary Conditions**: T = 0 on all boundaries
- **Initial Condition**: T(x,y,z,0) = sin(π*x/L)*sin(π*y/L)*sin(π*z/L)
- **Exact Solution**: Available for validation

### 3. Wave Equation
**Equation**: `u_tt = c² * u_xx`

- **Domain**: x ∈ [0, L], t ∈ [0, T]
- **Boundary Conditions**: u(0, t) = u(L, t) = 0
- **Initial Conditions**: u(x, 0) = sin(π*x/L), u_t(x, 0) = 0
- **Exact Solution**: Available for validation

### 4. Burgers Equation
**Equation**: `u_t + u * u_x = ν * u_xx`

- **Domain**: x ∈ [-1, 1], t ∈ [0, T]
- **Boundary Conditions**: Periodic u(-1, t) = u(1, t)
- **Initial Condition**: u(x, 0) = -sin(π*x)
- **Exact Solution**: Not implemented (nonlinear PDE)

## Project Structure

```
DED/
├── src/
│   ├── __init__.py          # Package initialization
│   ├── pinn.py              # Core PINN implementation
│   ├── pde_problems.py      # PDE problem definitions
│   └── utils.py             # Utility functions and visualization
├── examples/
│   ├── heat_equation_example.py
│   ├── heat_equation_3d_example.py
│   ├── wave_equation_example.py
│   └── burgers_equation_example.py
├── tests/
│   └── test_pinn.py         # Unit tests
├── train_pinn.py            # Command-line training interface
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## Command-Line Arguments

The `train_pinn.py` script supports various command-line arguments:

### Problem Selection
- `--problem`: Choose from 'heat', 'heat3d', 'wave', 'burgers'

### Model Architecture
- `--hidden_layers`: List of hidden layer sizes (default: [50, 50, 50, 50])
- `--activation`: Activation function ('tanh', 'relu', 'sigmoid', 'leaky_relu')

### Training Parameters
- `--epochs`: Number of training epochs (default: 5000)
- `--lr`: Learning rate (default: 0.001)
- `--lambda_pde`: Weight for PDE loss (default: 1.0)
- `--lambda_bc`: Weight for boundary condition loss (default: 100.0)
- `--lambda_ic`: Weight for initial condition loss (default: 100.0)

### Data Parameters
- `--n_collocation`: Number of collocation points (default: 2000)
- `--n_boundary`: Number of boundary points (default: 200)
- `--n_initial`: Number of initial condition points (default: 200)

### Output Options
- `--output_dir`: Output directory for results (default: 'results')
- `--seed`: Random seed for reproducibility (default: 42)

## Testing

Run the test suite to verify the implementation:
```bash
python tests/test_pinn.py
```

## Results

Training results are automatically saved to the specified output directory and typically include:

- **Model checkpoint**: Trained PINN model (`.pth` file)
- **Loss plots**: Training loss history (PNG)
- **Solution plots**: 2D and 3D visualizations of the learned solution (PNG)
- **Error metrics**: Text file with RMSE / timing information (when exact solution is available)

Example: Plate-with-Hole (results from `examples/plate_with_hole_example.py`)
---------------------------------------------------------------

When you run the `plate_with_hole_example.py` example the script saves a small set of artifacts to `results/plate_with_hole/`. Expected files (created by the example) are:

- `results/plate_with_hole/plate_hole_model.pth` — saved PyTorch model checkpoint
- `results/plate_with_hole/plate_hole_loss_adam.png` — Adam training loss curve
- `results/plate_with_hole/plate_hole_solution_and_error.png` — 2D panels (prediction, exact, absolute error)
- `results/plate_with_hole/plate_hole_solution_and_error_3d.png` — optional 3D surface panels (triangular surface)
- `results/plate_with_hole/plate_hole_metrics.txt` — simple metrics file (RMSE, timings)

Preview (these images point to files generated by the example and will display when the `results/plate_with_hole/` files exist):



(resu<img width="870" height="332" alt="Screenshot 2025-10-14 at 09 58 45" src="https://github.com/user-attachments/assets/cc3f3b9d-ea13-48eb-8e48-42ad25ab9cdb" />


<img width="933" height="295" alt="Screenshot 2025-10-14 at 10 11 06" src="https://github.com/user-attachments/assets/b28e31e2-44b3-4cdc-96a5-d2c13ea55e12" />

Notes on these outputs
- The 2D/3D plots are created from a triangular sampling of the domain that excludes the circular hole. Use these images to visually compare the PINN solution to the analytic solution `T(r) = (r - a)^2`.
- The metrics file contains a scalar RMSE computed on the held-out grid and the elapsed training times for Adam and L-BFGS (if L-BFGS was run).

How to reproduce the Plate-with-Hole results
1. From the repository root, run the example (default: Adam 10000 epochs + L-BFGS):

```bash
python examples/plate_with_hole_example.py
```

2. For a faster smoke test (short run), edit the `epochs` argument in the example or call the function with smaller values; reduce `n_collocation`/`n_boundary` for quick iteration.

3. After the run completes, open the generated images in `results/plate_with_hole/`.

Corner diagnostics
- Because corners are intersection points of vertical/horizontal outer boundaries, the example supports printing or saving corner diagnostics (predicted vs analytic) as a small post-training check. If you want the script to always write corner diagnostics to disk, I can add that snippet into the example.

## Dependencies

- PyTorch >= 1.9.0
- NumPy >= 1.20.0
- Matplotlib >= 3.3.0
- SciPy >= 1.7.0
- Pandas >= 1.3.0
- Seaborn >= 0.11.0
- tqdm >= 4.62.0

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for:

- New PDE problems
- Improved training algorithms
- Additional visualization tools
- Bug fixes and optimizations

## License

This project is open source and available under the MIT License.

## References

1. Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. Journal of Computational Physics, 378, 686-707.

2. Karniadakis, G. E., Kevrekidis, I. G., Lu, L., Perdikaris, P., Wang, S., & Yang, L. (2021). Physics-informed machine learning. Nature Reviews Physics, 3(6), 422-440.
