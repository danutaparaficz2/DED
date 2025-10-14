"""
Example: Training PINN to solve the 2D Heat Equation

This script demonstrates how to use the PINN implementation to solve
the 2D heat equation: u_t = α * (u_xx + u_yy)

Domain: x,y ∈ [0, 1], t ∈ [0, 1]
Boundary conditions: u = 0 on spatial boundaries
Initial condition: u(x,y,0) = sin(π*x) * sin(π*y)
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import numpy as np
import matplotlib.pyplot as plt

from pinn import PINN
from pde_problems import HeatEquation2D
from utils import save_results, compute_error_metrics, plot_solution_2d


def main():
    print("="*60)
    print("PINN Training Example: 2D Heat Equation")
    print("="*60)
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create the PDE problem
    print("Setting up 2D Heat Equation problem...")
    problem = HeatEquation2D(
        alpha=0.1,           # thermal diffusivity
        L=1.0,               # domain length
        T=1.0,               # time horizon
        n_collocation=4000,  # interior points
        n_boundary=800,      # boundary points
        n_initial=800        # initial condition points
    )
    
    # Create PINN model
    print("Creating PINN model...")
    model = PINN(
        input_dim=3,                    # (x, y, t)
        hidden_layers=[128, 128, 128],  # 3 hidden layers
        output_dim=1,                   # u(x, y, t)
        activation='tanh'
    )
    
    print(f"Model architecture:")
    print(f"  Input dimension: {model.input_dim}")
    print(f"  Hidden layers: {[128,128,128]}")
    print(f"  Output dimension: {model.output_dim}")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Train the model
    print("\nStarting training...")
    losses = model.train_model(
        pde_loss_fn=problem.pde_loss,
        boundary_loss_fn=problem.boundary_loss,
        initial_loss_fn=problem.initial_loss,
        num_epochs=1000,
        lr=1e-3,
        lambda_pde=1.0,
        lambda_bc=100.0,   # Higher weight for boundary conditions
        lambda_ic=100.0,   # Higher weight for initial conditions
        print_every=100
    )
    
    print("Training completed!")
    
    # Compute error metrics
    print("\nComputing error metrics...")
    metrics = compute_error_metrics(model, problem, n_test_points=2000)
    if metrics:
        print("Error Metrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.6e}")
    
    # Save results
    print("\nSaving results...")
    out_dir = "results/heat_equation_2d"
    os.makedirs(out_dir, exist_ok=True)
    # Save model and loss plot
    save_results(model, problem, losses, output_dir=out_dir)

    # Additionally, create spatial slices at several time values and save
    x_min, x_max = problem.domain['x']
    t_vals = [0.0, 0.25, 0.5, 0.75, 1.0]
    n_grid = 100
    xs = np.linspace(x_min, x_max, n_grid)
    ys = np.linspace(x_min, x_max, n_grid)
    X, Y = np.meshgrid(xs, ys)
    for t in t_vals:
        pts = np.vstack([X.flatten(), Y.flatten(), np.ones_like(X.flatten()) * t]).T
        with torch.no_grad():
            pred = model.predict(torch.tensor(pts, dtype=torch.float32)).numpy().reshape(n_grid, n_grid)
        exact = np.sin(np.pi * X) * np.sin(np.pi * Y) * np.exp(-2 * (np.pi ** 2) * problem.alpha * t)
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.contourf(X, Y, pred, levels=50, cmap='viridis')
        plt.title(f'PINN Prediction t={t:.2f}')
        plt.colorbar()
        plt.subplot(1, 3, 2)
        plt.contourf(X, Y, exact, levels=50, cmap='viridis')
        plt.title('Exact')
        plt.colorbar()
        plt.subplot(1, 3, 3)
        plt.contourf(X, Y, np.abs(pred - exact), levels=50, cmap='Reds')
        plt.title('Absolute Error')
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'heat2d_slice_t_{int(t*100):03d}.png'), dpi=200)
        plt.close()

    print("\nExample completed! Check the 'results/heat_equation_2d' directory for outputs.")


if __name__ == "__main__":
    main()
