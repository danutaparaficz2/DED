"""
Example: Training PINN to solve the Heat Equation

This script demonstrates how to use the PINN implementation to solve
the 1D heat equation: u_t = α * u_xx

Domain: x ∈ [0, 1], t ∈ [0, 1]
Boundary conditions: u(0, t) = u(1, t) = 0
Initial condition: u(x, 0) = sin(π*x)
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import numpy as np
import matplotlib.pyplot as plt

from pinn import PINN
from pde_problems import HeatEquation
from utils import save_results, compute_error_metrics


def main():
    print("="*60)
    print("PINN Training Example: Heat Equation")
    print("="*60)
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create the PDE problem
    print("Setting up Heat Equation problem...")
    problem = HeatEquation(
        alpha=0.1,           # thermal diffusivity
        L=1.0,               # domain length
        T=1.0,               # time horizon
        n_collocation=2000,  # interior points
        n_boundary=200,      # boundary points
        n_initial=200        # initial condition points
    )
    
    # Create PINN model
    print("Creating PINN model...")
    model = PINN(
        input_dim=2,                    # (x, t)
        hidden_layers=[50, 50, 50, 50], # 4 hidden layers with 50 neurons each
        output_dim=1,                   # u(x, t)
        activation='tanh'
    )
    
    print(f"Model architecture:")
    print(f"  Input dimension: {model.input_dim}")
    print(f"  Hidden layers: {[50, 50, 50, 50]}")
    print(f"  Output dimension: {model.output_dim}")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Train the model
    print("\nStarting training...")
    losses = model.train_model(
        pde_loss_fn=problem.pde_loss,
        boundary_loss_fn=problem.boundary_loss,
        initial_loss_fn=problem.initial_loss,
        num_epochs=5000,
        lr=1e-3,
        lambda_pde=1.0,
        lambda_bc=100.0,   # Higher weight for boundary conditions
        lambda_ic=100.0,   # Higher weight for initial conditions
        print_every=500
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
    save_results(model, problem, losses, output_dir="results/heat_equation")
    
    print("\nExample completed! Check the 'results/heat_equation' directory for outputs.")


if __name__ == "__main__":
    main()