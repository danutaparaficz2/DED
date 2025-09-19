"""
Example: Training PINN to solve Burgers' Equation

This script demonstrates how to use the PINN implementation to solve
the viscous Burgers equation: u_t + u * u_x = ν * u_xx

Domain: x ∈ [-1, 1], t ∈ [0, 1]
Boundary conditions: periodic u(-1, t) = u(1, t)
Initial condition: u(x, 0) = -sin(π*x)
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import numpy as np

from pinn import PINN
from pde_problems import BurgersEquation
from utils import save_results


def main():
    print("="*60)
    print("PINN Training Example: Burgers Equation")
    print("="*60)
    
    # Set random seed for reproducibility
    torch.manual_seed(456)
    np.random.seed(456)
    
    # Create the PDE problem
    print("Setting up Burgers Equation problem...")
    problem = BurgersEquation(
        nu=0.01,             # viscosity coefficient
        T=1.0,               # time horizon
        n_collocation=3000,  # interior points
        n_boundary=300,      # boundary points
        n_initial=300        # initial condition points
    )
    
    # Create PINN model
    print("Creating PINN model...")
    model = PINN(
        input_dim=2,                                    # (x, t)
        hidden_layers=[100, 100, 100, 100, 100, 100],  # 6 hidden layers
        output_dim=1,                                   # u(x, t)
        activation='tanh'
    )
    
    print(f"Model architecture:")
    print(f"  Input dimension: {model.input_dim}")
    print(f"  Hidden layers: {[100, 100, 100, 100, 100, 100]}")
    print(f"  Output dimension: {model.output_dim}")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Train the model
    print("\nStarting training...")
    losses = model.train_model(
        pde_loss_fn=problem.pde_loss,
        boundary_loss_fn=problem.boundary_loss,
        initial_loss_fn=problem.initial_loss,
        num_epochs=10000,
        lr=5e-4,           # Lower learning rate for stability
        lambda_pde=1.0,
        lambda_bc=10.0,    # Weight for boundary conditions
        lambda_ic=10.0,    # Weight for initial conditions
        print_every=500
    )
    
    print("Training completed!")
    
    # Save results (no exact solution available for comparison)
    print("\nSaving results...")
    save_results(model, problem, losses, output_dir="results/burgers_equation")
    
    print("\nExample completed! Check the 'results/burgers_equation' directory for outputs.")
    print("Note: Burgers equation is a challenging nonlinear PDE that may require")
    print("      longer training times or parameter tuning for optimal results.")


if __name__ == "__main__":
    main()