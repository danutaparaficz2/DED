"""
Example: Training PINN to solve the 3D Heat Equation

This script demonstrates how to use the PINN implementation to solve
the 3D heat equation: ∂T/∂t = α * (∂²T/∂x² + ∂²T/∂y² + ∂²T/∂z²)

Domain: x,y,z ∈ [0, 1], t ∈ [0, 1]
Boundary conditions: T = 0 on all boundaries
Initial condition: T(x,y,z,0) = sin(π*x)*sin(π*y)*sin(π*z)
Exact solution: T(x,y,z,t) = sin(π*x)*sin(π*y)*sin(π*z)*exp(-3*π²*α*t)
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from pinn import PINN
from pde_problems import HeatEquation3D
from utils import save_results, compute_error_metrics


def plot_3d_solution(model, problem, output_dir="results/heat_equation_3d"):
    """Create 3D visualizations of the solution at different time steps."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a grid for visualization (reduce resolution for 3D plotting)
    n_points = 20
    x = torch.linspace(0, problem.L, n_points)
    y = torch.linspace(0, problem.L, n_points)
    z = torch.linspace(0, problem.L, n_points)
    
    time_steps = [0.0, 0.1, 0.2, 0.5]
    
    fig = plt.figure(figsize=(20, 5))
    
    for i, t_val in enumerate(time_steps):
        ax = fig.add_subplot(1, 4, i+1, projection='3d')
        
        # Create meshgrid for middle slice (z = L/2)
        X, Y = torch.meshgrid(x, y, indexing='ij')
        Z = torch.ones_like(X) * problem.L / 2
        T = torch.ones_like(X) * t_val
        
        # Prepare input tensor
        points = torch.stack([X.flatten(), Y.flatten(), Z.flatten(), T.flatten()], dim=1)
        
        # Get predictions
        with torch.no_grad():
            T_pred = model(points).reshape(X.shape)
        
        # Plot surface
        surf = ax.plot_surface(X.numpy(), Y.numpy(), T_pred.numpy(), 
                              cmap='coolwarm', alpha=0.7)
        
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('Temperature')
        ax.set_title(f't = {t_val:.1f}')
        
        # Set consistent z-limits
        ax.set_zlim([0, 1])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'heat_equation_3d_solution.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_cross_sections(model, problem, output_dir="results/heat_equation_3d"):
    """Plot cross-sections of the 3D solution."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create cross-sections at t=0.1
    t_val = 0.1
    n_points = 50
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # XY cross-section (z = L/2)
    x = torch.linspace(0, problem.L, n_points)
    y = torch.linspace(0, problem.L, n_points)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    Z = torch.ones_like(X) * problem.L / 2
    T = torch.ones_like(X) * t_val
    
    points = torch.stack([X.flatten(), Y.flatten(), Z.flatten(), T.flatten()], dim=1)
    with torch.no_grad():
        T_pred = model(points).reshape(X.shape)
    
    im1 = axes[0].contourf(X.numpy(), Y.numpy(), T_pred.numpy(), levels=20, cmap='coolwarm')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    axes[0].set_title(f'XY cross-section (z={problem.L/2:.1f}, t={t_val})')
    plt.colorbar(im1, ax=axes[0])
    
    # XZ cross-section (y = L/2)
    z = torch.linspace(0, problem.L, n_points)
    X, Z = torch.meshgrid(x, z, indexing='ij')
    Y = torch.ones_like(X) * problem.L / 2
    T = torch.ones_like(X) * t_val
    
    points = torch.stack([X.flatten(), Y.flatten(), Z.flatten(), T.flatten()], dim=1)
    with torch.no_grad():
        T_pred = model(points).reshape(X.shape)
    
    im2 = axes[1].contourf(X.numpy(), Z.numpy(), T_pred.numpy(), levels=20, cmap='coolwarm')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('z')
    axes[1].set_title(f'XZ cross-section (y={problem.L/2:.1f}, t={t_val})')
    plt.colorbar(im2, ax=axes[1])
    
    # YZ cross-section (x = L/2)
    Y, Z = torch.meshgrid(y, z, indexing='ij')
    X = torch.ones_like(Y) * problem.L / 2
    T = torch.ones_like(Y) * t_val
    
    points = torch.stack([X.flatten(), Y.flatten(), Z.flatten(), T.flatten()], dim=1)
    with torch.no_grad():
        T_pred = model(points).reshape(Y.shape)
    
    im3 = axes[2].contourf(Y.numpy(), Z.numpy(), T_pred.numpy(), levels=20, cmap='coolwarm')
    axes[2].set_xlabel('y')
    axes[2].set_ylabel('z')
    axes[2].set_title(f'YZ cross-section (x={problem.L/2:.1f}, t={t_val})')
    plt.colorbar(im3, ax=axes[2])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'heat_equation_3d_cross_sections.png'), dpi=300, bbox_inches='tight')
    plt.close()


def main():
    print("="*70)
    print("PINN Training Example: 3D Heat Equation")
    print("="*70)
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create the PDE problem
    print("Setting up 3D Heat Equation problem...")
    problem = HeatEquation3D(
        alpha=1.0,           # thermal diffusivity
        L=1.0,               # domain length in each dimension
        T=1.0,               # time horizon
        n_collocation=5000,  # interior points (more needed for 3D)
        n_boundary=600,      # boundary points (6 faces)
        n_initial=500        # initial condition points
    )
    
    # Create PINN model using the specialized 3D architecture
    print("Creating 3D PINN model...")
    model = PINN.create_3d_heat_pinn()
    
    print(f"Model architecture:")
    print(f"  Input dimension: {model.input_dim}")
    print(f"  Hidden layers: [256, 256, 256]")
    print(f"  Output dimension: {model.output_dim}")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    print(f"\nProblem configuration:")
    print(f"  Domain: x,y,z ∈ [0, {problem.L}], t ∈ [0, {problem.T}]")
    print(f"  Thermal diffusivity α = {problem.alpha}")
    print(f"  Training points - Collocation: {problem.n_collocation}, Boundary: {problem.n_boundary}, Initial: {problem.n_initial}")
    
    # Train the model
    print("\nStarting training...")
    losses = model.train_model(
        pde_loss_fn=problem.pde_loss,
        boundary_loss_fn=problem.boundary_loss,
        initial_loss_fn=problem.initial_loss,
        num_epochs=8000,     # More epochs for 3D problem
        lr=1e-3,
        lambda_pde=1.0,
        lambda_bc=100.0,     # Higher weight for boundary conditions
        lambda_ic=100.0,     # Higher weight for initial conditions
        print_every=500
    )
    
    print("Training completed!")
    
    # Compute error metrics
    print("\nComputing error metrics...")
    metrics = compute_error_metrics(model, problem, n_test_points=1000)
    if metrics:
        print("Error Metrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.6e}")
    
    # Create custom 3D visualizations
    print("\nCreating 3D visualizations...")
    output_dir = "results/heat_equation_3d"
    plot_3d_solution(model, problem, output_dir)
    plot_cross_sections(model, problem, output_dir)
    
    # Save results
    print("Saving results...")
    save_results(model, problem, losses, output_dir=output_dir)
    
    print(f"\nExample completed! Check the '{output_dir}' directory for outputs.")
    print("Generated files:")
    print("  - Model checkpoint: heat_equation_3d_model.pth")
    print("  - Training loss plot: heat_equation_3d_loss.png") 
    print("  - 3D solution visualization: heat_equation_3d_solution.png")
    print("  - Cross-sections: heat_equation_3d_cross_sections.png")
    print("  - Error metrics: heat_equation_3d_metrics.txt")


if __name__ == "__main__":
    main()