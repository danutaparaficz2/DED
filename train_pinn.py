#!/usr/bin/env python3
"""
Main training script for Physics Informed Neural Networks (PINNs)

This script provides a command-line interface for training PINNs on various PDE problems.

Usage:
    python train_pinn.py --problem heat --epochs 5000 --lr 0.001
    python train_pinn.py --problem wave --epochs 8000 --hidden_layers 64 64 64 64
    python train_pinn.py --problem burgers --epochs 10000 --lr 0.0005
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import argparse
import torch
import numpy as np
from typing import List

from src.pinn import PINN
from src.pde_problems import HeatEquation, WaveEquation, BurgersEquation
from src.utils import save_results, compute_error_metrics


def create_problem(problem_name: str, **kwargs):
    """Create a PDE problem instance based on the problem name."""
    problems = {
        'heat': HeatEquation,
        'wave': WaveEquation, 
        'burgers': BurgersEquation
    }
    
    if problem_name not in problems:
        raise ValueError(f"Unknown problem: {problem_name}. Available: {list(problems.keys())}")
    
    return problems[problem_name](**kwargs)


def main():
    parser = argparse.ArgumentParser(description='Train Physics Informed Neural Networks')
    
    # Problem selection
    parser.add_argument('--problem', type=str, required=True,
                       choices=['heat', 'wave', 'burgers'],
                       help='PDE problem to solve')
    
    # Model architecture
    parser.add_argument('--hidden_layers', type=int, nargs='+', default=[50, 50, 50, 50],
                       help='Hidden layer sizes (default: [50, 50, 50, 50])')
    parser.add_argument('--activation', type=str, default='tanh',
                       choices=['tanh', 'relu', 'sigmoid', 'leaky_relu'],
                       help='Activation function (default: tanh)')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=5000,
                       help='Number of training epochs (default: 5000)')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate (default: 0.001)')
    parser.add_argument('--lambda_pde', type=float, default=1.0,
                       help='Weight for PDE loss (default: 1.0)')
    parser.add_argument('--lambda_bc', type=float, default=100.0,
                       help='Weight for boundary condition loss (default: 100.0)')
    parser.add_argument('--lambda_ic', type=float, default=100.0,
                       help='Weight for initial condition loss (default: 100.0)')
    
    # Problem-specific parameters
    parser.add_argument('--n_collocation', type=int, default=2000,
                       help='Number of collocation points (default: 2000)')
    parser.add_argument('--n_boundary', type=int, default=200,
                       help='Number of boundary points (default: 200)')
    parser.add_argument('--n_initial', type=int, default=200,
                       help='Number of initial condition points (default: 200)')
    
    # Output options
    parser.add_argument('--output_dir', type=str, default='results',
                       help='Output directory for results (default: results)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--print_every', type=int, default=500,
                       help='Print loss every N epochs (default: 500)')
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    print("="*70)
    print(f"PINN Training: {args.problem.upper()} EQUATION")
    print("="*70)
    
    # Create problem-specific parameters
    problem_params = {
        'n_collocation': args.n_collocation,
        'n_boundary': args.n_boundary,
        'n_initial': args.n_initial
    }
    
    # Add problem-specific default parameters
    if args.problem == 'heat':
        problem_params.update({'alpha': 0.1, 'L': 1.0, 'T': 1.0})
    elif args.problem == 'wave':
        problem_params.update({'c': 1.0, 'L': 1.0, 'T': 2.0})
    elif args.problem == 'burgers':
        problem_params.update({'nu': 0.01, 'T': 1.0})
    
    # Create the PDE problem
    print(f"Setting up {args.problem.title()} Equation problem...")
    problem = create_problem(args.problem, **problem_params)
    
    # Create PINN model
    print("Creating PINN model...")
    model = PINN(
        input_dim=2,  # (x, t) for all problems
        hidden_layers=args.hidden_layers,
        output_dim=1,  # u(x, t)
        activation=args.activation
    )
    
    print(f"Model configuration:")
    print(f"  Input dimension: {model.input_dim}")
    print(f"  Hidden layers: {args.hidden_layers}")
    print(f"  Output dimension: {model.output_dim}")
    print(f"  Activation: {args.activation}")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    print(f"\\nTraining configuration:")
    print(f"  Epochs: {args.epochs}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Loss weights - PDE: {args.lambda_pde}, BC: {args.lambda_bc}, IC: {args.lambda_ic}")
    print(f"  Training points - Collocation: {args.n_collocation}, Boundary: {args.n_boundary}, Initial: {args.n_initial}")
    
    # Train the model
    print("\\nStarting training...")
    losses = model.train_model(
        pde_loss_fn=problem.pde_loss,
        boundary_loss_fn=problem.boundary_loss,
        initial_loss_fn=problem.initial_loss,
        num_epochs=args.epochs,
        lr=args.lr,
        lambda_pde=args.lambda_pde,
        lambda_bc=args.lambda_bc,
        lambda_ic=args.lambda_ic,
        print_every=args.print_every
    )
    
    print("\\nTraining completed!")
    
    # Compute error metrics (if exact solution is available)
    print("\\nComputing error metrics...")
    try:
        metrics = compute_error_metrics(model, problem, n_test_points=2000)
        if metrics:
            print("Error Metrics:")
            for key, value in metrics.items():
                print(f"  {key}: {value:.6e}")
    except Exception as e:
        print(f"Could not compute error metrics: {e}")
    
    # Save results
    output_path = os.path.join(args.output_dir, f"{args.problem}_equation")
    print(f"\\nSaving results to {output_path}...")
    save_results(model, problem, losses, output_dir=output_path)
    
    print("\\nTraining completed successfully!")
    print(f"Check the '{output_path}' directory for model, plots, and metrics.")


if __name__ == "__main__":
    main()