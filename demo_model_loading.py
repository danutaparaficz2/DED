#!/usr/bin/env python3
"""
Demonstration script showing how to load and use a trained PINN model.

This script demonstrates how to:
1. Load a pre-trained PINN model
2. Make predictions at new points
3. Compare with exact solutions
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import torch
import numpy as np
import matplotlib.pyplot as plt

from src.pinn import PINN
from src.pde_problems import HeatEquation


def main():
    print("="*60)
    print("PINN Model Loading and Prediction Demo")
    print("="*60)
    
    # Check if model exists
    model_path = "results/heat_equation/heat_equation_model.pth"
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        print("Please run the heat equation example first:")
        print("  python examples/heat_equation_example.py")
        return
    
    # Load the trained model
    print("Loading trained PINN model...")
    model = PINN.load_model(model_path, hidden_layers=[50, 50, 50, 50])
    print("Model loaded successfully!")
    
    # Create problem for exact solution comparison
    problem = HeatEquation()
    
    # Generate test points
    print("\\nGenerating predictions...")
    x_test = torch.linspace(0, 1, 50).reshape(-1, 1)
    t_values = [0.0, 0.2, 0.5, 1.0]
    
    print("\\nPredictions at different times:")
    for t in t_values:
        t_test = torch.ones_like(x_test) * t
        xt_test = torch.cat([x_test, t_test], dim=1)
        
        # PINN prediction
        u_pred = model.predict(xt_test)
        
        # Exact solution
        u_exact = problem.exact_solution(xt_test)
        
        # Compute error
        error = torch.mean((u_pred - u_exact) ** 2).item()
        
        print(f"  t = {t:.1f}: Mean squared error = {error:.2e}")
    
    print("\\nDemo completed!")
    print("The trained PINN model successfully predicts the heat equation solution.")


if __name__ == "__main__":
    main()