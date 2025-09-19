"""
Utility functions for PINN training and analysis.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Optional, Dict
import os


def plot_loss_history(losses: List[float], save_path: Optional[str] = None, title: str = "Training Loss"):
    """
    Plot the training loss history.
    
    Args:
        losses: List of loss values during training
        save_path: Path to save the plot (optional)
        title: Title for the plot
    """
    plt.figure(figsize=(10, 6))
    plt.semilogy(losses)
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss (log scale)')
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_solution_1d(model, problem, t_values: List[float] = [0.0, 0.25, 0.5, 0.75, 1.0],
                    n_points: int = 100, save_path: Optional[str] = None):
    """
    Plot 1D solution at different time instances.
    
    Args:
        model: Trained PINN model
        problem: PDE problem instance
        t_values: Time values to plot
        n_points: Number of spatial points
        save_path: Path to save the plot
    """
    x_min, x_max = problem.domain['x']
    x_plot = torch.linspace(x_min, x_max, n_points).reshape(-1, 1)
    
    plt.figure(figsize=(12, 8))
    
    for i, t in enumerate(t_values):
        t_plot = torch.ones_like(x_plot) * t
        x_test = torch.cat([x_plot, t_plot], dim=1)
        
        # Predicted solution
        u_pred = model.predict(x_test).detach().numpy()
        
        plt.subplot(2, 3, i + 1)
        plt.plot(x_plot.numpy(), u_pred, 'b-', linewidth=2, label='PINN')
        
        # Plot exact solution if available
        try:
            u_exact = problem.exact_solution(x_test).detach().numpy()
            plt.plot(x_plot.numpy(), u_exact, 'r--', linewidth=2, label='Exact')
            plt.legend()
        except:
            pass
        
        plt.title(f't = {t:.2f}')
        plt.xlabel('x')
        plt.ylabel('u(x,t)')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_solution_2d(model, problem, save_path: Optional[str] = None, resolution: int = 100):
    """
    Plot 2D heatmap of the solution in space-time domain.
    
    Args:
        model: Trained PINN model
        problem: PDE problem instance
        save_path: Path to save the plot
        resolution: Grid resolution for plotting
    """
    x_min, x_max = problem.domain['x']
    t_min, t_max = problem.domain['t']
    
    # Create meshgrid
    x = np.linspace(x_min, x_max, resolution)
    t = np.linspace(t_min, t_max, resolution)
    X, T = np.meshgrid(x, t)
    
    # Flatten and convert to tensor
    x_flat = torch.tensor(X.flatten(), dtype=torch.float32).reshape(-1, 1)
    t_flat = torch.tensor(T.flatten(), dtype=torch.float32).reshape(-1, 1)
    xt = torch.cat([x_flat, t_flat], dim=1)
    
    # Predict solution
    u_pred = model.predict(xt).detach().numpy().reshape(resolution, resolution)
    
    plt.figure(figsize=(15, 5))
    
    # PINN solution
    plt.subplot(1, 3, 1)
    plt.contourf(X, T, u_pred, levels=50, cmap='viridis')
    plt.colorbar(label='u(x,t)')
    plt.title('PINN Solution')
    plt.xlabel('x')
    plt.ylabel('t')
    
    # Exact solution (if available)
    try:
        u_exact = problem.exact_solution(xt).detach().numpy().reshape(resolution, resolution)
        
        plt.subplot(1, 3, 2)
        plt.contourf(X, T, u_exact, levels=50, cmap='viridis')
        plt.colorbar(label='u(x,t)')
        plt.title('Exact Solution')
        plt.xlabel('x')
        plt.ylabel('t')
        
        # Error plot
        plt.subplot(1, 3, 3)
        error = np.abs(u_pred - u_exact)
        plt.contourf(X, T, error, levels=50, cmap='Reds')
        plt.colorbar(label='|Error|')
        plt.title('Absolute Error')
        plt.xlabel('x')
        plt.ylabel('t')
        
    except:
        # If no exact solution, just plot PINN solution
        pass
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def compute_error_metrics(model, problem, n_test_points: int = 1000) -> Dict[str, float]:
    """
    Compute error metrics for the trained model.
    
    Args:
        model: Trained PINN model
        problem: PDE problem instance
        n_test_points: Number of test points for evaluation
        
    Returns:
        Dictionary containing error metrics
    """
    try:
        # Generate test points
        x_min, x_max = problem.domain['x']
        t_min, t_max = problem.domain['t']
        
        x_test = torch.rand(n_test_points, 1) * (x_max - x_min) + x_min
        t_test = torch.rand(n_test_points, 1) * (t_max - t_min) + t_min
        xt_test = torch.cat([x_test, t_test], dim=1)
        
        # Compute predictions and exact solutions
        u_pred = model.predict(xt_test).detach()
        u_exact = problem.exact_solution(xt_test).detach()
        
        # Compute metrics
        l2_error = torch.mean((u_pred - u_exact) ** 2).item()
        l2_relative_error = (torch.mean((u_pred - u_exact) ** 2) / torch.mean(u_exact ** 2)).item()
        linf_error = torch.max(torch.abs(u_pred - u_exact)).item()
        
        return {
            'L2_error': l2_error,
            'L2_relative_error': l2_relative_error,
            'Linf_error': linf_error,
            'RMSE': np.sqrt(l2_error)
        }
    
    except Exception as e:
        print(f"Could not compute error metrics: {e}")
        return {}


def save_results(model, problem, losses: List[float], output_dir: str = "results"):
    """
    Save model, plots, and metrics to output directory.
    
    Args:
        model: Trained PINN model
        problem: PDE problem instance
        losses: Training loss history
        output_dir: Directory to save results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(output_dir, f"{problem.name.lower().replace(' ', '_')}_model.pth")
    model.save_model(model_path)
    
    # Save loss plot
    loss_path = os.path.join(output_dir, f"{problem.name.lower().replace(' ', '_')}_loss.png")
    plot_loss_history(losses, save_path=loss_path, title=f"{problem.name} - Training Loss")
    
    # Save solution plots
    if problem.name in ["Heat Equation", "Wave Equation", "Burgers Equation"]:
        # 1D plots
        sol_1d_path = os.path.join(output_dir, f"{problem.name.lower().replace(' ', '_')}_solution_1d.png")
        plot_solution_1d(model, problem, save_path=sol_1d_path)
        
        # 2D plot
        sol_2d_path = os.path.join(output_dir, f"{problem.name.lower().replace(' ', '_')}_solution_2d.png")
        plot_solution_2d(model, problem, save_path=sol_2d_path)
    
    # Compute and save error metrics
    metrics = compute_error_metrics(model, problem)
    if metrics:
        metrics_path = os.path.join(output_dir, f"{problem.name.lower().replace(' ', '_')}_metrics.txt")
        with open(metrics_path, 'w') as f:
            f.write(f"Error Metrics for {problem.name}\n")
            f.write("="*40 + "\n")
            for key, value in metrics.items():
                f.write(f"{key}: {value:.6e}\n")
    
    print(f"Results saved to {output_dir}")


def generate_training_data(problem, n_collocation: int = 1000, n_boundary: int = 100, 
                          n_initial: int = 100) -> Dict[str, torch.Tensor]:
    """
    Generate training data for a PDE problem.
    
    Args:
        problem: PDE problem instance
        n_collocation: Number of collocation points
        n_boundary: Number of boundary points
        n_initial: Number of initial condition points
        
    Returns:
        Dictionary containing training data points
    """
    x_min, x_max = problem.domain['x']
    t_min, t_max = problem.domain['t']
    
    # Collocation points (interior)
    x_col = torch.rand(n_collocation, 1) * (x_max - x_min) + x_min
    t_col = torch.rand(n_collocation, 1) * (t_max - t_min) + t_min
    collocation_points = torch.cat([x_col, t_col], dim=1)
    
    # Boundary points
    t_bc = torch.rand(n_boundary, 1) * (t_max - t_min) + t_min
    x_bc_left = torch.ones(n_boundary, 1) * x_min
    x_bc_right = torch.ones(n_boundary, 1) * x_max
    boundary_points_left = torch.cat([x_bc_left, t_bc], dim=1)
    boundary_points_right = torch.cat([x_bc_right, t_bc], dim=1)
    
    # Initial condition points
    x_ic = torch.rand(n_initial, 1) * (x_max - x_min) + x_min
    t_ic = torch.ones(n_initial, 1) * t_min
    initial_points = torch.cat([x_ic, t_ic], dim=1)
    
    return {
        'collocation': collocation_points,
        'boundary_left': boundary_points_left,
        'boundary_right': boundary_points_right,
        'initial': initial_points
    }