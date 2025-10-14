"""
Example: PINN solution for stationary heat equation on a square plate with a circular hole.

This reproduces (in spirit) the study referenced in the prompt: the domain is
[-b,b] x [-b,b] with inner hole radius a. The exact solution used is
T(x,y) = (sqrt(x^2 + y^2) - a)^2 for r >= a and boundary conditions are set
accordingly.

The script trains a PINN using Adam and optionally refines with L-BFGS.
"""

import sys
import os
import time
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.tri as mtri

from pinn import PINN
from pde_problems import PlateWithHole
from utils import save_results


def train_plate_with_hole(refine_with_lbfgs: bool = True,
                          bc_weight: float = 10.0,
                          epochs: int = 10000,
                          lr: float = 0.015,
                          clip_grad: float = None,
                          bc_schedule: dict = None,
                          bc_inner_weight: float = None,
                          bc_outer_weight: float = None,
                          bc_neumann_weight: float = None):
    torch.manual_seed(42)
    np.random.seed(42)

    problem = PlateWithHole(a=0.2, b=1.0, n_collocation=3600, n_boundary=1300)

    # Create a PINN suitable for 2D stationary PDE (inputs: x, y)
    # slightly increased capacity and different activation (gelu)
    model = PINN(input_dim=2, hidden_layers=[64]*4, output_dim=1, activation='gelu')

    # Training settings
    epochs = epochs
    optimizer = torch.optim.Adam(model.parameters(), lr=float(lr))

    losses = []
    t0 = time.time()
    for epoch in range(epochs):
        optimizer.zero_grad()
        # support a two-stage BC schedule with keys:
        # {'start_weight': float, 'switch_epoch': int, 'new_weight': float}
        # If bc_schedule is not provided, fall back to scalar bc_weight.
        if bc_schedule is not None and isinstance(bc_schedule, dict):
            start_w = float(bc_schedule.get('start_weight', bc_weight))
            switch_epoch = int(bc_schedule.get('switch_epoch', epochs + 1))
            new_w = float(bc_schedule.get('new_weight', bc_weight))
            current_bc_weight = new_w if epoch >= switch_epoch else start_w
        else:
            current_bc_weight = float(bc_weight)

        # compute PDE loss and boundary components each epoch
        pde_l = problem.pde_loss(model)
        bcs = problem.boundary_losses(model)
        inner_w = float(bc_inner_weight) if bc_inner_weight is not None else current_bc_weight
        outer_w = float(bc_outer_weight) if bc_outer_weight is not None else current_bc_weight
        neu_w = float(bc_neumann_weight) if bc_neumann_weight is not None else current_bc_weight
        bc_l = inner_w * bcs['inner_dirichlet'] + outer_w * bcs['outer_dirichlet'] + neu_w * bcs['outer_neumann']
        loss = pde_l + bc_l  # combined BC-weighted loss
        loss.backward()
        # optional gradient clipping
        if clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(clip_grad))
        optimizer.step()

        losses.append(loss.item())
        if epoch % 500 == 0:
            print(f"Epoch {epoch}/{epochs} | Loss: {loss.item():.4e} | PDE: {pde_l.item():.4e} | BC: {bc_l.item():.4e}")

    t_adam = time.time() - t0
    print(f"Adam training completed in {t_adam:.1f} s")

    # Optional L-BFGS refinement
    t_lbfgs = 0.0
    if refine_with_lbfgs:
        def closure():
            optimizer_lbfgs.zero_grad()
            pde_l = problem.pde_loss(model)
            bcs = problem.boundary_losses(model)
            # use final BC weight after schedule (if any)
            if bc_schedule is not None and isinstance(bc_schedule, dict):
                final_bc_weight = float(bc_schedule.get('new_weight', bc_weight))
            else:
                final_bc_weight = float(bc_weight)

            if bc_inner_weight is None and bc_outer_weight is None and bc_neumann_weight is None:
                bc_val = final_bc_weight * (bcs['inner_dirichlet'] + bcs['outer_dirichlet'] + bcs['outer_neumann'])
            else:
                inner_w = float(bc_inner_weight) if bc_inner_weight is not None else final_bc_weight
                outer_w = float(bc_outer_weight) if bc_outer_weight is not None else final_bc_weight
                neu_w = float(bc_neumann_weight) if bc_neumann_weight is not None else final_bc_weight
                bc_val = inner_w * bcs['inner_dirichlet'] + outer_w * bcs['outer_dirichlet'] + neu_w * bcs['outer_neumann']

            loss = pde_l + bc_val
            loss.backward()
            return loss

        optimizer_lbfgs = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=500, tolerance_grad=1e-9)
        t1 = time.time()
        optimizer_lbfgs.step(closure)
        t_lbfgs = time.time() - t1
        print(f"L-BFGS refinement completed in {t_lbfgs:.1f} s")

    total_time = t_adam + t_lbfgs

    # Evaluate on a test grid
    test_pts = problem.sample_test_grid(n=100)
    with torch.no_grad():
        pred = model.predict(test_pts).squeeze()
        exact = problem.exact_solution(test_pts).squeeze()
    error = torch.sqrt(torch.mean((pred - exact) ** 2)).item()
    print(f"RMSE on test grid: {error:.6e}")

    # Save results and plots
    out_dir = "results/plate_with_hole"
    os.makedirs(out_dir, exist_ok=True)

    # Loss plot
    plt.figure()
    plt.semilogy(losses)
    plt.title('Training Loss (Adam)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, 'plate_hole_loss_adam.png'), dpi=200)
    plt.close()

    # Solution plot (scatter)
    X = test_pts[:, 0].numpy()
    Y = test_pts[:, 1].numpy()
    Z_pred = pred.numpy()
    Z_exact = exact.numpy()
    Z_err = np.abs(Z_pred - Z_exact)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.tricontourf(X, Y, Z_pred, levels=50, cmap='viridis')
    plt.title('PINN Solution')
    plt.colorbar()

    plt.subplot(1, 3, 2)
    plt.tricontourf(X, Y, Z_exact, levels=50, cmap='viridis')
    plt.title('Exact Solution')
    plt.colorbar()

    plt.subplot(1, 3, 3)
    plt.tricontourf(X, Y, Z_err, levels=50, cmap='Reds')
    plt.title('Absolute Error')
    plt.colorbar()

    plt.savefig(os.path.join(out_dir, 'plate_hole_solution_and_error.png'), dpi=200)
    plt.close()

    # 3D surface plots (triangular surface since domain has hole)
    try:
        fig = plt.figure(figsize=(18, 5))
        ax1 = fig.add_subplot(1, 3, 1, projection='3d')
        tri = mtri.Triangulation(X, Y)
        ax1.plot_trisurf(tri, Z_pred, cmap='viridis', linewidth=0.2)
        ax1.set_title('PINN Solution (3D)')
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')

        ax2 = fig.add_subplot(1, 3, 2, projection='3d')
        ax2.plot_trisurf(tri, Z_exact, cmap='viridis', linewidth=0.2)
        ax2.set_title('Exact Solution (3D)')
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')

        ax3 = fig.add_subplot(1, 3, 3, projection='3d')
        ax3.plot_trisurf(tri, Z_err, cmap='Reds', linewidth=0.2)
        ax3.set_title('Absolute Error (3D)')
        ax3.set_xlabel('x')
        ax3.set_ylabel('y')

        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'plate_hole_solution_and_error_3d.png'), dpi=200)
        plt.close()
    except Exception as e:
        print(f"Could not create 3D plots: {e}")

    # Save model and simple metrics file
    model.save_model(os.path.join(out_dir, 'plate_hole_model.pth'))
    with open(os.path.join(out_dir, 'plate_hole_metrics.txt'), 'w') as f:
        f.write(f"RMSE: {error:.6e}\n")
        f.write(f"Time (Adam): {t_adam:.2f} s\n")
        f.write(f"Time (L-BFGS): {t_lbfgs:.2f} s\n")
        f.write(f"Total time: {total_time:.2f} s\n")

    print(f"Saved results to {out_dir}")


if __name__ == '__main__':
    train_plate_with_hole(refine_with_lbfgs=True)
