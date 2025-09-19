"""
Common PDE problems for testing PINN implementations.
"""

import torch
import numpy as np
from typing import Callable, Tuple, Dict


class PDEProblem:
    """Base class for PDE problems."""
    
    def __init__(self, domain: Dict, name: str = "Generic PDE"):
        self.domain = domain
        self.name = name
    
    def pde_loss(self, model) -> torch.Tensor:
        """Compute PDE residual loss."""
        raise NotImplementedError
    
    def boundary_loss(self, model) -> torch.Tensor:
        """Compute boundary condition loss."""
        raise NotImplementedError
    
    def initial_loss(self, model) -> torch.Tensor:
        """Compute initial condition loss."""
        raise NotImplementedError
    
    def exact_solution(self, x: torch.Tensor) -> torch.Tensor:
        """Exact solution if available."""
        raise NotImplementedError


class HeatEquation(PDEProblem):
    """
    1D Heat equation: u_t = alpha * u_xx
    Domain: x ∈ [0, L], t ∈ [0, T]
    
    Boundary conditions: u(0, t) = u(L, t) = 0
    Initial condition: u(x, 0) = sin(π*x/L)
    """
    
    def __init__(self, alpha: float = 0.1, L: float = 1.0, T: float = 1.0, 
                 n_collocation: int = 1000, n_boundary: int = 100, n_initial: int = 100):
        domain = {'x': [0, L], 't': [0, T]}
        super().__init__(domain, "Heat Equation")
        
        self.alpha = alpha
        self.L = L
        self.T = T
        self.n_collocation = n_collocation
        self.n_boundary = n_boundary
        self.n_initial = n_initial
        
        # Generate collocation points
        self.x_collocation = torch.rand(n_collocation, 1) * L
        self.t_collocation = torch.rand(n_collocation, 1) * T
        self.collocation_points = torch.cat([self.x_collocation, self.t_collocation], dim=1)
        
        # Generate boundary points
        self.x_boundary_left = torch.zeros(n_boundary, 1)
        self.x_boundary_right = torch.ones(n_boundary, 1) * L
        self.t_boundary = torch.rand(n_boundary, 1) * T
        
        # Generate initial condition points
        self.x_initial = torch.rand(n_initial, 1) * L
        self.t_initial = torch.zeros(n_initial, 1)
    
    def pde_loss(self, model) -> torch.Tensor:
        """Compute PDE residual: u_t - alpha * u_xx = 0"""
        x = self.collocation_points.clone().requires_grad_(True)
        
        derivatives = model.compute_derivatives(x, order=2)
        u_t = derivatives['u_1']  # derivative w.r.t. t
        u_xx = derivatives['u_00']  # second derivative w.r.t. x
        
        pde_residual = u_t - self.alpha * u_xx
        return torch.mean(pde_residual ** 2)
    
    def boundary_loss(self, model) -> torch.Tensor:
        """Compute boundary condition loss: u(0, t) = u(L, t) = 0"""
        # Left boundary
        x_left = torch.cat([self.x_boundary_left, self.t_boundary], dim=1)
        u_left = model(x_left)
        
        # Right boundary  
        x_right = torch.cat([self.x_boundary_right, self.t_boundary], dim=1)
        u_right = model(x_right)
        
        bc_loss = torch.mean(u_left ** 2) + torch.mean(u_right ** 2)
        return bc_loss
    
    def initial_loss(self, model) -> torch.Tensor:
        """Compute initial condition loss: u(x, 0) = sin(π*x/L)"""
        x_ic = torch.cat([self.x_initial, self.t_initial], dim=1)
        u_pred = model(x_ic)
        u_exact = torch.sin(np.pi * self.x_initial / self.L)
        
        return torch.mean((u_pred - u_exact) ** 2)
    
    def exact_solution(self, x: torch.Tensor) -> torch.Tensor:
        """Exact solution: u(x,t) = sin(π*x/L) * exp(-α*π²*t/L²)"""
        x_coord = x[:, 0:1]
        t_coord = x[:, 1:2]
        return torch.sin(np.pi * x_coord / self.L) * torch.exp(-self.alpha * (np.pi / self.L) ** 2 * t_coord)


class WaveEquation(PDEProblem):
    """
    1D Wave equation: u_tt = c² * u_xx
    Domain: x ∈ [0, L], t ∈ [0, T]
    
    Boundary conditions: u(0, t) = u(L, t) = 0
    Initial conditions: u(x, 0) = sin(π*x/L), u_t(x, 0) = 0
    """
    
    def __init__(self, c: float = 1.0, L: float = 1.0, T: float = 2.0,
                 n_collocation: int = 1000, n_boundary: int = 100, n_initial: int = 100):
        domain = {'x': [0, L], 't': [0, T]}
        super().__init__(domain, "Wave Equation")
        
        self.c = c
        self.L = L
        self.T = T
        self.n_collocation = n_collocation
        self.n_boundary = n_boundary
        self.n_initial = n_initial
        
        # Generate points (similar to heat equation)
        self.x_collocation = torch.rand(n_collocation, 1) * L
        self.t_collocation = torch.rand(n_collocation, 1) * T
        self.collocation_points = torch.cat([self.x_collocation, self.t_collocation], dim=1)
        
        self.x_boundary_left = torch.zeros(n_boundary, 1)
        self.x_boundary_right = torch.ones(n_boundary, 1) * L
        self.t_boundary = torch.rand(n_boundary, 1) * T
        
        self.x_initial = torch.rand(n_initial, 1) * L
        self.t_initial = torch.zeros(n_initial, 1)
    
    def pde_loss(self, model) -> torch.Tensor:
        """Compute PDE residual: u_tt - c² * u_xx = 0"""
        x = self.collocation_points.clone().requires_grad_(True)
        
        derivatives = model.compute_derivatives(x, order=2)
        u_tt = derivatives['u_11']  # second derivative w.r.t. t
        u_xx = derivatives['u_00']  # second derivative w.r.t. x
        
        pde_residual = u_tt - (self.c ** 2) * u_xx
        return torch.mean(pde_residual ** 2)
    
    def boundary_loss(self, model) -> torch.Tensor:
        """Compute boundary condition loss: u(0, t) = u(L, t) = 0"""
        x_left = torch.cat([self.x_boundary_left, self.t_boundary], dim=1)
        u_left = model(x_left)
        
        x_right = torch.cat([self.x_boundary_right, self.t_boundary], dim=1)
        u_right = model(x_right)
        
        return torch.mean(u_left ** 2) + torch.mean(u_right ** 2)
    
    def initial_loss(self, model) -> torch.Tensor:
        """Compute initial conditions: u(x, 0) = sin(π*x/L), u_t(x, 0) = 0"""
        x_ic = torch.cat([self.x_initial, self.t_initial], dim=1).requires_grad_(True)
        
        # Initial displacement
        u_pred = model(x_ic)
        u_exact = torch.sin(np.pi * self.x_initial / self.L)
        ic_displacement = torch.mean((u_pred - u_exact) ** 2)
        
        # Initial velocity
        derivatives = model.compute_derivatives(x_ic, order=1)
        u_t_pred = derivatives['u_1']
        ic_velocity = torch.mean(u_t_pred ** 2)
        
        return ic_displacement + ic_velocity
    
    def exact_solution(self, x: torch.Tensor) -> torch.Tensor:
        """Exact solution: u(x,t) = sin(π*x/L) * cos(π*c*t/L)"""
        x_coord = x[:, 0:1]
        t_coord = x[:, 1:2]
        return torch.sin(np.pi * x_coord / self.L) * torch.cos(np.pi * self.c * t_coord / self.L)


class BurgersEquation(PDEProblem):
    """
    1D Viscous Burgers equation: u_t + u * u_x = ν * u_xx
    Domain: x ∈ [-1, 1], t ∈ [0, T]
    """
    
    def __init__(self, nu: float = 0.01, T: float = 1.0,
                 n_collocation: int = 1000, n_boundary: int = 100, n_initial: int = 100):
        domain = {'x': [-1, 1], 't': [0, T]}
        super().__init__(domain, "Burgers Equation")
        
        self.nu = nu
        self.T = T
        self.n_collocation = n_collocation
        self.n_boundary = n_boundary
        self.n_initial = n_initial
        
        # Generate collocation points
        self.x_collocation = torch.rand(n_collocation, 1) * 2 - 1  # [-1, 1]
        self.t_collocation = torch.rand(n_collocation, 1) * T
        self.collocation_points = torch.cat([self.x_collocation, self.t_collocation], dim=1)
        
        # Boundary points
        self.x_boundary_left = -torch.ones(n_boundary, 1)
        self.x_boundary_right = torch.ones(n_boundary, 1)
        self.t_boundary = torch.rand(n_boundary, 1) * T
        
        # Initial points
        self.x_initial = torch.rand(n_initial, 1) * 2 - 1
        self.t_initial = torch.zeros(n_initial, 1)
    
    def pde_loss(self, model) -> torch.Tensor:
        """Compute PDE residual: u_t + u * u_x - ν * u_xx = 0"""
        x = self.collocation_points.clone().requires_grad_(True)
        
        derivatives = model.compute_derivatives(x, order=2)
        u = derivatives['u']
        u_t = derivatives['u_1']
        u_x = derivatives['u_0']
        u_xx = derivatives['u_00']
        
        pde_residual = u_t + u * u_x - self.nu * u_xx
        return torch.mean(pde_residual ** 2)
    
    def boundary_loss(self, model) -> torch.Tensor:
        """Compute periodic boundary conditions: u(-1, t) = u(1, t)"""
        x_left = torch.cat([self.x_boundary_left, self.t_boundary], dim=1)
        u_left = model(x_left)
        
        x_right = torch.cat([self.x_boundary_right, self.t_boundary], dim=1)
        u_right = model(x_right)
        
        return torch.mean((u_left - u_right) ** 2)
    
    def initial_loss(self, model) -> torch.Tensor:
        """Compute initial condition: u(x, 0) = -sin(π*x)"""
        x_ic = torch.cat([self.x_initial, self.t_initial], dim=1)
        u_pred = model(x_ic)
        u_exact = -torch.sin(np.pi * self.x_initial)
        
        return torch.mean((u_pred - u_exact) ** 2)