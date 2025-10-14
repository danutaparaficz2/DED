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


class HeatEquation2D(PDEProblem):
    """
    2D Heat equation: u_t = alpha * (u_xx + u_yy)
    Domain: x,y ∈ [0, L], t ∈ [0, T]

    Boundary conditions: u = 0 on spatial boundaries
    Initial condition: u(x,y,0) = sin(π*x/L) * sin(π*y/L)
    """

    def __init__(self, alpha: float = 0.1, L: float = 1.0, T: float = 1.0,
                 n_collocation: int = 2000, n_boundary: int = 400, n_initial: int = 400):
        domain = {'x': [0, L], 'y': [0, L], 't': [0, T]}
        super().__init__(domain, "2D Heat Equation")

        self.alpha = alpha
        self.L = L
        self.T = T
        self.n_collocation = n_collocation
        self.n_boundary = n_boundary
        self.n_initial = n_initial

        # collocation (interior) points: sample x,y in [0,L], t in [0,T]
        self.x_collocation = torch.rand(n_collocation, 1) * L
        self.y_collocation = torch.rand(n_collocation, 1) * L
        self.t_collocation = torch.rand(n_collocation, 1) * T
        self.collocation_points = torch.cat([self.x_collocation, self.y_collocation, self.t_collocation], dim=1)

        # boundary points on the 4 edges (x=0, x=L, y=0, y=L)
        n_per_edge = max(1, n_boundary // 4)
        # x = 0
        x0 = torch.zeros(n_per_edge, 1)
        y0 = torch.rand(n_per_edge, 1) * L
        t0 = torch.rand(n_per_edge, 1) * T
        left = torch.cat([x0, y0, t0], dim=1)
        # x = L
        xL = torch.ones(n_per_edge, 1) * L
        yL = torch.rand(n_per_edge, 1) * L
        tL = torch.rand(n_per_edge, 1) * T
        right = torch.cat([xL, yL, tL], dim=1)
        # y = 0
        y0b = torch.zeros(n_per_edge, 1)
        x0b = torch.rand(n_per_edge, 1) * L
        t0b = torch.rand(n_per_edge, 1) * T
        bottom = torch.cat([x0b, y0b, t0b], dim=1)
        # y = L
        yLb = torch.ones(n_per_edge, 1) * L
        xLb = torch.rand(n_per_edge, 1) * L
        tLb = torch.rand(n_per_edge, 1) * T
        top = torch.cat([xLb, yLb, tLb], dim=1)

        self.boundary_points = torch.cat([left, right, bottom, top], dim=0)

        # initial condition points (t=0)
        self.x_initial = torch.rand(n_initial, 1) * L
        self.y_initial = torch.rand(n_initial, 1) * L
        self.t_initial = torch.zeros(n_initial, 1)
        self.initial_points = torch.cat([self.x_initial, self.y_initial, self.t_initial], dim=1)

    def pde_loss(self, model) -> torch.Tensor:
        pts = self.collocation_points.clone().requires_grad_(True)
        x = pts[:, 0:1]
        y = pts[:, 1:2]
        t = pts[:, 2:3]

        derivatives = model.compute_derivatives(pts, order=2)
        # indices: 0 -> x, 1 -> y, 2 -> t
        u_t = derivatives.get('u_2')
        u_xx = derivatives.get('u_00')
        u_yy = derivatives.get('u_11')
        if u_t is None:
            u_t = torch.zeros_like(pts[:, 0:1])
        if u_xx is None:
            u_xx = torch.zeros_like(pts[:, 0:1])
        if u_yy is None:
            u_yy = torch.zeros_like(pts[:, 0:1])

        residual = u_t - self.alpha * (u_xx + u_yy)
        return torch.mean(residual ** 2)

    def boundary_loss(self, model) -> torch.Tensor:
        pts = self.boundary_points.clone()
        u = model(pts)
        return torch.mean(u ** 2)

    def initial_loss(self, model) -> torch.Tensor:
        pts = self.initial_points.clone()
        u_pred = model(pts)
        u_exact = torch.sin(np.pi * self.x_initial / self.L) * torch.sin(np.pi * self.y_initial / self.L)
        return torch.mean((u_pred - u_exact) ** 2)

    def exact_solution(self, x: torch.Tensor) -> torch.Tensor:
        # x is (N,3) with columns (x,y,t)
        x_coord = x[:, 0:1]
        y_coord = x[:, 1:2]
        t_coord = x[:, 2:3]
        return (torch.sin(np.pi * x_coord / self.L) * torch.sin(np.pi * y_coord / self.L) *
                torch.exp(-2 * (np.pi / self.L) ** 2 * self.alpha * t_coord))


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


class HeatEquation3D(PDEProblem):
    """
    3D Heat equation: ∂T/∂t = α * (∂²T/∂x² + ∂²T/∂y² + ∂²T/∂z²)
    Domain: x,y,z ∈ [0, L], t ∈ [0, T]
    
    Boundary conditions: T = 0 on all boundaries
    Initial condition: T(x, y, z, 0) = sin(π*x/L) * sin(π*y/L) * sin(π*z/L)
    """
    
    def __init__(self, alpha: float = 1.0, L: float = 1.0, T: float = 1.0,
                 n_collocation: int = 2000, n_boundary: int = 200, n_initial: int = 200):
        domain = {'x': [0, L], 'y': [0, L], 'z': [0, L], 't': [0, T]}
        super().__init__(domain, "3D Heat Equation")
        
        self.alpha = alpha
        self.L = L
        self.T = T
        self.n_collocation = n_collocation
        self.n_boundary = n_boundary
        self.n_initial = n_initial
        
        # Generate collocation points (interior points)
        self.x_collocation = torch.rand(n_collocation, 1) * L
        self.y_collocation = torch.rand(n_collocation, 1) * L
        self.z_collocation = torch.rand(n_collocation, 1) * L
        self.t_collocation = torch.rand(n_collocation, 1) * T
        self.collocation_points = torch.cat([
            self.x_collocation, self.y_collocation, 
            self.z_collocation, self.t_collocation
        ], dim=1)
        
        # Generate boundary points (6 faces of the cube)
        self._generate_boundary_points()
        
        # Generate initial condition points
        self.x_initial = torch.rand(n_initial, 1) * L
        self.y_initial = torch.rand(n_initial, 1) * L
        self.z_initial = torch.rand(n_initial, 1) * L
        self.t_initial = torch.zeros(n_initial, 1)
    
    def _generate_boundary_points(self):
        """Generate points on all 6 faces of the cube"""
        n_per_face = self.n_boundary // 6
        
        # Face 1: x = 0
        x1 = torch.zeros(n_per_face, 1)
        y1 = torch.rand(n_per_face, 1) * self.L
        z1 = torch.rand(n_per_face, 1) * self.L
        t1 = torch.rand(n_per_face, 1) * self.T
        
        # Face 2: x = L
        x2 = torch.ones(n_per_face, 1) * self.L
        y2 = torch.rand(n_per_face, 1) * self.L
        z2 = torch.rand(n_per_face, 1) * self.L
        t2 = torch.rand(n_per_face, 1) * self.T
        
        # Face 3: y = 0
        x3 = torch.rand(n_per_face, 1) * self.L
        y3 = torch.zeros(n_per_face, 1)
        z3 = torch.rand(n_per_face, 1) * self.L
        t3 = torch.rand(n_per_face, 1) * self.T
        
        # Face 4: y = L
        x4 = torch.rand(n_per_face, 1) * self.L
        y4 = torch.ones(n_per_face, 1) * self.L
        z4 = torch.rand(n_per_face, 1) * self.L
        t4 = torch.rand(n_per_face, 1) * self.T
        
        # Face 5: z = 0
        x5 = torch.rand(n_per_face, 1) * self.L
        y5 = torch.rand(n_per_face, 1) * self.L
        z5 = torch.zeros(n_per_face, 1)
        t5 = torch.rand(n_per_face, 1) * self.T
        
        # Face 6: z = L
        x6 = torch.rand(n_per_face, 1) * self.L
        y6 = torch.rand(n_per_face, 1) * self.L
        z6 = torch.ones(n_per_face, 1) * self.L
        t6 = torch.rand(n_per_face, 1) * self.T
        
        # Combine all boundary points
        self.boundary_points = torch.cat([
            torch.cat([x1, y1, z1, t1], dim=1),
            torch.cat([x2, y2, z2, t2], dim=1),
            torch.cat([x3, y3, z3, t3], dim=1),
            torch.cat([x4, y4, z4, t4], dim=1),
            torch.cat([x5, y5, z5, t5], dim=1),
            torch.cat([x6, y6, z6, t6], dim=1)
        ], dim=0)
    
    def pde_loss(self, model) -> torch.Tensor:
        """Compute PDE residual: ∂T/∂t - α * (∂²T/∂x² + ∂²T/∂y² + ∂²T/∂z²) = 0"""
        points = self.collocation_points.clone()
        x, y, z, t = points.split(1, dim=1)
        
        # Enable gradient tracking
        x.requires_grad_(True)
        y.requires_grad_(True)
        z.requires_grad_(True)
        t.requires_grad_(True)
        
        # Get model prediction
        T_pred = model(torch.cat([x, y, z, t], dim=1))
        
        # Calculate first-order derivatives
        dT_dt = torch.autograd.grad(T_pred, t, grad_outputs=torch.ones_like(T_pred), 
                                   create_graph=True)[0]
        dT_dx = torch.autograd.grad(T_pred, x, grad_outputs=torch.ones_like(T_pred), 
                                   create_graph=True)[0]
        dT_dy = torch.autograd.grad(T_pred, y, grad_outputs=torch.ones_like(T_pred), 
                                   create_graph=True)[0]
        dT_dz = torch.autograd.grad(T_pred, z, grad_outputs=torch.ones_like(T_pred), 
                                   create_graph=True)[0]
        
        # Calculate second-order derivatives
        dT_dxx = torch.autograd.grad(dT_dx, x, grad_outputs=torch.ones_like(dT_dx), 
                                    create_graph=True)[0]
        dT_dyy = torch.autograd.grad(dT_dy, y, grad_outputs=torch.ones_like(dT_dy), 
                                    create_graph=True)[0]
        dT_dzz = torch.autograd.grad(dT_dz, z, grad_outputs=torch.ones_like(dT_dz), 
                                    create_graph=True)[0]
        
        # PDE residual: ∂T/∂t - α * (∂²T/∂x² + ∂²T/∂y² + ∂²T/∂z²) = 0
        pde_residual = dT_dt - self.alpha * (dT_dxx + dT_dyy + dT_dzz)
        
        return torch.mean(pde_residual ** 2)
    
    def boundary_loss(self, model) -> torch.Tensor:
        """Compute boundary condition loss: T = 0 on all boundaries"""
        T_boundary = model(self.boundary_points)
        return torch.mean(T_boundary ** 2)
    
    def initial_loss(self, model) -> torch.Tensor:
        """Compute initial condition loss: T(x,y,z,0) = sin(π*x/L)*sin(π*y/L)*sin(π*z/L)"""
        x_ic = torch.cat([self.x_initial, self.y_initial, self.z_initial, self.t_initial], dim=1)
        T_pred = model(x_ic)
        
        T_exact = (torch.sin(np.pi * self.x_initial / self.L) * 
                  torch.sin(np.pi * self.y_initial / self.L) * 
                  torch.sin(np.pi * self.z_initial / self.L))
        
        return torch.mean((T_pred - T_exact) ** 2)
    
    def exact_solution(self, x: torch.Tensor) -> torch.Tensor:
        """
        Exact solution for the 3D heat equation with given initial and boundary conditions.
        T(x,y,z,t) = sin(π*x/L)*sin(π*y/L)*sin(π*z/L)*exp(-3*π²*α*t/L²)
        """
        x_coord, y_coord, z_coord, t_coord = x.split(1, dim=1)
        
        return (torch.sin(np.pi * x_coord / self.L) * 
                torch.sin(np.pi * y_coord / self.L) * 
                torch.sin(np.pi * z_coord / self.L) * 
                torch.exp(-3 * np.pi**2 * self.alpha * t_coord / self.L**2))


class PlateWithHole(PDEProblem):
    """
    Stationary 2D heat (Poisson) problem on a square plate with a circular hole.

    Analytical solution (used to build source term and BCs):
        T(x,y) = (sqrt(x^2 + y^2) - a)**2,    for r >= a

    Domain: square [-b, b] x [-b, b] with a circular hole of radius a at origin.

        Boundary conditions used (following the reference):
            - Dirichlet on the inner circle (r = a): T = 0
            - Dirichlet on the vertical outer edges x = +/- b: T(x=±b,y) = (sqrt(b^2 + y^2) - a)^2
            - Neumann on the horizontal outer edges y = +/- b: dT/dy(x,y=±b) = analytic_neumann
    """

    def __init__(self, a: float = 0.2, b: float = 1.0,
                 n_collocation: int = 3600, n_boundary: int = 1300,
                 outer_vertical_fraction: float = 0.6,
                 n_collocation_outer: int = 0,
                 outer_band_width: float = 0.15):
        domain = {'x': [-b, b], 'y': [-b, b]}
        super().__init__(domain, "Plate with Hole")

        self.a = a
        self.b = b
        self.n_collocation = n_collocation
        self.n_boundary = n_boundary
        # fraction of remaining boundary points allocated to outer vertical edges
        self.outer_vertical_fraction = float(outer_vertical_fraction)
        # number of extra collocation points to sample near outer edges/corners
        self.n_collocation_outer = int(n_collocation_outer)
        # width of the outer sampling band (fraction of b)
        self.outer_band_width = float(outer_band_width)

        # Generate collocation points in square but exclude interior of the hole
        self.collocation_points = self._sample_domain_points(n_collocation)
        # Optionally add focused collocation points near the outer boundary (corners/edges)
        if self.n_collocation_outer and self.n_collocation_outer > 0:
            extra = self._sample_outer_band(self.n_collocation_outer, band_width=self.outer_band_width)
            if extra.numel() > 0:
                self.collocation_points = torch.cat([self.collocation_points, extra], dim=0)

        # Boundary points: inner circle (Dirichlet), outer square edges
        self._generate_boundary_points()

    def _sample_domain_points(self, n_points: int):
        pts = []
        # Draw in batches until we have enough points outside the hole
        batch = max(1024, n_points)
        while len(pts) < n_points:
            xy = (torch.rand(batch, 2) * (self.b - (-self.b)) + (-self.b))
            r = torch.sqrt(xy[:, 0:1] ** 2 + xy[:, 1:2] ** 2)
            mask = (r >= self.a).squeeze()
            valid = xy[mask]
            if valid.numel() > 0:
                pts.append(valid)
            # limit growth
            if sum(p.size(0) for p in pts) > 10 * n_points:
                break
        if len(pts) == 0:
            return torch.zeros(0, 2)
        pts = torch.cat(pts, dim=0)[:n_points]
        return pts

    def _sample_outer_band(self, n_points: int, band_width: float = 0.15):
        """Sample additional collocation points in an outer band of width (band_width * b)."""
        band = band_width * self.b
        pts = []
        batch = max(1024, n_points)
        attempts = 0
        while len(pts) < n_points and attempts < 100:
            xy = (torch.rand(batch, 2) * (self.b - (-self.b)) + (-self.b))
            r = torch.sqrt(xy[:, 0:1] ** 2 + xy[:, 1:2] ** 2).squeeze()
            # select points in outer band: b - band <= max(|x|,|y|) <= b and outside hole
            mask_outer = (torch.max(torch.abs(xy), dim=1).values >= (self.b - band))
            mask_hole = (r >= self.a)
            mask = (mask_outer & mask_hole)
            valid = xy[mask]
            if valid.numel() > 0:
                pts.append(valid)
            attempts += 1
        if len(pts) == 0:
            return torch.zeros(0, 2)
        pts = torch.cat(pts, dim=0)[:n_points]
        return pts

    def _generate_boundary_points(self):
        # Inner circle (Dirichlet)
        # allocate a balanced fraction: more points to outer vertical Dirichlet edges
        # Default split: inner 40%, outer vertical 40%, outer horizontal 20%
        n_inner = max(10, int(self.n_boundary * 0.4))
        theta = torch.rand(n_inner, 1) * 2 * np.pi
        x_in = self.a * torch.cos(theta)
        y_in = self.a * torch.sin(theta)
        inner = torch.cat([x_in, y_in], dim=1)
        # Outer square edges: split remaining points between vertical (Dirichlet)
        # and horizontal (Neumann) edges
        remaining = max(0, self.n_boundary - n_inner)
        # allow caller to control fraction allocated to vertical outer edges
        n_vertical = max(1, int(remaining * float(getattr(self, 'outer_vertical_fraction', 0.6))))  # left+right combined
        n_horizontal = max(1, remaining - n_vertical)  # bottom+top combined

        # allocate equally between left and right
        n_per_vertical_edge = max(1, n_vertical // 2)
        y_lr = torch.rand(n_per_vertical_edge, 1) * (2 * self.b) - self.b
        left = torch.cat([torch.ones(n_per_vertical_edge, 1) * (-self.b), y_lr], dim=1)
        right = torch.cat([torch.ones(n_per_vertical_edge, 1) * self.b, y_lr], dim=1)

        # allocate equally between bottom and top
        n_per_horizontal_edge = max(1, n_horizontal // 2)
        x_tb = torch.rand(n_per_horizontal_edge, 1) * (2 * self.b) - self.b
        bottom = torch.cat([x_tb, torch.ones(n_per_horizontal_edge, 1) * (-self.b)], dim=1)
        top = torch.cat([x_tb, torch.ones(n_per_horizontal_edge, 1) * self.b], dim=1)

        self.boundary_inner = inner
        self.boundary_dirichlet = torch.cat([left, right], dim=0)
        self.boundary_neumann = torch.cat([bottom, top], dim=0)

    def boundary_losses(self, model) -> dict:
        """Return separate boundary loss components: inner_dirichlet, outer_dirichlet, outer_neumann."""
        losses = {}

        # Inner circle Dirichlet: T = 0
        if self.boundary_inner is not None and self.boundary_inner.size(0) > 0:
            u_in = model(self.boundary_inner)
            losses['inner_dirichlet'] = torch.mean((u_in - 0.0) ** 2)
        else:
            losses['inner_dirichlet'] = torch.tensor(0.0)

        # Outer vertical edges Dirichlet: T = 0
        if self.boundary_dirichlet is not None and self.boundary_dirichlet.size(0) > 0:
            pts = self.boundary_dirichlet
            u_dir = model(pts)
            x = pts[:, 0:1]
            y = pts[:, 1:2]
            r = torch.sqrt(x ** 2 + y ** 2)
            # analytic Dirichlet value on outer vertical edges
            analytic_dir = (r - self.a) ** 2
            losses['outer_dirichlet'] = torch.mean((u_dir - analytic_dir) ** 2)
        else:
            losses['outer_dirichlet'] = torch.tensor(0.0)

        # Outer horizontal edges Neumann: analytic dT/dy
        if self.boundary_neumann is not None and self.boundary_neumann.size(0) > 0:
            pts = self.boundary_neumann.clone().requires_grad_(True)
            x = pts[:, 0:1]
            y = pts[:, 1:2]
            u = model(pts)
            u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True, allow_unused=True)[0]
            if u_y is None:
                u_y = torch.zeros_like(u)
            r = torch.sqrt(x ** 2 + y ** 2)
            r = torch.where(r == 0.0, torch.ones_like(r) * 1e-6, r)
            # analytic Neumann (dT/dy) = 2*(r - a) * (y / r)
            analytic_neu = 2.0 * (r - self.a) * (y / r)
            losses['outer_neumann'] = torch.mean((u_y - analytic_neu) ** 2)
        else:
            losses['outer_neumann'] = torch.tensor(0.0)

        return losses

    def pde_loss(self, model) -> torch.Tensor:
        """PDE residual: u_xx + u_yy - source(x,y) = 0 where source = 4 - 2a/r"""
        pts = self.collocation_points.clone().requires_grad_(True)
        x = pts[:, 0:1]
        y = pts[:, 1:2]

        u = model(torch.cat([x, y], dim=1))

        # second derivatives
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True, allow_unused=True)[0]
        if u_x is None:
            u_x = torch.zeros_like(u)
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True, allow_unused=True)[0]
        if u_xx is None:
            u_xx = torch.zeros_like(u)
        u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True, allow_unused=True)[0]
        if u_y is None:
            u_y = torch.zeros_like(u)
        u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(u_y), create_graph=True, allow_unused=True)[0]
        if u_yy is None:
            u_yy = torch.zeros_like(u)

        r = torch.sqrt(x ** 2 + y ** 2)
        r = torch.where(r == 0.0, torch.ones_like(r) * 1e-6, r)
        source = 4.0 - 2.0 * self.a / r

        residual = u_xx + u_yy - source
        return torch.mean(residual ** 2)

    def boundary_loss(self, model) -> torch.Tensor:
        """Combine Dirichlet and Neumann boundary penalties."""
        loss = 0.0

        # Inner circle Dirichlet: T = 0
        if self.boundary_inner is not None and self.boundary_inner.size(0) > 0:
            u_in = model(self.boundary_inner)
            loss = loss + torch.mean((u_in - 0.0) ** 2)

        # Outer vertical edges Dirichlet: analytic Dirichlet on outer edge
        if self.boundary_dirichlet is not None and self.boundary_dirichlet.size(0) > 0:
            pts = self.boundary_dirichlet
            u_dir = model(pts)
            x = pts[:, 0:1]
            y = pts[:, 1:2]
            r = torch.sqrt(x ** 2 + y ** 2)
            analytic_dir = (r - self.a) ** 2
            loss = loss + torch.mean((u_dir - analytic_dir) ** 2)

        # Outer horizontal edges Neumann: analytic Neumann dT/dy
        if self.boundary_neumann is not None and self.boundary_neumann.size(0) > 0:
            pts = self.boundary_neumann.clone().requires_grad_(True)
            x = pts[:, 0:1]
            y = pts[:, 1:2]
            u = model(pts)
            u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True, allow_unused=True)[0]
            if u_y is None:
                u_y = torch.zeros_like(u)
            r = torch.sqrt(x ** 2 + y ** 2)
            r = torch.where(r == 0.0, torch.ones_like(r) * 1e-6, r)
            analytic_neu = 2.0 * (r - self.a) * (y / r)
            loss = loss + torch.mean((u_y - analytic_neu) ** 2)

        return loss

    def exact_solution(self, x: torch.Tensor) -> torch.Tensor:
        """Exact analytical solution: (r - a)^2 for r >= a"""
        x_coord = x[:, 0:1]
        y_coord = x[:, 1:2]
        r = torch.sqrt(x_coord ** 2 + y_coord ** 2)
        return (r - self.a) ** 2

    def sample_test_grid(self, n: int = 50):
        """Return a grid of points in the domain excluding the hole."""
        xs = torch.linspace(-self.b, self.b, n)
        ys = torch.linspace(-self.b, self.b, n)
        X, Y = torch.meshgrid(xs, ys, indexing='xy')
        pts = torch.stack([X.flatten(), Y.flatten()], dim=1)
        r = torch.sqrt(pts[:, 0:1] ** 2 + pts[:, 1:2] ** 2)
        mask = (r >= self.a).squeeze()
        return pts[mask]