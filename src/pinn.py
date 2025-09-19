"""
Core Physics Informed Neural Network implementation.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Callable, List, Tuple, Optional, Dict
import matplotlib.pyplot as plt
from tqdm import tqdm


class PINN(nn.Module):
    """
    Physics Informed Neural Network for solving PDEs.
    
    This class implements a neural network that can learn solutions to partial
    differential equations by incorporating both data points and physics constraints
    through automatic differentiation.
    """
    
    def __init__(self, 
                 input_dim: int, 
                 hidden_layers: List[int] = [50, 50, 50], 
                 output_dim: int = 1,
                 activation: str = 'tanh'):
        """
        Initialize the PINN model.
        
        Args:
            input_dim: Number of input dimensions (spatial + temporal)
            hidden_layers: List of hidden layer sizes
            output_dim: Number of output dimensions
            activation: Activation function ('tanh', 'relu', 'sigmoid')
        """
        super(PINN, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Create the neural network layers
        layers = []
        prev_size = input_dim
        
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(self._get_activation(activation))
            prev_size = hidden_size
            
        layers.append(nn.Linear(prev_size, output_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights using Xavier initialization
        self._initialize_weights()
        
    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function by name."""
        activations = {
            'tanh': nn.Tanh(),
            'relu': nn.ReLU(),
            'sigmoid': nn.Sigmoid(),
            'leaky_relu': nn.LeakyReLU()
        }
        return activations.get(activation, nn.Tanh())
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        return self.network(x)
    
    def compute_derivatives(self, x: torch.Tensor, order: int = 1) -> Dict[str, torch.Tensor]:
        """
        Compute derivatives of the network output with respect to input.
        
        Args:
            x: Input tensor with requires_grad=True
            order: Maximum order of derivatives to compute
            
        Returns:
            Dictionary containing computed derivatives
        """
        x.requires_grad_(True)
        u = self.forward(x)
        
        derivatives = {}
        
        # First-order derivatives
        if order >= 1:
            for i in range(self.input_dim):
                grad = torch.autograd.grad(
                    outputs=u, 
                    inputs=x,
                    grad_outputs=torch.ones_like(u),
                    create_graph=True,
                    retain_graph=True
                )[0]
                derivatives[f'u_{i}'] = grad[:, i:i+1]
        
        # Second-order derivatives
        if order >= 2:
            for i in range(self.input_dim):
                if f'u_{i}' in derivatives:
                    grad2 = torch.autograd.grad(
                        outputs=derivatives[f'u_{i}'],
                        inputs=x,
                        grad_outputs=torch.ones_like(derivatives[f'u_{i}']),
                        create_graph=True,
                        retain_graph=True
                    )[0]
                    derivatives[f'u_{i}{i}'] = grad2[:, i:i+1]
                    
                    # Cross derivatives
                    for j in range(i+1, self.input_dim):
                        derivatives[f'u_{i}{j}'] = grad2[:, j:j+1]
        
        derivatives['u'] = u
        return derivatives
    
    def train_model(self,
                   pde_loss_fn: Callable,
                   boundary_loss_fn: Callable,
                   initial_loss_fn: Optional[Callable] = None,
                   data_loss_fn: Optional[Callable] = None,
                   num_epochs: int = 1000,
                   lr: float = 1e-3,
                   lambda_pde: float = 1.0,
                   lambda_bc: float = 1.0,
                   lambda_ic: float = 1.0,
                   lambda_data: float = 1.0,
                   print_every: int = 100) -> List[float]:
        """
        Train the PINN model.
        
        Args:
            pde_loss_fn: Function to compute PDE residual loss
            boundary_loss_fn: Function to compute boundary condition loss
            initial_loss_fn: Function to compute initial condition loss (optional)
            data_loss_fn: Function to compute data fitting loss (optional)
            num_epochs: Number of training epochs
            lr: Learning rate
            lambda_pde: Weight for PDE loss
            lambda_bc: Weight for boundary condition loss
            lambda_ic: Weight for initial condition loss
            lambda_data: Weight for data fitting loss
            print_every: Print loss every N epochs
            
        Returns:
            List of total losses during training
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=100, factor=0.5)
        
        losses = []
        
        pbar = tqdm(range(num_epochs), desc="Training PINN")
        
        for epoch in pbar:
            optimizer.zero_grad()
            
            # Compute individual losses
            pde_loss = pde_loss_fn(self)
            bc_loss = boundary_loss_fn(self)
            
            total_loss = lambda_pde * pde_loss + lambda_bc * bc_loss
            
            loss_components = {
                'PDE': pde_loss.item(),
                'BC': bc_loss.item()
            }
            
            if initial_loss_fn is not None:
                ic_loss = initial_loss_fn(self)
                total_loss += lambda_ic * ic_loss
                loss_components['IC'] = ic_loss.item()
            
            if data_loss_fn is not None:
                data_loss = data_loss_fn(self)
                total_loss += lambda_data * data_loss
                loss_components['Data'] = data_loss.item()
            
            # Backward pass
            total_loss.backward()
            optimizer.step()
            scheduler.step(total_loss)
            
            losses.append(total_loss.item())
            
            # Update progress bar
            if epoch % print_every == 0:
                loss_str = " | ".join([f"{k}: {v:.2e}" for k, v in loss_components.items()])
                pbar.set_description(f"Epoch {epoch} | Total: {total_loss.item():.2e} | {loss_str}")
        
        return losses
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Make predictions with the trained model.
        
        Args:
            x: Input tensor
            
        Returns:
            Predicted output tensor
        """
        with torch.no_grad():
            return self.forward(x)
    
    def save_model(self, filepath: str):
        """Save the trained model."""
        torch.save({
            'model_state_dict': self.state_dict(),
            'input_dim': self.input_dim,
            'output_dim': self.output_dim
        }, filepath)
    
    @classmethod
    def load_model(cls, filepath: str, hidden_layers: List[int] = [50, 50, 50], activation: str = 'tanh'):
        """Load a saved model."""
        checkpoint = torch.load(filepath)
        model = cls(
            input_dim=checkpoint['input_dim'],
            hidden_layers=hidden_layers,
            output_dim=checkpoint['output_dim'],
            activation=activation
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        return model