"""
Basic tests for PINN implementation.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import numpy as np
import unittest

from src.pinn import PINN
from src.pde_problems import HeatEquation, WaveEquation, BurgersEquation


class TestPINN(unittest.TestCase):
    """Test cases for PINN implementation."""
    
    def setUp(self):
        """Set up test fixtures."""
        torch.manual_seed(42)
        np.random.seed(42)
        
        self.model = PINN(
            input_dim=2,
            hidden_layers=[10, 10],
            output_dim=1,
            activation='tanh'
        )
    
    def test_model_creation(self):
        """Test PINN model creation."""
        self.assertEqual(self.model.input_dim, 2)
        self.assertEqual(self.model.output_dim, 1)
        self.assertIsInstance(self.model.network, torch.nn.Sequential)
    
    def test_forward_pass(self):
        """Test forward pass through the network."""
        x = torch.randn(10, 2)
        output = self.model(x)
        self.assertEqual(output.shape, (10, 1))
    
    def test_derivatives_computation(self):
        """Test automatic differentiation for derivatives."""
        x = torch.randn(5, 2, requires_grad=True)
        derivatives = self.model.compute_derivatives(x, order=2)
        
        self.assertIn('u', derivatives)
        self.assertIn('u_0', derivatives)  # du/dx
        self.assertIn('u_1', derivatives)  # du/dt
        self.assertIn('u_00', derivatives) # d²u/dx²
        self.assertIn('u_11', derivatives) # d²u/dt²
    
    def test_prediction(self):
        """Test prediction method."""
        x = torch.randn(10, 2)
        prediction = self.model.predict(x)
        self.assertEqual(prediction.shape, (10, 1))


class TestPDEProblems(unittest.TestCase):
    """Test cases for PDE problem implementations."""
    
    def setUp(self):
        """Set up test fixtures."""
        torch.manual_seed(42)
        np.random.seed(42)
        
        self.model = PINN(input_dim=2, hidden_layers=[10, 10], output_dim=1)
    
    def test_heat_equation(self):
        """Test Heat equation problem setup."""
        problem = HeatEquation(n_collocation=100, n_boundary=20, n_initial=20)
        
        self.assertEqual(problem.name, "Heat Equation")
        self.assertIn('x', problem.domain)
        self.assertIn('t', problem.domain)
        
        # Test loss computations
        pde_loss = problem.pde_loss(self.model)
        bc_loss = problem.boundary_loss(self.model)
        ic_loss = problem.initial_loss(self.model)
        
        self.assertIsInstance(pde_loss, torch.Tensor)
        self.assertIsInstance(bc_loss, torch.Tensor)
        self.assertIsInstance(ic_loss, torch.Tensor)
        
        self.assertEqual(pde_loss.dim(), 0)  # scalar loss
        self.assertEqual(bc_loss.dim(), 0)   # scalar loss
        self.assertEqual(ic_loss.dim(), 0)   # scalar loss
    
    def test_wave_equation(self):
        """Test Wave equation problem setup."""
        problem = WaveEquation(n_collocation=100, n_boundary=20, n_initial=20)
        
        self.assertEqual(problem.name, "Wave Equation")
        
        # Test loss computations
        pde_loss = problem.pde_loss(self.model)
        bc_loss = problem.boundary_loss(self.model)
        ic_loss = problem.initial_loss(self.model)
        
        self.assertIsInstance(pde_loss, torch.Tensor)
        self.assertIsInstance(bc_loss, torch.Tensor)
        self.assertIsInstance(ic_loss, torch.Tensor)
    
    def test_burgers_equation(self):
        """Test Burgers equation problem setup."""
        problem = BurgersEquation(n_collocation=100, n_boundary=20, n_initial=20)
        
        self.assertEqual(problem.name, "Burgers Equation")
        
        # Test loss computations
        pde_loss = problem.pde_loss(self.model)
        bc_loss = problem.boundary_loss(self.model)
        ic_loss = problem.initial_loss(self.model)
        
        self.assertIsInstance(pde_loss, torch.Tensor)
        self.assertIsInstance(bc_loss, torch.Tensor)
        self.assertIsInstance(ic_loss, torch.Tensor)


class TestTraining(unittest.TestCase):
    """Test training functionality."""
    
    def test_short_training(self):
        """Test short training run."""
        torch.manual_seed(42)
        np.random.seed(42)
        
        model = PINN(input_dim=2, hidden_layers=[10, 10], output_dim=1)
        problem = HeatEquation(n_collocation=50, n_boundary=10, n_initial=10)
        
        # Short training run
        losses = model.train_model(
            pde_loss_fn=problem.pde_loss,
            boundary_loss_fn=problem.boundary_loss,
            initial_loss_fn=problem.initial_loss,
            num_epochs=10,
            lr=1e-3,
            print_every=5
        )
        
        self.assertEqual(len(losses), 10)
        self.assertIsInstance(losses[0], float)


def run_tests():
    """Run all tests."""
    print("Running PINN implementation tests...")
    print("="*50)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestPINN))
    suite.addTests(loader.loadTestsFromTestCase(TestPDEProblems))
    suite.addTests(loader.loadTestsFromTestCase(TestTraining))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    if result.wasSuccessful():
        print("\\nAll tests passed! ✅")
        return True
    else:
        print("\\nSome tests failed! ❌")
        return False


if __name__ == "__main__":
    success = run_tests()