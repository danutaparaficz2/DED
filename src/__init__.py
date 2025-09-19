"""
Physics Informed Neural Network (PINN) Implementation
======================================================

This module provides a complete implementation of Physics Informed Neural Networks
for solving partial differential equations.
"""

__version__ = "1.0.0"
__author__ = "DED Project"

from .pinn import PINN
from .pde_problems import *
from .utils import *

__all__ = ['PINN']