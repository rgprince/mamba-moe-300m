"""
Mamba-MoE 300M - Core package initialization
"""

__version__ = "0.1.0"
__author__ = "Prince"

# Only import model subpackage (others will be added in Phase 2)
from . import model

__all__ = ["model"]
