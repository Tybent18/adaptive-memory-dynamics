"""Adaptive Memory Dynamics experimental framework."""

from .config import ExperimentConfig
from .model import AdaptiveMLP

__all__ = ["AdaptiveMLP", "ExperimentConfig"]
__version__ = "0.1.0"
