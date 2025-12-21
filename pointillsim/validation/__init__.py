"""Validation module for assessing simulation quality and difficulty.

This module provides classes for:
- DifficultyScorer: Assess classification difficulty of simulated data
- ValidationMetrics: Compare simulated vs real data distributions
"""

from .difficulty import DifficultyScorer
from .metrics import ValidationMetrics

__all__ = [
    "DifficultyScorer",
    "ValidationMetrics",
]
