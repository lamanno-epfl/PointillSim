"""Effects module for simulating technical variation and noise in spatial transcriptomics.

This module provides classes for modeling:
- Batch effects: Systematic technical variation between FOVs/batches
- Technical noise: Optical and amplification-related variation
- Background noise: False positive transcript detections
- Dropout: Gene-specific detection failures
"""

from .batch import BatchEffectModel
from .noise import BackgroundNoise, DropoutModel, TechnicalNoise

__all__ = [
    "BatchEffectModel",
    "TechnicalNoise",
    "BackgroundNoise",
    "DropoutModel",
]
