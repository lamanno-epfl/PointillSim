"""Effects module for simulating technical variation and noise in spatial transcriptomics.

This module provides classes for modeling:
- Batch effects: Systematic technical variation between FOVs/batches
- Technical noise: Optical and amplification-related variation
- Background noise: False positive transcript detections
- Dropout: Gene-specific detection failures
- Admixture: Transcript misassignment from segmentation errors and 3D tissue complexity
"""

from .batch import BatchEffectModel
from .noise import BackgroundNoise, DropoutModel, TechnicalNoise, SpatialNoise
from .admixture import (
    AdmixtureModel,
    Lateral2DAdmixture,
    ZAxisAdmixture,
    CompositeAdmixture,
    AdmixtureMetrics,
)

__all__ = [
    "BatchEffectModel",
    "TechnicalNoise",
    "BackgroundNoise",
    "DropoutModel",
    "SpatialNoise",
    "AdmixtureModel",
    "Lateral2DAdmixture",
    "ZAxisAdmixture",
    "CompositeAdmixture",
    "AdmixtureMetrics",
]
