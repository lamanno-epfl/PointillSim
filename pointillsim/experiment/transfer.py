"""Transfer functions for modeling detection efficiency."""

import numpy as np
from ..utils.math import lognorm_params_to_mean_std


class TransferFunctionBase:
    """Abstract base class for gene expression transfer functions.

    Transfer functions model systematic biases in gene detection,
    such as efficiency differences between genes or non-linear effects.
    """
    def __init__(self, **kwargs):
        self.params = kwargs

    def transform(self, x):
        """Transform expression values.

        Parameters
        ----------
        x : np.ndarray
            Input expression matrix.

        Returns
        -------
        np.ndarray
            Transformed expression values.
        """
        raise NotImplementedError("This is an abstract method")


class IdentityTransfer(TransferFunctionBase):
    """Identity transfer function (no transformation).

    Passes expression values through unchanged. Use when no
    systematic biases need to be modeled.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def transform(self, x):
        """Return input unchanged."""
        return x


class AffineNonNegTransfer(TransferFunctionBase):
    """Affine transfer function with non-negative output.

    Models gene-specific detection efficiency as a linear transformation
    y = scale * x + offset, clipped to be non-negative.

    Parameters
    ----------
    scales : float or np.ndarray, optional
        Scale factors per gene. If float, samples from lognormal.
    offsets : float or np.ndarray, optional
        Offset values per gene. If float, samples from normal.
    scales_std : float, optional
        Std for lognormal sampling of scales.
    offsets_std : float, optional
        Std for normal sampling of offsets.

    Notes
    -----
    Scales are drawn from a lognormal distribution to ensure positivity.
    The final output is clipped to a minimum of 1e-6 to avoid zeros.
    """
    def __init__(self, scales=None, offsets=None, scales_std=None, offsets_std=None):
        super().__init__(
            scales=scales,
            offsets=offsets,
            scales_std=scales_std,
            offsets_std=offsets_std,
        )
        self.scales = None
        self.offsets = None

    def transform(self, x):
        """Apply affine transformation with non-negative output."""
        n = x.shape[0]
        if self.scales is not None:
            pass
        elif self.params["scales"] is None:
            raise ValueError("scales must be provided")
        elif isinstance(self.params["scales"], (float, int)):
            self.scales = np.random.lognormal(
                *lognorm_params_to_mean_std(
                    self.params["scales"], self.params["scales_std"]
                ),
                n,
            )
        else:
            self.scales = self.params["scales"]

        if self.offsets is not None:
            pass
        elif self.params["offsets"] is None:
            raise ValueError("offsets must be provided")
        elif isinstance(self.params["offsets"], (float, int)):
            self.offsets = np.random.normal(
                self.params["offsets"], self.params["offsets_std"], n
            )
        else:
            self.offsets = self.params["offsets"]

        return np.maximum(self.scales[:, None] * x + self.offsets[:, None], 1e-6)
