"""Mathematical utility functions."""

import numpy as np


def lognorm_params_to_mean_std(mu, sigma):
    """Convert desired lognormal mean/std to underlying normal distribution parameters.

    Given the desired mean and standard deviation of a lognormal distribution,
    computes the mean and standard deviation of the underlying normal distribution.

    Parameters
    ----------
    mu : float or np.ndarray
        Desired mean of the lognormal distribution.
    sigma : float or np.ndarray
        Desired standard deviation of the lognormal distribution.

    Returns
    -------
    tuple
        (mu_normal, sigma_normal) parameters for np.random.lognormal.

    Notes
    -----
    This is useful when you want to specify the lognormal distribution in terms
    of its actual mean and variance rather than the underlying normal parameters.
    """
    return np.log(mu) - 0.5 * np.log(1 + (sigma / mu) ** 2), np.sqrt(
        np.log(1 + (sigma / mu) ** 2)
    )


def intuitive_rand_lognormal(mu, sigma, size):
    """Generate lognormal random samples with specified mean and standard deviation.

    Wrapper around np.random.lognormal that accepts the desired mean and
    standard deviation of the output distribution directly.

    Parameters
    ----------
    mu : float or np.ndarray
        Desired mean of the lognormal distribution.
    sigma : float or np.ndarray
        Desired standard deviation of the lognormal distribution.
    size : int or tuple
        Output shape.

    Returns
    -------
    np.ndarray
        Lognormal random samples with the specified mean and std.

    See Also
    --------
    lognorm_params_to_mean_std : Computes underlying normal parameters.
    """
    return np.random.lognormal(*lognorm_params_to_mean_std(mu, sigma), size)
