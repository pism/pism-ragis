# Copyright (C) 2023-25 Andy Aschwanden
#
# This file is part of pism-ragis.
#
# PISM-RAGIS is free software; you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation; either version 3 of the License, or (at your option) any later
# version.
#
# PISM-RAGIS is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License
# along with PISM; if not, write to the Free Software
# Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA

# pylint: disable=too-many-positional-arguments,unused-argument
"""
Log likelihood models.
"""

import numpy as np
import xarray as xr
from scipy.special import pseudo_huber
from sklearn.metrics import jaccard_score


def log_jaccard_score(
    x: np.ndarray | xr.DataArray,
    mu: np.ndarray | xr.DataArray,
    fudge_factor: float = 1.0,
) -> np.ndarray | xr.DataArray:
    """
    Calculate the negative log-likelihood using the Jaccard score.

    Parameters
    ----------
    x : np.ndarray or xr.DataArray
        The data for which the log-likelihood is to be calculated.
    mu : np.ndarray or xr.DataArray
        The mean of the distribution.
    fudge_factor : float, optional
        A multiplicative factor applied to the Jaccard score, by default 1.0.

    Returns
    -------
    np.ndarray or xr.DataArray
        The log-likelihood of the data given the distribution parameters.
    """

    return -fudge_factor * jaccard_score(mu, x, average="binary")


def log_jaccard_score_xr(
    x: np.ndarray | xr.DataArray,
    mu: np.ndarray | xr.DataArray,
    std: float | np.ndarray | xr.DataArray,
    fudge_factor: float = 1.0,
    dim="z",
    sum_dims=["y", "x", "time"],
) -> xr.DataArray:
    """
    Calculate the log-likelihood using the Jaccard score for xarray DataArrays.

    Parameters
    ----------
    x : np.ndarray or xr.DataArray
        The data for which the log-likelihood is to be calculated.
    mu : np.ndarray or xr.DataArray
        The mean of the distribution.
    std : float or np.ndarray or xr.DataArray
        The standard deviation of the distribution.
    fudge_factor : float, optional
        A multiplicative factor applied to the Jaccard score, by default 1.0.
    dim : str, optional
        The dimension along which to calculate the Jaccard score, by default "z".
    sum_dims : list of str, optional
        The dimensions to sum over when computing the Jaccard score, by default ["y", "x", "time"].

    Returns
    -------
    xr.DataArray
        The log-likelihood of the data given the distribution parameters.
    """
    da = xr.apply_ufunc(
        log_jaccard_score,
        mu.stack({dim: sum_dims}).chunk({dim: -1}),
        x.stack({dim: sum_dims}).chunk({dim: -1}),
        input_core_dims=[[dim], [dim]],
        output_core_dims=[[]],
        kwargs={"fudge_factor": fudge_factor},
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
    )
    da.name = "log_likelihood"
    da.attrs.update({"units": "1", "long_name": "negative log likelihood"})
    return da


def log_normal(
    x: np.ndarray | xr.DataArray,
    mu: np.ndarray | xr.DataArray,
    std: float | np.ndarray | xr.DataArray,
    fudge_factor: float = 1.0,
) -> np.ndarray | xr.DataArray:
    """
    Calculate the negative log-likelihood of data given a Normal distribution.

    Parameters
    ----------
    x : np.ndarray or xr.DataArray
        The data for which the log-likelihood is to be calculated.
    mu : np.ndarray or xr.DataArray
        The mean of the distribution.
    std : float or np.ndarray or xr.DataArray
        The standard deviation of the distribution.
    fudge_factor : float, optional
        A multiplicative factor applied to the standard deviation, by default 1.0.

    Returns
    -------
    np.ndarray or xr.DataArray
        The log-likelihood of the data given the distribution parameters.
    """
    return -0.5 * ((x - mu) / (fudge_factor * std)) ** 2 - 0.5 * np.log(
        2 * np.pi * (fudge_factor * std) ** 2
    )


def log_normal_xr(
    x: xr.DataArray,
    mu: xr.DataArray,
    std: float | xr.DataArray,
    fudge_factor: float = 3.0,
    sum_dims=["y", "x", "time"],
) -> xr.DataArray:
    """
    Calculate the log-likelihood of data given a Normal distribution, summed along sum_dims.

    Parameters
    ----------
    x : xr.DataArray
        The data for which the log-likelihood is to be calculated.
    mu : xr.DataArray
        The mean of the distribution.
    std : float or xr.DataArray
        The standard deviation of the distribution.
    fudge_factor : float, optional
        A multiplicative factor applied to the standard deviation, by default 3.0.
    sum_dims : list of str, optional
        The dimensions to sum over when computing the log-likelihood, by default ["y", "x", "time"].

    Returns
    -------
    xr.DataArray
        The log-likelihood of the data given the distribution parameters.
    """

    da = log_normal(x, mu, std, fudge_factor=fudge_factor)
    da = da.where(da != 0, np.nan)
    da.name = "log_likelihood"
    da.attrs.update({"units": "1", "long_name": "negative log likelihood"})
    return da.sum(dim=sum_dims)


def log_pseudo_huber(
    x: np.ndarray | xr.DataArray,
    mu: np.ndarray | xr.DataArray,
    std: float | np.ndarray | xr.DataArray,
    fudge_factor: float = 1.0,
    delta: float = 2.0,
) -> np.ndarray | xr.DataArray:
    """
    Calculate the negative log-likelihood of data given a pseudo-Huber distribution.

    Parameters
    ----------
    x : np.ndarray or xr.DataArray
        The data for which the log-likelihood is to be calculated.
    mu : np.ndarray or xr.DataArray
        The mean of the distribution.
    std : float or np.ndarray or xr.DataArray
        The standard deviation of the distribution.
    fudge_factor : float, optional
        A multiplicative factor applied to the standard deviation, by default 1.0.
    delta : float, optional
        The delta parameter for the pseudo-Huber loss function, by default 2.0.

    Returns
    -------
    np.ndarray or xr.DataArray
        The log-likelihood of the data given the distribution parameters.
    """

    return -pseudo_huber(delta, (x - mu) / (fudge_factor * std)) - 1


def log_pseudo_huber_xr(
    x: xr.DataArray | None = None,
    mu: xr.DataArray | None = None,
    std: float | xr.DataArray | None = None,
    fudge_factor: float = 3.0,
    sum_dims=["y", "x", "time"],
    delta: float = 2.0,
) -> xr.DataArray:
    """
    Calculate the log-likelihood of data given a pseudo-Huber distribution, summed along sum_dims.

    Parameters
    ----------
    x : xr.DataArray
        The data for which the log-likelihood is to be calculated.
    mu : xr.DataArray
        The mean of the distribution.
    std : float or xr.DataArray
        The standard deviation of the distribution.
    fudge_factor : float, optional
        A multiplicative factor applied to the standard deviation, by default 3.0.
    sum_dims : list of str, optional
        The dimensions to sum over when computing the log-likelihood, by default ["y", "x", "time"].
    delta : float, optional
        The delta parameter for the pseudo-Huber loss function, by default 2.0.

    Returns
    -------
    xr.DataArray
        The log-likelihood of the data given the distribution parameters.
    """

    da = log_pseudo_huber(x, mu, std, fudge_factor=fudge_factor, delta=delta)
    da = da.where(da != 0, np.nan)
    da.name = "log_likelihood"
    da.attrs.update({"units": "1", "long_name": "negative log likelihood"})
    return da.sum(dim=sum_dims)
