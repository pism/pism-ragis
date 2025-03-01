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
    std: float | np.ndarray | xr.DataArray,
) -> np.ndarray | xr.DataArray:
    """
    Calculate the log-likelihood using the Jaccard score.

    Parameters
    ----------
    x : np.ndarray or xr.DataArray
        The data for which the log-likelihood is to be calculated.
    mu : np.ndarray or xr.DataArray
        The mean of the distribution.
    std : float or np.ndarray or xr.DataArray
        The standard deviation of the distribution.

    Returns
    -------
    np.ndarray or xr.DataArray
        The log-likelihood of the data given the distribution parameters.
    """

    return -std * jaccard_score(x, mu, average="binary")


def log_jaccard_score_xr(
    y_true: xr.DataArray, y_pred: xr.DataArray, dim="z", sum_dims=["y", "x", "time"]
) -> xr.DataArray:
    """
    Calculate the log-likelihood using the Jaccard score for xarray DataArrays.

    Parameters
    ----------
    y_true : xr.DataArray
        The true data values.
    y_pred : xr.DataArray
        The predicted data values.
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
        jaccard_score,
        y_true.stack({dim: sum_dims}),
        y_pred.stack({dim: sum_dims}).chunk({dim: -1}),
        input_core_dims=[[dim], [dim]],
        output_core_dims=[[]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
    )
    da.name = "log_jaccard_score"


def log_normal(
    x: np.ndarray | xr.DataArray,
    mu: np.ndarray | xr.DataArray,
    std: float | np.ndarray | xr.DataArray,
) -> np.ndarray | xr.DataArray:
    """
    Calculate the log-likelihood of data given a Normal distribution.

    Parameters
    ----------
    x : np.ndarray or xr.DataArray
        The data for which the log-likelihood is to be calculated.
    mu : np.ndarray or xr.DataArray
        The mean of the distribution.
    std : float or np.ndarray or xr.DataArray
        The standard deviation of the distribution.

    Returns
    -------
    np.ndarray or xr.DataArray
        The log-likelihood of the data given the distribution parameters.
    """

    return -0.5 * ((x - mu) / std) ** 2 - 0.5 * np.log(2 * np.pi * std**2)


def log_pseudo_huber(
    x: np.ndarray | xr.DataArray,
    mu: np.ndarray | xr.DataArray,
    std: float | np.ndarray | xr.DataArray,
    delta: float | np.ndarray | xr.DataArray = 2.0,
) -> np.ndarray | xr.DataArray:
    """
    Calculate the log-likelihood of data given a pseudo-Huber distribution.

    Parameters
    ----------
    x : np.ndarray or xr.DataArray
        The data for which the log-likelihood is to be calculated.
    mu : np.ndarray or xr.DataArray
        The mean of the distribution.
    std : float or np.ndarray or xr.DataArray
        The standard deviation of the distribution.
    delta : float or np.ndarray or xr.DataArray, optional
        The delta parameter for the pseudo-Huber loss function, by default 2.0.

    Returns
    -------
    np.ndarray or xr.DataArray
        The log-likelihood of the data given the distribution parameters.
    """

    return -pseudo_huber(delta, (x - mu) / std) - 1
