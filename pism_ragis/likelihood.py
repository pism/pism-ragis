# Copyright (C) 2023 Andy Aschwanden
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

from typing import Union

import numpy as np
import xarray as xr
from scipy.special import pseudo_huber


def log_normal(
    x: Union[np.ndarray, xr.DataArray],
    mu: Union[np.ndarray, xr.DataArray],
    std: Union[np.ndarray, xr.DataArray],
    n: int,
) -> Union[np.ndarray, xr.DataArray]:
    """
    Calculate the log-likelihood of data given a distribution.

    This function computes the log-likelihood of the data `x` given the mean and standard deviation
    for the Normal distribution.

    Parameters
    ----------
    x : Union[np.ndarray, xr.DataArray]
        The data for which the log-likelihood is to be calculated. Can be an array of values or an xarray.DataArray.
    mu : Union[float, np.ndarray, xr.DataArray]
        The mean of the distribution. Can be a single value, an array of values or an xarray.DataArray.
    std : Union[float, np.ndarray, xr.DataArray]
        The standard deviation of the distribution. Can be a single value, an array of values, or an xarray.DataArray.
    n : int
        The number of data points.
    kind : str, optional
        The type of distribution. Currently, only "gaussian" is implemented. Default is "normal".

    Returns
    -------
    Union[np.ndarray, xr.DataArray]
        The log-likelihood of the data given the distribution parameters.

    """

    return -0.5 * ((x - mu) / std) ** 2 - n * 0.5 * np.log(2 * np.pi * std**2)


def log_pseudo_huber(
    x: Union[np.ndarray, xr.DataArray],
    mu: Union[np.ndarray, xr.DataArray],
    std: Union[np.ndarray, xr.DataArray],
    delta: Union[np.ndarray, xr.DataArray] = 2.0,
    n: int = 0,
) -> Union[np.ndarray, xr.DataArray]:
    """
    Calculate the log-likelihood of data given a distribution.

    This function computes the log-likelihood of the data `x` given the mu and standard deviation
    for smooth Huber loss is implemented.

    Parameters
    ----------
    x : Union[np.ndarray, xr.DataArray]
        The data for which the log-likelihood is to be calculated. Can be an array of values or an xarray.DataArray.
    mu : Union[float, np.ndarray, xr.DataArray]
        The mean of the distribution. Can be a single value, an array of values or an xarray.DataArray.
    std : Union[float, np.ndarray, xr.DataArray]
        The standard deviation of the distribution. Can be a single value, an array of values, or an xarray.DataArray.
    n : int
        The number of data points.

    Returns
    -------
    Union[np.ndarray, xr.DataArray]
        The log-likelihood of the data given the distribution parameters.

    """

    return -pseudo_huber(delta, (x - mu) / std) - n
