# Copyright (C) 24 Andy Aschwanden, Constantine Khroulev
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
Tests for interpolation module.
"""

import time
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from pism_ragis.interpolation import fill_missing_petsc, laplace


def create_dataarray(
    M: int = 301, N: int = 201, R: int = 2, p: float = 0.975, seed: int = 42
) -> Tuple[xr.DataArray, np.ndarray]:
    """
    Create a DataArray with missing values and the true underlying data.

    Parameters
    ----------
    M : int, optional
        Number of rows in the data array, by default 301.
    N : int, optional
        Number of columns in the data array, by default 201.
    R : int, optional
        Number of time steps or repetitions, by default 2.
    p : float, optional
        Probability of a value being present (not missing), by default 0.975.
    seed : int, optional
        Seed for the random number generator, by default 42.

    Returns
    -------
    Tuple[xr.DataArray, np.ndarray]
        A tuple containing the DataArray with missing values and the true underlying data array.
    """
    x = np.linspace(-1, 1, N)
    y = np.linspace(-1, 1, M)
    xx, yy = np.meshgrid(x, y)
    zz = np.sin(2.5 * np.pi * yy) * np.cos(2.0 * np.pi * xx)
    zz_true = zz.reshape(1, M, N).repeat(R, axis=0)

    rng = np.random.default_rng(seed=seed)
    mask = (rng.random(size=zz_true.size) < p).reshape(zz_true.shape)

    da = xr.DataArray(zz_true, dims=["time", "y", "x"], name="test")
    da = da.where(mask, np.nan).chunk("auto")
    return da, zz_true


def test_fill_missing_petsc():
    """
    Test the fill_missing_petsc function by comparing the filled data to the true data.

    This function creates a data array with missing values, fills the missing values using
    the fill_missing_petsc function, and asserts that the filled data is almost equal to
    the true data within a specified decimal precision.

    Raises
    ------
    AssertionError
        If the filled data does not match the true data within the specified decimal precision.
    """
    da, data_true = create_dataarray()

    data = da.isel(time=0)
    mask = da.isel(time=0).isnull()
    data_masked = np.ma.array(data=data, mask=mask)
    data_filled = fill_missing_petsc(data_masked, method="direct")
    np.testing.assert_array_almost_equal(data_true[0], data_filled, decimal=2)
    data_filled = fill_missing_petsc(data_masked, method="iterative")
    np.testing.assert_array_almost_equal(data_true[0], data_filled, decimal=2)


def test_fill_missing_laplace():
    """
    Test the laplace function by comparing the filled data to the true data.

    This function creates a data array with missing values, fills the missing values using
    the fill_missing_petsc function, and asserts that the filled data is almost equal to
    the true data within a specified decimal precision.

    Raises
    ------
    AssertionError
        If the filled data does not match the true data within the specified decimal precision.
    """
    da, data_true = create_dataarray()

    data = da.isel(time=0).to_numpy()
    mask = da.isel(time=0).isnull().to_numpy()
    data_filled = laplace(data, mask)
    # Fix test, decimal=0 is not good enough
    np.testing.assert_array_almost_equal(data_true[0], data_filled, decimal=0)


def test_fill_missing_xr():
    """
    Test the laplace function by comparing the filled data to the true data.

    This function creates a data array with missing values, fills the missing values using
    the fill_missing_petsc function, and asserts that the filled data is almost equal to
    the true data within a specified decimal precision.

    Raises
    ------
    AssertionError
        If the filled data does not match the true data within the specified decimal precision.
    """
    da, data_true = create_dataarray(4, 4)

    data_filled = da.utils.fillna()
    # Fix test, decimal=0 is not good enough
    np.testing.assert_array_almost_equal(data_true, data_filled, decimal=0)

    data_filled = da.utils.fillna()
    data_filled = data_filled.expand_dims("exp_id")
    data_true = data_true.reshape(1, *data_true.shape)
    np.testing.assert_array_almost_equal(data_true, data_filled, decimal=0)


if __name__ == "__main__":
    __spec__ = None  # type: ignore

    def profile_scipy(D):
        """
        Profile Scipy.
        """
        da, data_true = create_dataarray(D, D)

        data = da.isel(time=0).to_numpy()
        mask = da.isel(time=0).isnull().to_numpy()
        data_masked = np.ma.array(data=data, mask=mask)
        data_filled = laplace(data, mask)
        return data_filled, data_masked, data_true

    def profile_petsc(D, method: str = "iterative"):
        """
        Profile PETSc.
        """
        da, data_true = create_dataarray(D, D)

        data = da.isel(time=0)
        mask = da.isel(time=0).isnull()
        data_masked = np.ma.array(data=data, mask=mask)
        data_filled = fill_missing_petsc(data_masked, method=method)
        return data_filled, data_masked, data_true

    for D in [100, 200, 500, 1000, 2000, 5000]:
        start = time.time()
        data_filled, data_masked, data_true = profile_scipy(D)
        end = time.time()
        time_elapsed = end - start
        print(f"Scipy: filling a {D}x{D} matrix took {time_elapsed:.2f}s")
        fig, axs = plt.subplots(1, 3)
        axs[0].imshow(data_true[0])
        axs[1].imshow(data_true[0])
        axs[2].imshow(data_filled)
        fig.savefig(f"laplace_scipy_true_vs_inter_{D}x{D}.png", dpi=300)
        start = time.time()
        data_filled, data_masked, data_true = profile_petsc(D, method="iterative")
        end = time.time()
        time_elapsed = end - start
        print(f"PETSc iterative solver: filling a {D}x{D} matrix took {time_elapsed:.2f}s")
        start = time.time()
        data_filled, data_masked, data_true = profile_petsc(D, method="direct")
        end = time.time()
        time_elapsed = end - start
        print(f"PETSc direct solver: filling a {D}x{D} matrix took {time_elapsed:.2f}s")
        fig, axs = plt.subplots(1, 2)
        axs[0].imshow(data_true[0])
        axs[1].imshow(data_filled)
        fig.savefig(f"laplace_petsc_true_vs_inter_{D}x{D}.png", dpi=300)
