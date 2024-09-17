# Copyright (C) 2024 Andy Aschwanden, Constantine Khroulev
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

# pylint: skip-file
"""
Module for data processing
"""

from typing import Hashable, Iterable, Optional, Union

import numpy as np
import xarray as xr


@xr.register_dataset_accessor("interp")
class InterpolationMethods:
    """
    Interpolationes methods for xarray DataArray.

    This class is used to add custom methods to xarray DataArray objects. The methods can be accessed via the 'interpolation' attribute.

    Parameters
    ----------

    xarray_obj : xr.DataArray
      The xarray DataArray to which to add the custom methods.
    """

    def __init__(self, xarray_obj: xr.DataArray):
        """
        Initialize the InterpolationesMethods class.

        Parameters
        ----------

        xarray_obj : xr.DataArray
            The xarray DataArray to which to add the custom methods.
        """
        self._obj = xarray_obj

    def init(self):
        """
        Do-nothing method.

        This method is needed to work with joblib Parallel.
        """

    def __repr__(self):
        """
        Interpolation methods.
        """
        return """
Interpolationes methods for xarray DataArray.

This class is used to add custom methods to xarray DataArray objects. The methods can be accessed via the 'interpolation' attribute.

Parameters
----------

xarray_obj : xr.DataArray
  The xarray DataArray to which to add the custom methods.
      """

    def fillna(
        self,
        dim: Optional[Union[str, Iterable[Hashable]]] = ["y", "x"],
        method: str = "laplace",
    ):
        """
        Fill missing values using Laplacian.
        """
        data = self._obj.to_numpy()
        mask = np.array()
        self._obj = xr.apply_ufunc(
            self._fillna,
            data.copy(),
            mask.copy(),
            input_core_dims=[dim, dim],
            output_core_dims=[dim],
            kwargs={"method": method},
            vectorize=True,
        )
        return self._obj

    def _fillna(self, data, mask, method: str = "laplace"):
        """
        Fill missing values.
        """

        result = laplace(data, mask, -1, 1e-4)

        return result


def rho_jacobi(dimensions):
    """
    Calculate the Jacobi relaxation factor for a given grid size.

    Parameters
    ----------
    dimensions : tuple of int
        A tuple containing the dimensions of the grid (J, L).

    Returns
    -------
    float
        The Jacobi relaxation factor.
    """
    J, L = dimensions
    return (np.cos(np.pi / J) + np.cos(np.pi / L)) / 2


def fix_indices(Is, Js, dimensions):
    """
    Adjust indices to wrap around the grid, allowing the use of a 4-point stencil
    for all points, even those on the edge of the grid.

    Parameters
    ----------
    Is : numpy.ndarray
        Array of row indices.
    Js : numpy.ndarray
        Array of column indices.
    dimensions : tuple of int
        A tuple containing the grid dimensions (M, N).

    Returns
    -------
    tuple of numpy.ndarray
        Adjusted row and column indices.
    """
    M, N = dimensions
    Is[Is == M] = 0
    Is[Is == -1] = M - 1
    Js[Js == N] = 0
    Js[Js == -1] = N - 1
    return Is, Js


def laplace(data, mask, eps1, eps2, initial_guess="mean", max_iter=10000):
    """
    Solve the Laplace equation using the SOR method with Chebyshev acceleration.

    This function solves the Laplace equation using the Successive Over-Relaxation (SOR)
    method with Chebyshev acceleration as described in 'Numerical Recipes in Fortran:
    the art of scientific computing' by William H. Press et al -- 2nd edition, section 19.5.

    Parameters
    ----------
    data : numpy.ndarray
        A 2D array representing the computation grid.
    mask : numpy.ndarray
        A boolean array where True indicates points to be modified. For example, setting
        mask to 'data == 0' results in only modifying points where 'data' is zero.
    eps1 : float
        The first stopping criterion. Iterations stop if the norm of the residual becomes
        less than eps1 * initial_norm, where 'initial_norm' is the initial norm of the residual.
        Setting eps1 to zero or a negative number disables this stopping criterion.
    eps2 : float
        The second stopping criterion. Iterations stop if the absolute value of the maximal
        change in value between successive iterations is less than eps2. Setting eps2 to zero
        or a negative number disables this stopping criterion.
    initial_guess : float or str, optional
        The initial guess used for all the values in the domain. The default is 'mean', which
        uses the mean of all the present values as the initial guess for missing values.
        initial_guess must be 'mean' or a number.
    max_iter : int, optional
        The maximum number of iterations allowed. The default is 10000.

    Returns
    -------
    None
    """
    dimensions = data.shape
    rjac = rho_jacobi(dimensions)
    i, j = np.indices(dimensions)
    # This splits the grid into 'odd' and 'even' parts, according to the checkerboard pattern:
    odd = (i % 2 == 1) ^ (j % 2 == 0)
    even = (i % 2 == 0) ^ (j % 2 == 0)
    # odd and even parts _in_ the domain:
    odd_part = list(zip(i[mask & odd], j[mask & odd]))
    even_part = list(zip(i[mask & even], j[mask & even]))
    # relative indices of the stencil points:
    k = np.array([0, 1, 0, -1])
    l = np.array([-1, 0, 1, 0])
    parts = [odd_part, even_part]

    try:
        initial_guess = float(initial_guess)
    except:
        if initial_guess == "mean":
            present = mask == False
            initial_guess = np.mean(data[present])
        else:
            print(
                f"ERROR: initial_guess of '{initial_guess}' is not supported (it should be a number or 'mean').\nNote: your data was not modified."
            )
            return

    data[mask] = initial_guess
    print(f"Using the initial guess of {initial_guess:.10f}.")

    # compute the initial norm of residual
    initial_norm = 0.0
    for m in [0, 1]:
        for i, j in parts[m]:
            Is, Js = fix_indices(i + k, j + l, dimensions)
            xi = np.sum(data[Is, Js]) - 4 * data[i, j]
            initial_norm += abs(xi)
    print(f"Initial norm of residual = {initial_norm}")
    print(f"Criterion is (change < {eps2}) OR (res norm < {eps1} (initial norm)).")

    omega = 1.0
    # The main loop:
    for n in np.arange(max_iter):
        anorm = 0.0
        change = 0.0
        for m in [0, 1]:
            for i, j in parts[m]:
                # stencil points:
                Is, Js = fix_indices(i + k, j + l, dimensions)
                residual = sum(data[Is, Js]) - 4 * data[i, j]
                delta = omega * 0.25 * residual
                data[i, j] += delta

                # record the maximal change and the residual norm:
                anorm += abs(residual)
                change = max(change, abs(delta))
                # Chebyshev acceleration (see formula 19.5.30):
                if n == 1 and m == 1:
                    omega = 1.0 / (1.0 - 0.5 * rjac**2)
                else:
                    omega = 1.0 / (1.0 - 0.25 * rjac**2 * omega)
        print(f"max change = {change:.10f}, residual norm = {anorm:.10f}")
        if (anorm < eps1 * initial_norm) or (change < eps2):
            print(
                f"Exiting with change={change}, anorm={anorm} after {n + 1} iteration(s)."
            )
            return
    print("Exceeded the maximum number of iterations.")
    return
