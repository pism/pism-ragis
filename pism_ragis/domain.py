# Copyright (C) 2023-24 Andy Aschwanden
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
Module provides functions to deal domains
"""

from typing import List, Union

import numpy as np
import xarray as xr


def new_range(x: np.array, dx: float):
    """
    Compute the center and half-width of a domain that will contain all values in `x`.

    The resulting half-width is an integer multiple of `dx`.

    Parameters
    ----------
    x : np.array
        A 1D array of coordinate values.
    dx : float
        The desired resolution for the new domain.

    Returns
    -------
    center : float
        The center of the new domain.
    Lx : float
        The half-width of the new domain.
    N : int
        The number of grid points in the new domain.

    Notes
    -----
    The function assumes that `x` is sorted in ascending order and that the spacing
    between consecutive elements in `x` is uniform.
    """
    x_min = x[0]
    x_max = x[-1]
    dx_old = x[1] - x[0]

    # Note: add dx_old because the cell centered grid interpretation implies
    # that the domain extends by 0.5*dx_old past x_min and x_max
    width = dx_old + (x_max - x_min)
    center = 0.5 * (x_min + x_max)

    # compute the number of grid points
    # (in a cell centered grid the numbers of points and spaces are the same)
    N = np.ceil(width / dx)

    # compute the new domain half-width
    Lx = 0.5 * N * dx

    return center, Lx, int(N)


def create_domain(
    x_bnds: Union[List, np.ndarray], y_bnds: Union[List, np.ndarray]
) -> xr.Dataset:
    """
    Create an xarray.Dataset representing a domain with specified x and y boundaries.

    Parameters
    ----------
    x_bnds : Union[List, np.ndarray]
        A list or array containing the minimum and maximum x-coordinate boundaries.
    y_bnds : Union[List, np.ndarray]
        A list or array containing the minimum and maximum y-coordinate boundaries.

    Returns
    -------
    ds : xarray.Dataset
        An xarray.Dataset containing the domain information, including coordinates,
        boundary data, and mapping attributes.

    Notes
    -----
    The dataset includes:
    - `x` and `y` coordinates with associated metadata.
    - A `mapping` DataArray with polar stereographic projection attributes.
    - A `domain` DataArray with a reference to the `mapping`.
    - `x_bnds` and `y_bnds` DataArrays representing the boundaries of the domain.
    """
    coords = {
        "x": (
            ["x"],
            [1],
            {
                "units": "m",
                "axis": "X",
                "standard_name": "projection_x_coordinate",
                "long_name": "x-coordinate in projected coordinate system",
            },
        ),
        "y": (
            ["y"],
            [1],
            {
                "units": "m",
                "axis": "Y",
                "standard_name": "projection_y_coordinate",
                "long_name": "y-coordinate in projected coordinate system",
            },
        ),
    }
    ds = xr.Dataset(
        {
            "mapping": xr.DataArray(
                data=0,
                attrs={
                    "grid_mapping_name": "polar_stereographic",
                    "false_easting": 0.0,
                    "false_northing": 0.0,
                    "latitude_of_projection_origin": 90.0,
                    "scale_factor_at_projection_origin": 1.0,
                    "standard_parallel": 70.0,
                    "straight_vertical_longitude_from_pole": -45,
                },
            ),
            "domain": xr.DataArray(
                data=0,
                attrs={
                    "dimensions": "x y",
                    "grid_mapping": "mapping",
                },
            ),
            "x_bnds": xr.DataArray(
                data=[[x_bnds[0], x_bnds[1]]],
                dims=["x", "nv2"],
                coords={"x": coords["x"]},
                attrs={"_FillValue": False},
            ),
            "y_bnds": xr.DataArray(
                data=[[y_bnds[0], y_bnds[1]]],
                dims=["y", "nv2"],
                coords={"y": coords["y"]},
                attrs={"_FillValue": False},
            ),
        },
        attrs={"Conventions": "CF-1.8"},
    )
    return ds
