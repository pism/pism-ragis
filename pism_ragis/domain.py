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

# pylint: disable=too-many-positional-arguments,unused-import

"""
Module provides functions to handle domains.
"""

import geopandas as gp
import numpy as np
import rioxarray
import xarray as xr


def new_range(x: np.array, dx: float) -> tuple[float, float, int]:
    """
    Compute the center and half-width of a domain that will contain all values in `x`.

    The resulting half-width is an integer multiple of `dx`.

    Parameters
    ----------
    x : numpy.array
        A 1D array of coordinate values.
    dx : float
        The desired resolution for the new domain.

    Returns
    -------
    tuple
        A tuple containing:
        - center (float): The center of the new domain.
        - Lx (float): The half-width of the new domain.
        - N (int): The number of grid points in the new domain.

    Notes
    -----
    The function assumes that `x` is sorted in ascending order and that the spacing
    between consecutive elements in `x` is uniform.
    """
    x_min = np.min(x)
    x_max = np.max(x)
    dx_old = np.abs(x[1] - x[0])

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


def get_bounds(
    ds: xr.Dataset,
    base_resolution: int = 150,
    multipliers: list | np.ndarray = [1, 2, 3, 6, 8, 10, 12, 16, 20, 24, 30],
) -> tuple[list[float], list[float]]:
    """
    Compute the x and y boundaries for a given dataset and set of grid resolutions.

    Parameters
    ----------
    ds : xarray.Dataset
        The input dataset containing the x and y coordinates.
    base_resolution : int, optional
        The base resolution in meters, by default 150.
    multipliers : list or numpy.ndarray, optional
        A list or array of multipliers to compute the set of grid resolutions,
        by default [1, 2, 3, 6, 8, 10, 12, 16, 20, 24, 30].

    Returns
    -------
    tuple of list of float
        A tuple containing:
        - x boundaries (list of float)
        - y boundaries (list of float)

    Examples
    --------
    >>> ds = xr.Dataset({'x': ('x', np.linspace(0, 1000, 11)), 'y': ('y', np.linspace(0, 2000, 21))})
    >>> x_bnds, y_bnds = get_bounds(ds)
    >>> print(x_bnds, y_bnds)
    """
    x = ds.variables["x"][:]
    y = ds.variables["y"][:]

    # set of grid resolutions, in meters
    dx = base_resolution * np.array(multipliers)

    # compute x_bnds for this set of resolutions
    center, Lx, _ = new_range(x.values, np.lcm.reduce(dx))
    x_bnds = [center - Lx, center + Lx]

    # compute y_bnds for this set of resolutions
    center, Ly, _ = new_range(y.values, np.lcm.reduce(dx))
    y_bnds = [
        np.minimum(center - Ly, center + Ly),
        np.maximum(center - Ly, center + Ly),
    ]
    return x_bnds, y_bnds


def create_local_grid(
    series: gp.GeoSeries,
    ds: xr.Dataset,
    buffer: float = 500,
    base_resolution: int = 150,
    multipliers: list | np.ndarray = [1, 2, 3, 4, 6, 8, 10, 12, 16, 20, 24, 30],
) -> xr.Dataset:
    """
    Create a local grid around a GeoSeries geometry with a buffer.

    Parameters
    ----------
    series : geopandas.GeoSeries
        The GeoSeries containing the geometry to buffer.
    ds : xarray.Dataset
        The dataset containing the x and y coordinates.
    buffer : float, optional
        The buffer distance around the geometry, by default 500.
    base_resolution : int, optional
        The base resolution in meters, by default 150.
    multipliers : list or numpy.ndarray, optional
        A list or array of multipliers to compute the set of grid resolutions,
        by default [1, 2, 3, 4, 6, 8, 10, 12, 16, 20, 24, 30].

    Returns
    -------
    xarray.Dataset
        A dataset representing the local grid.

    Notes
    -----
    The function uses the buffered geometry to determine the bounds of the local grid.
    """
    minx, miny, maxx, maxy = series["geometry"].buffer(buffer).bounds
    max_mult = multipliers[-1]
    resolution_coarse = base_resolution * max_mult
    x_bnds, y_bnds = get_bounds(ds, base_resolution=base_resolution, multipliers=multipliers)
    coarse_ds = create_domain(x_bnds, y_bnds, resolution=resolution_coarse)

    ll = coarse_ds.sel({"x": minx, "y": miny}, method="nearest")
    ur = coarse_ds.sel({"x": maxx, "y": maxy}, method="nearest")

    if miny < maxy:
        local_ds = coarse_ds.sel({"x": slice(ll["x"], ur["x"]), "y": slice(ll["y"], ur["y"])})
    else:
        local_ds = coarse_ds.sel({"x": slice(ll["x"], ur["x"]), "y": slice(ur["y"], ll["y"])})

    x_bnds, y_bnds = [local_ds["x_bnds"][0, 0], local_ds["x_bnds"][-1, -1]], [
        local_ds["y_bnds"][0, 0],
        local_ds["y_bnds"][-1, -1],
    ]
    grid = create_domain(x_bnds, y_bnds)
    return grid


def create_domain(
    x_bnds: list | np.ndarray,
    y_bnds: list | np.ndarray,
    resolution: float | None = None,
    x_dim: str = "x",
    y_dim: str = "y",
    crs: str = "EPSG:3413",
) -> xr.Dataset:
    """
    Create an xarray.Dataset representing a domain with specified x and y boundaries.

    Parameters
    ----------
    x_bnds : list or numpy.ndarray
        A list or array containing the minimum and maximum x-coordinate boundaries.
    y_bnds : list or numpy.ndarray
        A list or array containing the minimum and maximum y-coordinate boundaries.
    resolution : float or None, optional
        The resolution of the grid, by default None.
    x_dim : str, optional
        The name of the x dimension, by default "x".
    y_dim : str, optional
        The name of the y dimension, by default "y".
    crs : str, optional
        The coordinate reference system (CRS) for the domain, by default "EPSG:3413".

    Returns
    -------
    xarray.Dataset
        An xarray.Dataset containing the domain information, including coordinates,
        boundary data, and mapping attributes.

    Notes
    -----
    The dataset includes:
    - `x` and `y` coordinates with associated metadata.
    - A `mapping` DataArray with polar stereographic projection attributes.
    - A `domain` DataArray with a reference to the `mapping`.
    - `x_bnds` and `y_bnds` DataArrays representing the boundaries of the domain.

    Examples
    --------
    >>> x_bnds = [0, 1000]
    >>> y_bnds = [0, 2000]
    >>> ds = create_domain(x_bnds, y_bnds)
    >>> print(ds)
    """

    if resolution is not None:
        x = np.arange(x_bnds[0] + resolution / 2, x_bnds[1], resolution)
        y = np.arange(y_bnds[0] + resolution / 2, y_bnds[1], resolution)
        xb = np.arange(x_bnds[0], x_bnds[1] + resolution, resolution)
        yb = np.arange(y_bnds[0], y_bnds[1] + resolution, resolution)
        x_bounds = np.stack([xb[:-1], xb[1:]]).T
        y_bounds = np.stack([yb[:-1], yb[1:]]).T
    else:
        x = [0]
        y = [0]
        x_bounds = [[x_bnds[0], x_bnds[1]]]
        y_bounds = [[y_bnds[0], y_bnds[1]]]

    x_bnds_dim = f"{x_dim}_bnds"
    y_bnds_dim = f"{y_dim}_bnds"
    coords = {
        x_dim: (
            [x_dim],
            x,
            {
                "units": "m",
                "axis": x_dim.upper(),
                "bounds": x_bnds_dim,
                "standard_name": "projection_x_coordinate",
                "long_name": f"{x_dim}-coordinate in projected coordinate system",
            },
        ),
        y_dim: (
            [y_dim],
            y,
            {
                "units": "m",
                "axis": y_dim.upper(),
                "bounds": y_bnds_dim,
                "standard_name": "projection_y_coordinate",
                "long_name": f"{y_dim}-coordinate in projected coordinate system",
            },
        ),
    }
    ds = xr.Dataset(
        {
            "domain": xr.DataArray(
                data=0,
                dims=[y_dim, x_dim],
                coords={x_dim: coords[x_dim], y_dim: coords[y_dim]},
                attrs={
                    "dimensions": f"{x_dim} {y_dim}",
                },
            ),
            x_bnds_dim: xr.DataArray(
                data=x_bounds,
                dims=[x_dim, "nv2"],
                coords={x_dim: coords[x_dim]},
                attrs={"_FillValue": False},
            ),
            y_bnds_dim: xr.DataArray(
                data=y_bounds,
                dims=[y_dim, "nv2"],
                coords={y_dim: coords[y_dim]},
                attrs={"_FillValue": False},
            ),
        },
        attrs={"Conventions": "CF-1.8"},
    ).rio.set_spatial_dims(x_dim=x_dim, y_dim=y_dim)
    ds.rio.write_crs(crs, inplace=True)
    return ds
