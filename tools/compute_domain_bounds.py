# Copyright (C) 2024 Constantine Khroulev, Andy Aschwanden
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
Compute domain bounds
"""

from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from pathlib import Path

import numpy as np
import xarray as xr


def new_range(x: np.array, dx: float):
    """Use values of a coordinate variable `x` to compute
    the center and the half-width of a domain that will contain all values in `x`.

    The resulting half-width is an integer multiple of `dx`.
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


if __name__ == "__main__":
    __spec__ = None

    # set up the option parser
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.description = "Prepare climate forcing."
    parser.add_argument("FILE", nargs=1, help="""netCDF file.""")
    options = parser.parse_args()
    infile = options.FILE[-1]
    f = Path(infile).expanduser()

    ds = xr.open_dataset(f)
    print("Coordinate variables\n", ds.coords)
    x = ds.variables["x"][:]
    y = ds.variables["y"][:]

    # set of grid resolutions, in meters
    dx = 150 * np.array([3, 6, 12, 20, 60, 120])
    print(f"resolutions: {dx} meters")

    # compute x_bnds for this set of resolutions
    center, Lx, Mx = new_range(x.values, np.lcm.reduce(dx))
    x_bnds = [center - Lx, center + Lx]
    print(f"new x bounds: {x_bnds}")

    # compute x_bnds for this set of resolutions
    center, Ly, Mx = new_range(y.values, np.lcm.reduce(dx))
    y_bnds = [center - Ly, center + Ly]
    print(f"new y bounds: {y_bnds}")

    # resulting set of -Mx values
    print(f"Mx values: {(2 * Lx) / dx}")
    # resulting set of -Mx values
    print(f"My values: {(2 * Ly) / dx}")

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

    domain_ds = xr.Dataset(
        {
            "domain": xr.DataArray(
                data=0,
                dims=["y", "x"],
                coords=coords,
                attrs={
                    "grid_mapping": "Polar_Stereographic",
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
    domain_ds["Polar_Stereographic"] = int()
    domain_ds.Polar_Stereographic.attrs["grid_mapping_name"] = "polar_stereographic"
    domain_ds.Polar_Stereographic.attrs["false_easting"] = 0.0
    domain_ds.Polar_Stereographic.attrs["false_northing"] = 0.0
    domain_ds.Polar_Stereographic.attrs["latitude_of_projection_origin"] = 90.0
    domain_ds.Polar_Stereographic.attrs["scale_factor_at_projection_origin"] = 1.0
    domain_ds.Polar_Stereographic.attrs["standard_parallel"] = 70.0
    domain_ds.Polar_Stereographic.attrs["straight_vertical_longitude_from_pole"] = -45

    domain_ds.to_netcdf("domain.nc")
