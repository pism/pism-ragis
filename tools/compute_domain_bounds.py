from pathlib import Path

import numpy as np
from netCDF4 import Dataset as NC


def new_range(x, dx):
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


f = NC(
    Path(
        "~/github/pism/example-inputs/std-greenland/Greenland_5km_v1.1.nc"
    ).expanduser()
)

x = f.variables["x1"][:]
y = f.variables["y1"][:]

# set of grid resolutions, in meters
dx = 150 * np.array([3, 6, 12, 20, 60, 120])
print(f"resolutions: {dx} meters")

# compute x_bnds for this set of resolutions
center, Lx, Mx = new_range(x, np.lcm.reduce(dx))
print(f"new x bounds: {[center - Lx, center + Lx]}")

# resulting set of -Mx values
print(f"Mx values: {(2 * Lx) / dx}")
