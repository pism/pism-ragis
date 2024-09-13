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

from pism_ragis.domain import create_domain, new_range

if __name__ == "__main__":
    __spec__ = None

    # set up the option parser
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.description = "Prepare climate forcing."
    parser.add_argument(
        "--base_resolution",
        help="Base resolution in meters. Default=150.",
        type=int,
        default=150,
    )
    parser.add_argument(
        "--resolution_multipliers",
        help="""Resolution multipliers. Use "--" to seperate from positional arguments. Default="2 3 6 12 20 30".""",
        type=int,
        nargs="+",
        default=[2, 3, 6, 8, 10, 12, 20, 30],
    )

    parser.add_argument(
        "INFILE",
        nargs="?",
        help="Input netCDF file.",
    )
    parser.add_argument(
        "OUTFILE",
        nargs="?",
        help="Output netCDF file.",
    )
    options = parser.parse_args()
    base_resolution = options.base_resolution
    multipliers = np.array(options.resolution_multipliers)
    infile = options.INFILE
    outfile = options.OUTFILE
    f = Path(infile).expanduser()

    ds = xr.open_dataset(f)
    x = ds.variables["x"][:]
    y = ds.variables["y"][:]

    # set of grid resolutions, in meters
    dx = base_resolution * np.array(multipliers)
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
    # resulting set of -My values
    print(f"My values: {(2 * Ly) / dx}")

    domain_ds = create_domain(x_bnds, y_bnds)
    domain_ds.to_netcdf(outfile)
