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
Compute domain bounds.
"""

from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from pathlib import Path

import numpy as np
import xarray as xr

from pism_ragis.domain import create_domain, get_bounds

if __name__ == "__main__":
    __spec__ = None  # type: ignore

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
        "--crs",
        help="Coordinate reference system. Default: epsg:3413.",
        type=str,
        default="EPSG:3413",
    )
    parser.add_argument(
        "--resolution_multipliers",
        help="""Resolution multipliers. Use "--" to seperate from positional arguments. Default="2 3 6 8 10 12 16 20 24 30".""",
        type=int,
        nargs="+",
        default=[1, 2, 3, 6, 8, 10, 12, 16, 20, 24, 30],
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
    crs = options.crs
    multipliers = np.array(options.resolution_multipliers)
    infile = options.INFILE
    outfile = options.OUTFILE
    f = Path(infile).expanduser()

    ds = xr.open_dataset(f)
    x_bnds, y_bnds = get_bounds(ds)
    domain_ds = create_domain(x_bnds, y_bnds)
    domain_ds.to_netcdf(outfile)
