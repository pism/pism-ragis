# Copyright (C) 2024 Andy Aschwanden
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
Adjust 'tillwat' in a PISM state file based on observed surface speeds.
"""

from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from pathlib import Path
from dask.diagnostics import ProgressBar
import xarray as xr

if __name__ == "__main__":
    # set up the option parser
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.description = "Generating scripts for warming experiments."
    parser.add_argument(
        "INFILE", nargs=1, help="Input file", default=None
    )
    parser.add_argument(
        "OUTFILE", nargs=1, help="Input file", default=None
    )
    parser.add_argument(
        "--speed_file",
        type=str,
        help="""File with observed velocities. Needs to be on the same projection but not the same resolution as the INFILE.""",
        default=None
    )
    parser.add_argument(
        "--speed_variable",
        type=str,
        help="""Variable to use. Default='v'.""",
        default="v"
    )

    speed_threshold = 200.0
    tillwat_max = 2.0
    options = parser.parse_args()
    infile = Path(options.INFILE[0])
    outfile = Path(options.OUTFILE[0])
    speed_file = Path(options.speed_file)
    speed_var = options.speed_variable

    pism_ds = xr.open_dataset(infile)
    speed_ds = xr.open_dataset(speed_file)
    # pism_ds["tillwat"] = pism_ds["tillwat"].where(speed_ds[speed_var].interp_like(pism_ds) > speed_threshold, tillwat_max)
    # comp={"zlib": True, "complevel": 2},
    # encoding = {var: comp for var in pism_ds.data_vars}
    # with ProgressBar():
    #     pism_ds.to_netcdf(outfile, encoding=encoding)

