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

import pint_xarray  # pylint: disable=unused-import
import xarray as xr
from dask.diagnostics import ProgressBar

if __name__ == "__main__":
    # set up the option parser
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.description = "Generating scripts for warming experiments."
    parser.add_argument("INFILE", nargs=1, help="Input file", default=None)
    parser.add_argument("OUTFILE", nargs=1, help="Input file", default=None)
    parser.add_argument(
        "--speed_file",
        type=str,
        help="""File with observed velocities. Needs to be on the same projection but not the same resolution as the INFILE.""",
        default=None,
    )
    parser.add_argument(
        "--speed_variable",
        type=str,
        help="""Variable to use. Default='v'.""",
        default="v",
    )
    parser.add_argument(
        "--speed_threshold",
        type=float,
        help="""Speed threshold. Default=200 m/yr.""",
        default=200.0,
    )

    tillwat_max = 2.0
    options = parser.parse_args()
    infile = Path(options.INFILE[0])
    outfile = Path(options.OUTFILE[0])
    speed_file = Path(options.speed_file)
    speed_var = options.speed_variable
    speed_threshold = options.speed_threshold

    pism_ds = xr.open_dataset(infile)

    pism_config = pism_ds.pism_config
    dx = pism_config.attrs["grid.dx"]
    dy = pism_config.attrs["grid.dy"]
    area = xr.DataArray(dx * dy).pint.quantify("m^2")
    area_km2 = area.pint.to("km^2")

    tillwat_mask = (pism_ds["tillwat"] > 0.0).where(pism_ds["mask"] == 2)
    speed_ds = xr.open_dataset(speed_file)
    speed_da = speed_ds[speed_var].interp_like(pism_ds["tillwat"])
    tillwat_da = pism_ds["tillwat"]
    pism_ds["tillwat"] = tillwat_da.where(speed_da <= speed_threshold, other=tillwat_max)
    tillwat_mask_updated = (pism_ds["tillwat"] > 0.0).where(pism_ds["mask"] == 2)

    twa_o = tillwat_mask.sum() * area_km2
    twa_u = tillwat_mask_updated.sum() * area_km2
    print(f"Original temperate area {twa_o.values}")
    print(f"New temperate area {twa_u.values}")
    print(f"Difference {(twa_u.values / twa_o.values) * 100 - 100 }%")
    comp = {"zlib": True, "complevel": 2, "_FillValue": None}
    encoding = {var: comp for var in pism_ds.data_vars}
    with ProgressBar():
        pism_ds.pint.dequantify().to_netcdf(outfile, encoding=encoding)
