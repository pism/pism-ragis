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
Adjust 'ftt_mask' to input file.
"""

from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from pathlib import Path

import xarray as xr
from dask.diagnostics import ProgressBar

if __name__ == "__main__":
    # set up the option parser
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.description = "Generating scripts for warming experiments."
    parser.add_argument("INFILE", nargs=1, help="Input file", default=None)
    parser.add_argument("OUTFILE", nargs=1, help="Input file", default=None)
    options = parser.parse_args()
    infile = Path(options.INFILE[0])
    outfile = Path(options.OUTFILE[0])

    ds = xr.open_dataset(infile).fillna(0)
    ds["ftt_mask"] = xr.full_like(ds["mask"], fill_value=False, dtype=bool)
    ds["ftt_mask"] = ds["ftt_mask"].where(ds["mask"] != (2 or 3), 1)
    ds["ftt_mask"].attrs.update(
        {
            "flag_meanings": "no_ftt apply_ftt",
            "long_name": "force to thickness mask",
            "flag_values": [0, 1],
            "valid_range": [0, 1],
        }
    )

    comp = {"zlib": True, "complevel": 2, "_FillValue": None}
    encoding = {var: comp for var in ds.data_vars}
    with ProgressBar():
        ds.to_netcdf(outfile, encoding=encoding)
