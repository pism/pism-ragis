# Copyright (C) 2024-25 Andy Aschwanden, Constantine Khroulev
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

# pylint: disable=unused-import,too-many-positional-arguments,unused-argument
"""
Analyze RAGIS ensemble.
"""

import re
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from pathlib import Path

import xarray as xr
from dask.diagnostics import ProgressBar

from pism_ragis.download import save_netcdf
from pism_ragis.processing import preprocess_nc as preprocess

if __name__ == "__main__":
    __spec__ = None  # type: ignore

    # set up the option parser
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.description = "Compute ensemble statistics."
    parser.add_argument(
        "--result_dir",
        help="""Result directory.""",
        type=str,
        default="./results",
    )

    parser.add_argument(
        "FILES",
        help="""Ensemble netCDF files.""",
        nargs="*",
    )

    options = parser.parse_args()
    time_decoder = xr.coders.CFDatetimeCoder(use_cftime=False)

    infiles = options.FILES
    result_dir = Path(options.result_dir)
    result_dir.mkdir(exist_ok=True, parents=True)

    for infile in infiles:
        p = Path(infile)
        ds = xr.open_dataset(
            infile,
            decode_times=time_decoder,
            decode_timedelta=True,
            engine="h5netcdf",
        )
        for v in ds.data_vars:

            print(v, ds[v].dtype)
        ds = preprocess(ds, drop_vars=["time_bounds"], drop_dims=["nv"])
        print(ds)
        outfile = result_dir / p.name
        with ProgressBar():
            ds.to_netcdf(outfile)
