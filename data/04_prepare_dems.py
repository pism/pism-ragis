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
Prepare GrIMP and BedMachine.
"""
# pylint: disable=unused-import,assignment-from-none,unexpected-keyword-arg

import time
import warnings
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from pathlib import Path
from typing import Dict, Union

import dask
import numpy as np
import rioxarray as rxr
import xarray as xr
import xdem
from dask.diagnostics import ProgressBar

from pism_ragis.download import download_earthaccess, save_netcdf

xr.set_options(keep_attrs=True)
# Suppress specific warning from loky
warnings.filterwarnings("ignore", category=UserWarning)

if __name__ == "__main__":
    __spec__ = None

    # set up the option parser
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.description = "Prepare GrIMP and BedMachine."
    options = parser.parse_args()

    filter_str = "90m_v"
    result_dir = Path("dem")
    result_dir.mkdir(parents=True, exist_ok=True)
    grimp_dir = result_dir / Path("grimp")
    grimp_dir.mkdir(parents=True, exist_ok=True)

    print("Preparing GrIMP")
    doi = "10.5067/NV34YUIXLP9W"
    results = download_earthaccess(doi=doi, filter_str=filter_str, result_dir=grimp_dir)
    dem_file = result_dir / Path("egm96_gimpdem_90m_v01.1.tif")
    print(
        f"Converting Ellipsoid to EGM96 and saving as {dem_file}...", end="", flush=True
    )
    # dem = xdem.DEM(r)
    # dem_egm96 = dem.to_vcrs("EGM96", force_source_vcrs="Ellipsoid")
    # dem_egm96.save(dem_file)
    print("Done")

    doi = "10.5067/B8X58MQBFUPA"
    results = download_earthaccess(
        doi=doi, filter_str=filter_str, result_dir=result_dir
    )
    mask_file = results[0]

    dem_da = rxr.open_rasterio(dem_file).squeeze()
    dem_da.name = "usurf"
    dem_da = dem_da.assign_attrs({"units": "m", "standard_name": "surface_altitude"})
    dem_uncertainty_da = xr.zeros_like(dem_da) + 30
    dem_uncertainty_da.name = "usurf_uncertainty"
    dem_uncertainty_da = dem_uncertainty_da.assign_attrs({"units": "m", "standard_name": None})
    mask_da = rxr.open_rasterio(mask_file).squeeze()
    mask_da.name = "mask"
    mask_da = mask_da.assign_attrs({"mask": "m"})
    grimp_ds = xr.merge([dem_da, dem_uncertainty_da, mask_da])
    grimp_file = result_dir / Path("grimp_90m.nc")

    save_netcdf(grimp_file)

    print("Preparing BedMachine")
    result_dir = Path("dem")
    filter_str = "v5.nc"
    doi = "10.5067/GMEVBWFLWA7X"
    results = download_earthaccess(
        doi=doi, filter_str=filter_str, result_dir=result_dir
    )
