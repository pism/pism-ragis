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
Prepare CALFIN front retreat.
"""
# pylint: disable=unused-import

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

from pism_ragis.processing import download_earthaccess_dataset

xr.set_options(keep_attrs=True)
# Suppress specific warning from loky
warnings.filterwarnings("ignore", category=UserWarning)

x_min = -653000
x_max = 879700
y_min = -632750
y_max = -3384350
bbox = [x_min, y_min, x_max, y_max]
geom = {
    "type": "Polygon",
    "crs": {"properties": {"name": "EPSG:3413"}},
    "bbox": bbox,
    "coordinates": [
        [
            (x_min, y_min),
            (x_max, y_min),
            (x_max, y_max),
            (x_min, y_max),
            (x_min, y_min),  # Close the loop by repeating the first point
        ]
    ],
}


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

    doi = "10.5067/NV34YUIXLP9W"
    result = download_earthaccess_dataset(
        doi=doi, filter_str=filter_str, result_dir=grimp_dir
    )
    r = Path(result[0])
    dem_file = result_dir / Path("egm96_gimpdem_90m_v01.1.tif")
    print(
        f"Converting Ellipsoid to EGM96 and saving as {dem_file}...", end="", flush=True
    )
    dem = xdem.DEM(r)
    dem_egm96 = dem.to_vcrs("EGM96", force_source_vcrs="Ellipsoid")
    dem_egm96.save(dem_file)
    print("Done")

    doi = "10.5067/B8X58MQBFUPA"
    result = download_earthaccess_dataset(
        doi=doi, filter_str=filter_str, result_dir=result_dir
    )
    mask_file = result[0]

    dem_da = rxr.open_rasterio(dem_file).squeeze()
    dem_da.name = "usurf"
    dem_da = dem_da.assign_attrs({"units": "m", "standard_name": "surface_altitude"})
    dem_uncertainty_da = xr.zeros_like(dem_da) + 30
    dem_uncertainty_da.name = "usurf_uncertainty"
    dem_uncertainty_da = dem_uncertainty_da.assign_attrs({"units": "m"})
    mask_da = rxr.open_rasterio(mask_file).squeeze()
    mask_da.name = "mask"
    mask_da = mask_da.assign_attrs({"mask": "m"})
    grimp_ds = xr.merge([dem_da, dem_uncertainty_da, mask_da])
    grimp_file = result_dir / Path("grimp_90m.nc")

    comp = {"zlib": True, "complevel": 2}
    encoding = {var: comp for var in grimp_ds.data_vars}
    grimp_ds.to_netcdf(grimp_file, encoding=encoding)

    result_dir = Path("dem")
    filter_str = "v5.nc"
    doi = "10.5067/GMEVBWFLWA7X"
    result = download_earthaccess_dataset(
        doi=doi, filter_str=filter_str, result_dir=result_dir
    )
