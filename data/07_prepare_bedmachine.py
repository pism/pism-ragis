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

import numpy as np
import xarray as xr

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
    parser.description = "Prepare BedMachine."
    options = parser.parse_args()

    filter_str = "v5.nc"
    result_dir = Path("bedmachine")
    doi = "10.5067/GMEVBWFLWA7X"
    result = download_earthaccess_dataset(
        doi=doi, filter_str=filter_str, result_dir=result_dir
    )
