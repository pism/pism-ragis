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
Prepare ITS_LIVE.
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


if __name__ == "__main__":
    __spec__ = None

    # set up the option parser
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.description = "Prepare ITS_LIVE."
    options = parser.parse_args()

    print("Prepare ITS_LIVE")
    filter_str = "GRE_G0240"
    result_dir = Path("itslive")
    doi = "10.5067/6II6VW8LLWJ7"
    result = download_earthaccess_dataset(
        doi=doi, filter_str=filter_str, result_dir=result_dir
    )
