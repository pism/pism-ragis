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

import dask.array as da
import numpy as np
import xarray as xr
from dask.diagnostics import ProgressBar

from pism_ragis.processing import download_earthaccess_dataset, preprocess_time

xr.set_options(keep_attrs=True)
# Suppress specific warning from loky
warnings.filterwarnings("ignore", category=UserWarning)


def save(
    ds: xr.Dataset,
    output_filename: Union[str, Path] = "GRE_G0240_1985_2018_IDW_EXP_1.nc",
    comp={"zlib": True, "complevel": 2},
):
    """
    Save the xarray dataset to a NetCDF file with specified compression.

    Parameters
    ----------
    ds : xarray.Dataset
        The dataset to be saved.
    output_filename : Union[str, Path], optional
        The output filename for the NetCDF file, by default "GRE_G0240_1985_2018_IDW_EXP_1.nc".
    comp : dict, optional
        Compression settings for the NetCDF file, by default {"zlib": True, "complevel": 2}.

    Returns
    -------
    None
    """
    encoding = {var: comp for var in ds.data_vars}
    ds.to_netcdf(output_filename, encoding=encoding)


def idw_weights(distance: xr.DataArray, power: float = 1.0):
    """
    Calculate inverse distance weighting (IDW) weights.

    Parameters
    ----------
    distance : xarray.DataArray
        The array of distances.
    power : float, optional
        The power parameter for IDW, by default 1.0.

    Returns
    -------
    xarray.DataArray
        The calculated IDW weights.
    """
    return 1.0 / (distance + 1e-12) ** power


if __name__ == "__main__":
    __spec__ = None

    # set up the option parser
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.description = "Prepare ITS_LIVE."
    options = parser.parse_args()

    print("Preparing ITS_LIVE")
    filter_str = "GRE_G0240"
    result_dir = Path("itslive")
    doi = "10.5067/6II6VW8LLWJ7"
    result = download_earthaccess_dataset(
        doi=doi, filter_str=filter_str, result_dir=result_dir
    )

    comp = {"zlib": True, "complevel": 2}
    regexp = "GRE_G0240_(.+?).nc"
    vars_to_process = ["v", "vx", "vy", "v_err", "vx_err", "vy_err", "ice"]

    output_files = []
    for r in result:
        p = Path(r)
        ds = xr.open_dataset(p, chunks="auto")[vars_to_process]
        ds = preprocess_time(ds, regexp=regexp, freq="YS")
        encoding = {
            var: comp for var in ds.data_vars if var not in ("time", "time_bounds")
        }
        if p.name != "GRE_G0240_0000.nc":
            ofile = "ITS_LIVE_" + p.name
            output_file = p.with_name(ofile)
            print(f"Saving to {output_file}")
            # with ProgressBar():
            #     ds.to_netcdf(output_file, encoding=encoding)
            output_files.append(output_file)
        del ds

    power = 1
    ds = xr.open_mfdataset(
        output_files,
        parallel=False,
        chunks={"time": -1},
    )
    ds = ds.where(ds["ice"])
    nt = ds.time.size
    dt = xr.DataArray(
        da.arange(nt, chunks=-1),
        dims=("time"),
    )
    speed = ds["v"]
    distance = np.isfinite(speed) * dt.broadcast_like(speed)
    weights = idw_weights(distance, power=power)
    idw_ofile = result_dir / Path(f"GRE_G0240_1985_2018_IDW_EXP_{power}.nc")
    print(f"Inverse-Distance Weighting with power = {power} and saving to {idw_ofile}")
    with ProgressBar():
        weighted_mean = ds.weighted(weights).mean(dim="time")
        save(weighted_mean, idw_ofile)
