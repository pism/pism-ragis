# Copyright (C) 2024 Andy Aschwanden, Constantine Khroulev
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
Compute basins.
"""

# pylint: disable=redefined-outer-name,unused-import

import re
import time
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from collections import OrderedDict
from functools import partial
from pathlib import Path
from typing import Union

import dask
import geopandas as gp
import numpy as np
import pint_xarray
import rioxarray
import xarray as xr
from dask.distributed import Client, progress

from pism_ragis.processing import compute_basin

xr.set_options(keep_attrs=True)

if __name__ == "__main__":
    __spec__ = None  # type: ignore

    # set up the option parser
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.description = "Compute ensemble statistics."
    parser.add_argument(
        "--cf",
        help="""Make output file CF Convetions compliant. Default="False".""",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--crs",
        help="""Coordinate reference system. Default is EPSG:3413.""",
        type=str,
        default="EPSG:3413",
    )
    parser.add_argument(
        "--engine",
        help="""Engine for xarray. Default="netcdf4".""",
        type=str,
        default="netcdf4",
    )
    parser.add_argument(
        "--ensemble",
        help="""Name of the ensemble. Default=RAGIS.""",
        type=str,
        default="RAGIS",
    )
    parser.add_argument(
        "--regexp",
        help="""Regular expression.""",
        type=str,
        default="id_(.+?)_",
    )
    parser.add_argument(
        "--result_dir",
        help="""Result directory.""",
        type=str,
        default="./results",
    )
    parser.add_argument(
        "--basin_url",
        help="""Basin shapefile.""",
        type=str,
        default="data/mouginot/Greenland_Basins_PS_v1.4.2_w_shelves.gpkg",
    )
    parser.add_argument(
        "--temporal_range",
        help="""Time slice to extract.""",
        type=str,
        nargs=2,
        default=None,
    )
    parser.add_argument("FILE", nargs=1, help="netCDF file to process", default=None)

    options = parser.parse_args()
    cf = options.cf
    crs = options.crs
    engine = options.engine
    ensemble = options.ensemble
    regexp = options.regexp
    result_dir = Path(options.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    basin_url = Path(options.basin_url)
    basins = gp.read_file(basin_url).to_crs(crs)

    mb_vars = [
        "ice_mass",
        "ice_mass_transport_across_grounding_line",
        "tendency_of_ice_mass",
        "tendency_of_ice_mass_due_to_basal_mass_flux",
        "tendency_of_ice_mass_due_to_basal_mass_flux_grounded",
        "tendency_of_ice_mass_due_to_basal_mass_flux_floating",
        # "tendency_of_ice_mass_due_to_frontal_melt",
        "tendency_of_ice_mass_due_to_discharge",
        "tendency_of_ice_mass_due_to_surface_mass_flux",
        "tendency_of_ice_mass_due_to_conservation_error",
        "tendency_of_ice_mass_due_to_flow",
        # "tendency_of_ice_mass_due_to_calving",
        # "tendency_of_ice_mass_due_to_forced_retreat",
    ]

    client = Client()
    print(f"Open client in browser: {client.dashboard_link}")

    start = time.time()
    time_coder = xr.coders.CFDatetimeCoder(use_cftime=False)

    with dask.config.set(**{"array.slicing.split_large_chunks": True}):
        ds = xr.open_dataset(options.FILE[-1], decode_timedelta=True, decode_times=time_coder, chunks="auto")
        stats = ds[["pism_config", "run_stats"]]

        m_id_re = re.search(regexp, ds.encoding["source"])
        assert m_id_re is not None
        m_id: Union[str, int]
        try:
            m_id = int(m_id_re.group(1))
        except:
            m_id = str(m_id_re.group(1))

        p_config = ds["pism_config"]
        p_run_stats = ds["run_stats"]

        # List of suffixes to exclude
        suffixes_to_exclude = ["_doc", "_type", "_units", "_option", "_choices"]

        # Filter the dictionary
        config = {
            k: v for k, v in p_config.attrs.items() if not any(k.endswith(suffix) for suffix in suffixes_to_exclude)
        }
        if "geometry.front_retreat.prescribed.file" not in config.keys():
            config["geometry.front_retreat.prescribed.file"] = "false"

        config_sorted = OrderedDict(sorted(config.items()))

    if cf:
        pc_keys = np.array(list(config_sorted.keys()), dtype="S1024")
        pc_vals = np.array(list(config_sorted.values()), dtype="S128")
        rs_keys = np.array(list(stats.attrs.keys()), dtype="S1024")
        rs_vals = np.array(list(stats.attrs.values()), dtype="S128")
    else:
        pc_keys = list(config_sorted.keys())
        pc_vals = list(config_sorted.values())
        rs_keys = list(stats.attrs.keys())
        rs_vals = list(stats.attrs.values())

        pism_config = xr.DataArray(
            pc_vals,
            dims="pism_config_axis",
            coords={"pism_config_axis": pc_keys},
            name="pism_config",
        )
        run_stats = xr.DataArray(
            rs_vals,
            dims="run_stats_axis",
            coords={"run_stats_axis": rs_keys},
            name="run_stats",
        )

        if "ice_mass" in ds:
            ds["ice_mass"] /= 1e12
            ds["ice_mass"].attrs["units"] = "Gt"

        if options.temporal_range:
            ds = ds.sel(time=slice(options.temporal_range[0], options.temporal_range[1]))

    bmb_var = "tendency_of_ice_mass_due_to_basal_mass_flux"
    if bmb_var in ds:
        bmb_grounded_da = ds[bmb_var].where(ds["mask"] == 2)
        bmb_grounded_da.name = "tendency_of_ice_mass_due_to_basal_mass_flux_grounded"
        bmb_floating_da = ds[bmb_var].where(ds["mask"] == 3)
        bmb_floating_da.name = "tendency_of_ice_mass_due_to_basal_mass_flux_floating"
        ds = xr.merge([ds, bmb_grounded_da, bmb_floating_da])

    del ds["time"].attrs["bounds"]
    ds = ds.drop_vars(["time_bounds", "timestamp"], errors="ignore").rio.set_spatial_dims(x_dim="x", y_dim="y")
    ds.rio.write_crs(crs, inplace=True)
    ds = ds[mb_vars].drop_vars(["pism_config", "run_stats"], errors="ignore")
    print(f"Size in memory: {(ds.nbytes / 1024**3):.1f} GB")

    basins_file = result_dir / f"basins_sums_ensemble_{ensemble}_id_{m_id}.nc"

    basins_ds_scattered = client.scatter([ds] + [ds.rio.clip([basin.geometry]) for _, basin in basins.iterrows()])
    basin_names = ["GRACE"] + [basin["SUBREGION1"] for _, basin in basins.iterrows()]
    n_basins = len(basin_names)
    futures = client.map(compute_basin, basins_ds_scattered, basin_names)
    progress(futures)
    result = client.gather(futures)
    basin_sums = xr.concat(result, dim="basin").drop_vars(["mapping", "spatial_ref"]).sortby(["basin"])
    basin_sums = xr.merge([basin_sums, pism_config, run_stats])
    basin_sums = basin_sums.expand_dims({"exp_id": [m_id]})
    if cf:
        basin_sums["basin"] = basin_sums["basin"].astype(f"S{n_basins}")
        basin_sums.attrs["Conventions"] = "CF-1.8"

    basin_sums.to_netcdf(basins_file, engine=engine)

    client.close()
    end = time.time()
    time_elapsed = end - start
    print(f"Time elapsed {time_elapsed:.0f}s")
