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

# pylint: disable=redefined-outer-name

import re
import time
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from pathlib import Path
from typing import Union

import dask
import geopandas as gp
import xarray as xr
from dask.distributed import Client, LocalCluster, progress

from pism_ragis.processing import compute_basin

if __name__ == "__main__":
    __spec__ = None

    # set up the option parser
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.description = "Compute ensemble statistics."
    parser.add_argument(
        "--ensemble",
        help="""Name of the ensemble. Default=RAGIS.""",
        type=str,
        default="RAGIS",
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
    parser.add_argument(
        "--crs",
        help="""Coordinate reference system. Default is EPSG:3413.""",
        type=str,
        default="EPSG:3413",
    )
    parser.add_argument(
        "--memory_limit",
        help="""Maximum memory used by Client. Default=16GB.""",
        type=str,
        default="16GB",
    )
    parser.add_argument(
        "--n_jobs", help="""Number of parallel jobs.""", type=int, default=4
    )
    parser.add_argument("FILE", nargs=1, help="netCDF file to process", default=None)

    options = parser.parse_args()
    crs = options.crs
    n_jobs = options.n_jobs

    ensemble = options.ensemble
    result_dir = Path(options.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    # Load basins, merge all ICE_CAP geometries
    basin_url = Path(options.basin_url)
    basins = gp.read_file(basin_url).to_crs(crs)

    mb_vars = [
        "ice_mass",
        "ice_mass_transport_across_grounding_line",
        "tendency_of_ice_mass",
        "tendency_of_ice_mass_due_to_basal_mass_flux",
        "tendency_of_ice_mass_due_to_basal_mass_flux_grounded",
        "tendency_of_ice_mass_due_to_basal_mass_flux_floating",
        "tendency_of_ice_mass_due_to_frontal_melt",
        "tendency_of_ice_mass_due_to_discharge",
        "tendency_of_ice_mass_due_to_surface_mass_flux",
        "tendency_of_ice_mass_due_to_conservation_error",
        "tendency_of_ice_mass_due_to_flow",
        "tendency_of_ice_mass_due_to_calving",
        "tendency_of_ice_mass_due_to_forced_retreat",
    ]
    regexp: str = "id_(.+?)_"

    with dask.config.set(**{"array.slicing.split_large_chunks": True}):
        ds = xr.open_dataset(
            options.FILE[-1],
            chunks="auto",
        )
        m_id_re = re.search(regexp, ds.encoding["source"])
        assert m_id_re is not None
        m_id: Union[str, int]
        try:
            m_id = int(m_id_re.group(1))
        except:
            m_id = str(m_id_re.group(1))

        ds = ds.expand_dims(
            {"ensemble_id": [ensemble], "exp_id": [m_id]}, axis=[-1, -2]
        )

        if "ice_mass" in ds:
            ds["ice_mass"] /= 1e12
            ds["ice_mass"].attrs["units"] = "Gt"

        if options.temporal_range:
            ds = ds.sel(time=slice(options.temporal_range))

    bmb_var = "tendency_of_ice_mass_due_to_basal_mass_flux"
    if bmb_var in ds:
        bmb_grounded_da = ds[bmb_var].where(ds["mask"] == 2)
        bmb_grounded_da.name = "tendency_of_ice_mass_due_to_basal_mass_flux_grounded"
        bmb_floating_da = ds[bmb_var].where(ds["mask"] == 3)
        bmb_floating_da.name = "tendency_of_ice_mass_due_to_basal_mass_flux_floating"
        ds = xr.merge([ds, bmb_grounded_da, bmb_floating_da])

    config = ds["pism_config"]
    stats = ds["run_stats"]
    ds = ds[mb_vars]
    ds.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=True)
    ds.rio.write_crs(crs, inplace=True)

    pism_config = xr.DataArray(
        list(config.attrs.values()),
        dims=["pism_config_axis"],
        coords={"pism_config_axis": list(config.attrs.keys())},
        name="pism_config",
    )
    run_stats = xr.DataArray(
        list(stats.attrs.values()),
        dims=["run_stats_axis"],
        coords={"run_stats_axis": list(stats.attrs.keys())},
        name="run_stats",
    )
    ds = xr.merge([ds, pism_config, run_stats])

    print(f"Size in memory: {(ds.nbytes / 1024**3):.1f} GB")

    basins_file = result_dir / f"basins_sums_ensemble_{ensemble}_id_{m_id}.nc"

    cluster = LocalCluster(n_workers=options.n_jobs, threads_per_worker=1)
    client = Client(cluster, asynchronous=True)
    print(f"Open client in browser: {client.dashboard_link}")

    start = time.time()

    basins_ds_scattered = client.scatter(
        [ds.rio.clip([basin.geometry]) for _, basin in basins.iterrows()]
    )
    basin_names = [basin["SUBREGION1"] for _, basin in basins.iterrows()]
    futures = client.map(compute_basin, basins_ds_scattered, basin_names)
    progress(futures)
    basin_sums = xr.concat(client.gather(futures), dim="basin")
    if "time_bounds" in ds.data_vars:
        basin_sums["time_bounds"] = ds["time_bounds"]
    basin_sums.to_netcdf(basins_file)

    end = time.time()
    time_elapsed = end - start
    print(f"Time elapsed {time_elapsed:.0f}s")
