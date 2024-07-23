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

import time
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from pathlib import Path

import dask
import geopandas as gp
import xarray as xr
from dask.distributed import Client, LocalCluster, progress


def compute_basin(ds: xr.Dataset, name: str) -> xr.Dataset:
    """
    Compute the sum of the dataset over the 'x' and 'y' dimensions and add a new dimension 'basin'.

    Parameters
    ----------
    ds : xr.Dataset
        The input dataset.
    name : str
        The name to assign to the new 'basin' dimension.

    Returns
    -------
    xr.Dataset
        The computed dataset with the new 'basin' dimension.

    Examples
    --------
    >>> ds = xr.Dataset({'var': (('x', 'y'), np.random.rand(5, 5))})
    >>> compute_basin(ds, 'new_basin')
    """
    ds = ds.sum(dim=["x", "y"]).expand_dims("basin")
    ds["basin"] = [name]
    return ds.compute()


if __name__ == "__main__":
    # set up the option parser
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.description = "Compute ensemble statistics."
    parser.add_argument(
        "--ensemble_dir",
        help="""Base directory of ensemble.""",
        type=str,
        default="/mnt/storstrommen/ragis/data/pism",
    )
    parser.add_argument(
        "--ensemble_id",
        help="""Name of the ensemble. Default=RAGIS.""",
        type=str,
        default="MAR",
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
        default="data/basins/Greenland_Basins_PS_v1.4.2.shp",
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

    x_dim, y_dim = "x", "y"
    ensemble_dir = Path(options.ensemble_dir)
    assert ensemble_dir.exists()
    ensemble_id = options.ensemble_id
    result_dir = Path(options.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    fig_dir = result_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    # Load basins, merge all ICE_CAP geometries
    basin_url = Path(options.basin_url)
    basins = gp.read_file(basin_url).to_crs(crs)
    basins.to_file("b_rot.gpkg")

    with dask.config.set(**{"array.slicing.split_large_chunks": True}):
        ds = xr.open_dataset(
            options.FILE[-1],
            chunks="auto",
        ).drop_vars(["time_bnds"], errors="ignore")

        ds = ds.expand_dims({"ensemble_id": [ensemble_id]})

        if options.temporal_range:
            ds = ds.sel(time=slice(options.temporal_range))

    ds.rio.set_spatial_dims(x_dim=x_dim, y_dim=y_dim, inplace=True)
    ds.rio.write_crs(crs, inplace=True)

    print(f"Size in memory: {(ds.nbytes / 1024**3):.1f} GB")

    basins_file = result_dir / f"basins_sums_ensemble_{ensemble_id}.nc"

    cluster = LocalCluster(n_workers=options.n_jobs, threads_per_worker=1)
    with Client(cluster, asynchronous=True) as client:
        print(f"Open client in browser: {client.dashboard_link}")

        start = time.time()

        basins_ds_scattered = client.scatter(
            [ds.rio.clip([basin.geometry]) for _, basin in basins.iterrows()]
        )
        basin_names = [basin["SUBREGION1"] for _, basin in basins.iterrows()]
        futures = client.map(compute_basin, basins_ds_scattered, basin_names)
        progress(futures)
        basin_sums = xr.concat(client.gather(futures), dim="basin")
        gris_sums = basin_sums.sum(dim="basin").expand_dims("basin")
        gris_sums["basin"] = ["GIS"]
        basin_sums = xr.concat([basin_sums, gris_sums], dim="basin")
        basin_sums.to_netcdf(basins_file)

        end = time.time()
        time_elapsed = end - start
        print(f"Time elapsed {time_elapsed:.0f}s")

        client.close()
