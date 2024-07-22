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
from typing import Any, Dict, List, Union

import dask
import geopandas as gp
import toml
import xarray as xr
from dask.distributed import Client, progress
from dask_mpi import initialize

from pism_ragis.processing import preprocess_nc


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


def load_experiments(
    experiments: List[Dict[str, Any]],
    data_type: str = "spatial",
    engine: str = "h5netcdf",
    chunks: Union[None, dict, str] = None,
    temporal_range: Union[None, List[str]] = None,
) -> xr.Dataset:
    """
    Load experiments from multiple netCDF files and concatenate them into a single xarray Dataset.

    Parameters
    ----------
    experiments : list of dict
        A list of dictionaries, each containing the details of an experiment.
    data_type : str, optional
        The type of data to load, by default "spatial".
    engine : str, optional
        The engine to use for reading the files, by default "h5netcdf".
    chunks : None, dict, or str, optional
        Chunking sizes to use for dask, by default None.
    temporal_range : None or list of str, optional
        The temporal range to select from the data, by default None.

    Returns
    -------
    xr.Dataset
        The concatenated dataset.

    Examples
    --------
    >>> experiments = [{'proj_dir': 'proj1', 'spatial_dir': 'spatial1', 'resolution': 'res1', 'ensemble_id': 'ens1'}]
    >>> load_experiments(experiments)
    """

    dss = []
    for exp in experiments:
        url = ensemble_dir / Path(exp["proj_dir"]) / Path(exp[f"{data_type}_dir"])
        urls = url.glob(f"""*_gris_g{exp["resolution"]}m*.nc""")
        ds = xr.open_mfdataset(
            urls,
            preprocess=preprocess_nc,
            concat_dim="exp_id",
            combine="nested",
            engine=engine,
            parallel=True,
            chunks=chunks,
        )
        if "ice_mass" in ds:
            ds["ice_mass"] /= 1e12
            ds["ice_mass"].attrs["units"] = "Gt"

        ds.expand_dims("ensemble_id")
        ds["ensemble_id"] = exp["ensemble_id"]
        dss.append(ds)

    ds = xr.concat(dss, dim="ensemble_id")
    if temporal_range:
        return ds.sel(time=slice(*options.temporal_range))

    else:
        return ds


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
        "--notebook",
        help="""Do not use distributed Client.""",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--n_jobs", help="""Number of parallel jobs.""", type=int, default=4
    )
    parser.add_argument(
        "PROJECTFILES", nargs="*", help="Project files in toml", default=None
    )

    options = parser.parse_args()
    crs = options.crs
    n_jobs = options.n_jobs

    project_files = [Path(f) for f in options.PROJECTFILES]
    project_experiments = [toml.load(f) for f in project_files]

    ensemble_dir = Path(options.ensemble_dir)
    assert ensemble_dir.exists()
    result_dir = Path(options.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    fig_dir = result_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    # Load basins, merge all ICE_CAP geometries
    basin_url = Path(options.basin_url)
    basins = gp.read_file(basin_url).to_crs(crs)
    # if "SUBREGION1" in basins:
    #     ice_sheet = basins[basins["SUBREGION1"] != "ICE_CAP"]
    #     ice_caps = basins[basins["SUBREGION1"] == "ICE_CAP"].unary_union
    #     ice_caps = gp.GeoDataFrame(pd.DataFrame(data=["ICE_CAP"], columns=["SUBREGION1"]), geometry=[ice_caps], crs=basins.crs)
    #     basins = pd.concat([ice_sheet, ice_caps]).reset_index(drop=True)

    mb_vars = [
        "ice_mass",
        "ice_mass_transport_across_grounding_line",
        "tendency_of_ice_mass",
        "tendency_of_ice_mass_due_to_basal_mass_flux",
        "tendency_of_ice_mass_due_to_basal_mass_flux_grounded",
        "tendency_of_ice_mass_due_to_basal_mass_flux_floating",
        "tendency_of_ice_mass_due_to_discharge",
        "tendency_of_ice_mass_due_to_surface_mass_flux",
        "tendency_of_ice_mass_due_to_conservation_error",
        "tendency_of_ice_mass_due_to_flow",
    ]

    project_chunks = "auto"
    with dask.config.set(**{"array.slicing.split_large_chunks": True}):
        ds = load_experiments(
            project_experiments,
            data_type="spatial",
            chunks=project_chunks,
            temporal_range=options.temporal_range,
        )

    bmb_var = "tendency_of_ice_mass_due_to_basal_mass_flux"
    if bmb_var in ds:
        bmb_grounded_da = ds[bmb_var].where(ds["mask"] == 2)
        bmb_grounded_da.name = "tendency_of_ice_mass_due_to_basal_mass_flux_grounded"
        bmb_floating_da = ds[bmb_var].where(ds["mask"] == 3)
        bmb_floating_da.name = "tendency_of_ice_mass_due_to_basal_mass_flux_floating"
        ds = xr.merge([ds, bmb_grounded_da, bmb_floating_da])

    ds = ds[mb_vars]
    ds.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=True)
    ds.rio.write_crs(crs, inplace=True)

    print(f"Size in memory: {(ds.nbytes / 1024**3):.1f} GB")

    basins_file = result_dir / "basins_sums.nc"

    initialize(nthreads=options.n_jobs)
    with Client() as client:
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
