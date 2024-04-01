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

from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import pylab as plt
import xarray as xr
from pathlib import Path
import geopandas as gp
from matplotlib import cm, colors
from matplotlib.colors import LightSource

import re
import rioxarray as rxr
import geopandas as gp
import time
import pandas as pd
import cartopy.crs as ccrs
import dask
from dask.distributed import Client, progress, LocalCluster
from dask.diagnostics import ProgressBar
from typing import Union

from joblib import Parallel, delayed
from tqdm.auto import tqdm

import toml

from pism_ragis.processing import preprocess_nc
from pism_ragis.observations import load_mouginot
from pypism.utils import blend_multiply, qgis2cmap, tqdm_joblib

"""
Analyze RAGIS ensemble

"""
kg2cmsle = 1 / 1e12 * 1.0 / 362.5 / 10.0
gt2cmsle = 1 / 362.5 / 10.0




def process_file(url, chunks=None):
    ds = xr.open_dataset(url, chunks=chunks)
    if options.temporal_range:
        ds = ds.sel(time=slice(*options.temporal_range))
    m_id_re = re.search("id_(.+?)_", ds.encoding["source"])
    ds = ds.expand_dims("exp_id")
    assert m_id_re is not None
    exp_id: Union[str, int]
    try:
        exp_id = int(m_id_re.group(1))
    except:
        exp_id = str(m_id_re.group(1))
    ds["exp_id"] = [exp_id]
    if "ice_mass" in ds:
        ds["ice_mass"] /= 1e12
        ds["ice_mass"].attrs["units"] = "Gt"
    ds = ds.expand_dims("ensemble_id")
    ds["ensemble_id"] = [ensemble_id]
    ds["tendency_of_ice_mass_due_to_grounding_line_flux"] = ds["grounding_line_flux"] * ds["thk"] * (ds["x"][1]-ds["x"][0]) / 1e12
    ds["tendency_of_ice_mass_due_to_grounding_line_flux"].attrs["units"] = "Gt year-1"
    ds = ds[mb_vars]
    ds.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=True)
    ds.rio.write_crs("epsg:3413", inplace=True)
    b_sums = []
    for k, basin in basins.iterrows():
        basin_name = [basin["SUBREGION1"]][0]
        b_sum = ds.rio.clip([basin.geometry]).sum(dim=["x", "y"])
        b_sum = b_sum.expand_dims("basin")
        b_sum["basin"] = [basin_name]
        b_sums.append(b_sum)
    b_sum = xr.concat(b_sums, dim="basin")
    with ProgressBar():
        basin_file = result_dir / f"rignot_basins_ensemble_id_{ensemble_id}_exp_id_{exp_id}_sums.nc"
        start = time.time()
        print(f"Computing basin sums and saving to {basin_file}")
        encoding = {var: comp for var in b_sum.data_vars}
        b_sum.to_netcdf(basin_file, encoding=encoding)
        end = time.time()
        time_elapsed = end - start
        print(f"-  Time elapsed {time_elapsed:.0f}s")

    with ProgressBar():
        start = time.time()
        sums_file = result_dir / f"gris_rignot_ensemble_id_{ensemble_id}_exp_id_{exp_id}_sums.nc"
        print(f"Computing ice sheet-wide sums and saving to {sums_file}")
        sums = ds.rio.clip(basins.geometry).sum(dim=["x", "y"])
        sums = sums.expand_dims("basin")
        sums["basin"] = ["GIS"]
        encoding = {var: comp for var in ds.data_vars}
        ds.to_netcdf(sums_file, encoding=encoding)
        end = time.time()
        time_elapsed = end - start
        print(f"-  Time elapsed {time_elapsed:.0f}s")

    with ProgressBar():
        start = time.time()
        sums_file = result_dir / f"domain_ensemble_id_{ensemble_id}_exp_id_{exp_id}_sums.nc"
        print(f"Computing domain-wide sums and saving to {sums_file}")
        sums = ds.sum(dim=["x", "y"])
        sums = sums.expand_dims("basin")
        sums["basin"] = ["MD"]
        encoding = {var: comp for var in ds.data_vars}
        ds.to_netcdf(sums_file, encoding=encoding)
        end = time.time()
        time_elapsed = end - start
        print(f"-  Time elapsed {time_elapsed:.0f}s")

def load_experiments(experiments, data_type: str = "spatial", engine: str = "h5netcdf", chunks: Union[None, dict] = None) -> xr.Dataset:
    """
    Load experiments
    """

    dss = []
    for exp in experiments:

        url = ensemble_dir / Path(exp["proj_dir"]) / Path(exp[f"{data_type}_dir"])
        urls = url.glob(f"""*_gris_g{exp["resolution"]}m*.nc""")
        ds = xr.open_mfdataset(urls, preprocess=preprocess_nc, concat_dim="exp_id", combine="nested", engine=engine, parallel=True, chunks=chunks)
        if "ice_mass" in ds:
            ds["ice_mass"] /= 1e12
            ds["ice_mass"].attrs["units"] = "Gt"
        ds.expand_dims("ensemble_id")
        ds["ensemble_id"] = exp["ensemble_id"]
        dss.append(ds)

    return  xr.concat(dss, dim="ensemble_id")

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
        "--mouginot_url",
        help="""Path to Mouginot 2019 excel file.""",
        type=str,
        default="/mnt/storstrommen/data/mouginot/pnas.1904242116.sd02.xlsx"
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
        default="EPSG:3413"
    )
    parser.add_argument(
        "--memory_limit",
        help="""Maximum memory used by Client. Default=16GB.""",
        type=str,
        default="16GB"
    )
    parser.add_argument(
        "--notebook",
        help="""Do not use distributed Client.""",
        action="store_true",
        default=False
    )
    parser.add_argument("--n_jobs", help="""Number of parallel jobs.""", type=int, default=4)
    parser.add_argument("PROJECTFILES", nargs="*", help="Project files in toml", default=None)

    options = parser.parse_args()
    crs = options.crs
    n_jobs = options.n_jobs

    colorblind_colors = ["#882255", "#AA4499", "#CC6677", "#DDCC77", "#88CCEE", "#44AA99", "#117733"]

    project_files = [Path(f) for f in options.PROJECTFILES]
    experiments = [toml.load(f) for f in project_files]

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

    # Load observations
    mou = load_mouginot(url=Path(options.mouginot_url), norm_year=1980)
    mou_gis = mou[mou["Basin"] == "GIS"]

    mb_vars = ["ice_mass", "tendency_of_ice_mass", "tendency_of_ice_mass_due_to_surface_mass_flux", "tendency_of_ice_mass_due_to_discharge", "tendency_of_ice_mass_due_to_grounding_line_flux"]
    
    comp = dict(zlib=True, complevel=2)

    chunks = "auto"
    with dask.config.set(**{'array.slicing.split_large_chunks': True}):
        spatial_ds = load_experiments(experiments, data_type="spatial", chunks=chunks)
    spatial_ds["tendency_of_ice_mass_due_to_grounding_line_flux"] = spatial_ds["grounding_line_flux"] * spatial_ds["thk"] * (spatial_ds["x"][1]-spatial_ds["x"][0]) / 1e12
    spatial_ds["tendency_of_ice_mass_due_to_grounding_line_flux"].attrs["units"] = "Gt year-1"
    
    spatial_ds = spatial_ds[mb_vars]
    spatial_ds.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=True)
    spatial_ds.rio.write_crs("epsg:3413", inplace=True)

    basins_file = result_dir / "basins_sums.nc"
    gris_file = result_dir / "gris_sums.nc"
    domain_file = result_dir / "domain_sums.nc"

    cluster = LocalCluster(n_workers=n_jobs, threads_per_worker=1, memory_limit=options.memory_limit)
    client = Client(cluster)
    print(f"Open client in browser: {client.dashboard_link}"

    b_sums = []
    start = time.time()
    with client:
        for k, basin in basins.iterrows():
            basin_name = basin["SUBREGION1"]
            basin_file = result_dir / f"basin_{basin_name}_sums.nc"
            b_sum = spatial_ds.rio.clip([basin.geometry]).sum(dim=["x", "y"])
            b_sum = b_sum.expand_dims("basin")
            b_sum["basin"] = [basin_name]
            b_sums.append(b_sum)
        basins_sums = xr.concat(b_sums, dim="basin").persist()
        print(f"Computing basin sums and saving to {basins_file}")
        progress(basins_sums)
        encoding = {var: comp for var in basins_sums.data_vars}
        basins_sums.to_netcdf(basins_file, encoding=encoding)

        gris_sums = spatial_ds.rio.clip(basins.geometry).sum(dim=["x", "y"]).persist()
        print(f"Computing ice sheet-wide sums and saving to {gris_file}")
        encoding = {var: comp for var in gris_sums.data_vars}
        progress(gris_sums)
        gris_sums.to_netcdf(gris_file, encoding=encoding)

        domain_sums = spatial_ds.sum(dim=["x", "y"]).persist()
        print(f"Computing domain-wide sums and saving to {domain_file}")
        encoding = {var: comp for var in domain_sums.data_vars}
        progress(domain_sums)
        progress(domain_sums.to_netcdf(domain_file, encoding=encoding))
    end = time.time()
    time_elapsed = end - start
    print(f"Time elapsed {time_elapsed:.0f}s")

