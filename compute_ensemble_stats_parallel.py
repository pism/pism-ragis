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
    parser.add_argument(
        "--compute",
        help="""Compute. Default=False.""",
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


    chunks = {"x": -1, "y": -1, "time": -1}
    chunks = "auto"

    for exp in experiments:
        ensemble_id =  exp["ensemble_id"]
        data_type="spatial"
        url = ensemble_dir / Path(exp["proj_dir"]) / Path(exp[f"{data_type}_dir"])
        urls = url.glob(f"""*_gris_g{exp["resolution"]}m*.nc""")
        result = Parallel(n_jobs=options.n_jobs)(
                delayed(process_file)(url, chunks="auto") for url in list(urls)
            )
    # basins_file = result_dir / "basins_sums.nc"
    # gris_file = result_dir / "gris_sums.nc"
    # domain_file = result_dir / "domain_sums.nc"
   
    # if options.compute:
    #     print("Loading ensembles")
    #     start = time.time()
    #     chunks = {"x": 4, "y": 4, "time": 24}
    #     chunks = "auto"
    #     spatial_ds = load_experiments(experiments, data_type="spatial", engine="netcdf4", chunks=chunks)
    #     spatial_ds["tendency_of_ice_mass_due_to_grounding_line_flux"] = spatial_ds["grounding_line_flux"] * spatial_ds["thk"] * (spatial_ds["x"][1]-spatial_ds["x"][0]) / 1e12
    #     spatial_ds["tendency_of_ice_mass_due_to_grounding_line_flux"].attrs["units"] = "Gt year-1"
    #     if options.temporal_range:
    #         spatial_ds = spatial_ds.sel(time=slice(*options.temporal_range))
    #     spatial_ds.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=True)
    #     spatial_ds.rio.write_crs("epsg:3413", inplace=True)

    #     # scalar_ds = load_experiments(experiments, data_type="scalar", engine="netcdf4")
    #     # scalar_sums = scalar_ds[mb_vars]
    #     end = time.time()
    #     time_elapsed = end - start
    #     print(f"-  Time elapsed {time_elapsed:.0f}s")

    #     start = time.time()
    #     b_sums = []
    #     if options.notebook:
    #         for k, basin in basins.iterrows():
    #             b_sum = spatial_ds[mb_vars].rio.clip([basin.geometry]).sum(dim=["x", "y"])
    #             b_sum = b_sum.expand_dims("basin")
    #             b_sum["basin"] = [basin["SUBREGION1"]]
    #             b_sums.append(b_sum)
    #         basins_sums = xr.concat(b_sums, dim="basin")
    #         with ProgressBar():
    #             print(f"Computing basin sums and saving to {basins_file}")
    #             encoding = {var: comp for var in basins_sums.data_vars}
    #             basins_sums.to_netcdf(basins_file, encoding=encoding)

    #         gris_sums = spatial_ds[mb_vars].rio.clip(basins.geometry).sum(dim=["x", "y"])
    #         with ProgressBar():
    #             print(f"Computing ice sheet-wide sums and saving to {gris_file}")
    #             encoding = {var: comp for var in gris_sums.data_vars}
    #             gris_sums.to_netcdf(gris_file, encoding=encoding)
    #         domain_sums = spatial_ds[mb_vars].sum(dim=["x", "y"])
    #         with ProgressBar():
    #             print(f"Computing domain-wide sums and saving to {domain_file}")
    #             encoding = {var: comp for var in domain_sums.data_vars}
    #             domain_sums.to_netcdf(domain_file, encoding=encoding)
    #     else:
    #         cluster = LocalCluster(n_workers=n_jobs, threads_per_worker=1, memory_limit=options.memory_limit)
    #         client = Client(cluster)
    #         with client:
    #             for k, basin in basins.iterrows():
    #                 basin_name = basin["SUBREGION1"]
    #                 basin_file = result_dir / f"basin_{basin_name}_sums.nc"
    #                 b_sum = spatial_ds[mb_vars].rio.clip([basin.geometry]).sum(dim=["x", "y"]).persist()
    #                 b_sum = b_sum.expand_dims("basin")
    #                 b_sum["basin"] = [basin_name]
    #                 b_sums.append(b_sum)
    #             basins_sums = xr.concat(b_sums, dim="basin").persist()
    #             print(f"Computing basin sums and saving to {basins_file}")
    #             progress(basins_sums)
    #             encoding = {var: comp for var in basins_sums.data_vars}
    #             basins_sums.to_netcdf(basins_file, encoding=encoding)
            
    #             gris_sums = spatial_ds[mb_vars].rio.clip(basins.geometry).sum(dim=["x", "y"]).persist()
    #             print(f"Computing ice sheet-wide sums and saving to {gris_file}")
    #             encoding = {var: comp for var in gris_sums.data_vars}
    #             progress(gris_sums)
    #             gris_sums.to_netcdf(gris_file, encoding=encoding)

    #             domain_sums = spatial_ds[mb_vars].sum(dim=["x", "y"]).persist()
    #             print(f"Computing domain-wide sums and saving to {domain_file}")
    #             encoding = {var: comp for var in domain_sums.data_vars}
    #             progress(domain_sums)
    #             progress(domain_sums.to_netcdf(domain_file, encoding=encoding))
    #     end = time.time()
    #     time_elapsed = end - start
    #     print(f"Time elapsed {time_elapsed:.0f}s")

    #     del gris_sums
    #     del domain_sums

    # basins_sums = xr.open_dataset(basins_file)
    # gris_sums = xr.open_dataset(gris_file)
    # domain_sums = xr.open_dataset(domain_file)

    
    # gris_sums["ice_mass"] = gris_sums["ice_mass"] - gris_sums.sel(time="1980-01-01", method="nearest")["ice_mass"]
    # basins_sums["ice_mass"] = basins_sums["ice_mass"] - basins_sums.sel(time="1980-01-01", method="nearest")["ice_mass"]
    # domain_sums["ice_mass"]  = domain_sums["ice_mass"] - domain_sums.sel(time="1980-01-01", method="nearest")["ice_mass"]

    # plt.rc('font', size=6)
    # colorblind_colors = ["#882255", "#DDCC77", "#AA4499", "#CC6677", "#DDCC77", "#88CCEE", "#44AA99", "#117733"]

    # sigma = 1
    # sigma_smb = 2
    # sigma_discharge = 2
    # mass_varname = "Cumulative ice sheet mass change (Gt)"
    # mass_uncertainty_varname = "Cumulative ice sheet mass change uncertainty (Gt)"
    # discharge_varname = "Rate of ice discharge (Gt/yr)"
    # discharge_uncertainty_varname = "Rate of ice discharge uncertainty (Gt/yr)"
    # smb_varname = "Rate of surface mass balance (Gt/yr)"
    # smb_uncertainty_varname = "Rate of surface mass balance uncertainty (Gt/yr)"

    # sim_mass_varname = "ice_mass"
    # sim_smb_varname = "tendency_of_ice_mass_due_to_surface_mass_flux"
    # sim_discharge_varname = "tendency_of_ice_mass_due_to_discharge"

    # sim_colors = colorblind_colors
    # obs = mou_gis
    # obs_color = ".5"
    # obs_alpha = 0.5
    # sim_alpha = 0.5

    # fig, axs = plt.subplots(nrows=3, ncols=1, sharex="col", figsize=(6.2, 6.2), height_ratios=[16, 9, 9])
    # fig.subplots_adjust(wspace=0, hspace=0.075, bottom=0.125, top=0.975, left=0.175, right=0.96)

    # sim_cis = []
    # for k, (ens_id, da) in enumerate(gris_sums.groupby("ensemble_id")):

    #     q_low = da.quantile(0.16, dim="exp_id")
    #     q_med = da.quantile(0.50, dim="exp_id")
    #     q_high = da.quantile(0.84, dim="exp_id")

    #     sim_ci = axs[0].fill_between(da.time, q_low[sim_mass_varname], q_high[sim_mass_varname],
    #                         alpha=sim_alpha, color=sim_colors[k], lw=0, label=ens_id)
    #     axs[0].plot(da.time, q_med[sim_mass_varname], lw=0.75, color=sim_colors[k])
        
    #     axs[1].fill_between(q_low["time"], q_low.rolling({"time": 13}).mean()[sim_discharge_varname], q_high.rolling({"time": 13}).mean()[sim_discharge_varname],
    #                         alpha=sim_alpha, color=sim_colors[k], lw=0, label=ens_id)
    #     axs[1].plot(q_med["time"], q_med.rolling({"time": 13}).mean()[sim_discharge_varname], lw=0.75, color=sim_colors[k])
    #     axs[2].fill_between(da.time, q_low.rolling({"time": 13}).mean()[sim_smb_varname], q_high.rolling({"time": 13}).mean()[sim_smb_varname],
    #                         alpha=sim_alpha, color=sim_colors[k], lw=0, label=ens_id)
    #     axs[2].plot(da.time, q_med.rolling({"time": 13}).mean()[sim_smb_varname], lw=0.75, color=sim_colors[k])
    #     sim_cis.append(sim_ci)

    # #scalar_limnsw_scaled.sel(exp_id="BAYES-MEDIAN").plot.line(x="time", hue="ensemble_id", ax=ax, label="limnsw")
    # #scalar_mass_scaled.sel(exp_id="BAYES-MEDIAN").plot.line(x="time", ax=ax, hue="ensemble_id", label="scalar_mass")
    # obs_ci = axs[0].fill_between(obs["Date"], 
    #                     (obs[mass_varname] + sigma * obs[mass_uncertainty_varname]), 
    #                     (obs[mass_varname] - sigma * obs[mass_uncertainty_varname]), 
    #                     ls="solid", color=obs_color, lw=0, alpha=obs_alpha, label="1-$\sigma$")

    # axs[1].fill_between(obs["Date"], 
    #                     (obs[discharge_varname] + sigma_discharge * obs[discharge_uncertainty_varname]), 
    #                     (obs[discharge_varname] - sigma_discharge * obs[discharge_uncertainty_varname]), 
    #                     ls="solid", color=obs_color, lw=0, alpha=obs_alpha)
    # axs[2].fill_between(obs["Date"], 
    #                     (obs[smb_varname] + sigma_smb * obs[smb_uncertainty_varname]), 
    #                     (obs[smb_varname] - sigma_smb * obs[smb_uncertainty_varname]), 
    #                     ls="solid", color=obs_color, lw=0, alpha=obs_alpha)

    # legend_obs = axs[0].legend(handles=[obs_ci], loc="lower left",
    #                                title="Observed")
    # legend_obs.get_frame().set_linewidth(0.0)
    # legend_obs.get_frame().set_alpha(0.0)
    # legend_sim = axs[0].legend(handles=sim_cis, loc="lower center",
    #                                title="Simulated")
    # legend_sim.get_frame().set_linewidth(0.0)
    # legend_sim.get_frame().set_alpha(0.0)

    # axs[0].add_artist(legend_obs)
    # axs[0].add_artist(legend_sim)
    # #scalar_mass.sel(exp_id="BAYES-MEDIAN").plot.line(x="time", ax=ax)
    # axs[-1].set_xlim(pd.to_datetime("1980-1-1"), pd.to_datetime("2000-1-1"))
    # #axs[1].set_ylim(-750, 0)
    # fig.savefig(fig_dir / "ragis-comp-3_scalar_1980-2000.pdf")
    # fig.savefig(fig_dir / "ragis-comp-3_scalar_1980-2000.png", dpi=600)

