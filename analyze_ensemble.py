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

import rioxarray as rxr
import geopandas as gp
import time
import pandas as pd
import cartopy.crs as ccrs
from dask.distributed import Client, progress

import toml

from pism_ragis.processing import preprocess_nc
from pism_ragis.observations import load_mouginot
from pypism.utils import blend_multiply, qgis2cmap

"""
Analyze RAGIS ensemble

"""
kg2cmsle = 1 / 1e12 * 1.0 / 362.5 / 10.0
gt2cmsle = 1 / 362.5 / 10.0


data_dir = Path("/mnt/storstrommen/ragis/data/pism")
obs_dir = Path("/mnt/storstrommen/data/")
assert data_dir.exists()

results_dir = "2024_03_analysis"
o_dir = data_dir / results_dir
o_dir.mkdir(exist_ok=True)
fig_dir = o_dir / "figures"
fig_dir.mkdir(exist_ok=True)

def load_experiments(experiments, data_type: str = "spatial") -> xr.Dataset:
    """
    Load experiments
    """

    dss = []
    for exp in experiments:

        url = data_dir / Path(exp["proj_dir"]) / Path(exp[f"{data_type}_dir"])
        urls = url.glob(f"""*_gris_g{exp["resolution"]}m*.nc""")
        ds = xr.open_mfdataset(urls, preprocess=preprocess_nc, concat_dim="exp_id", combine="nested", parallel=True)
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
    parser.description = "Compute pathlines (forward/backward) given a velocity field (xr.Dataset) and starting points (geopandas.GeoDataFrame)."
    parser.add_argument(
        "--ensemble_dir",
        help="""Base directory of ensemble.""",
        type=str,
        default="/mnt/storstrommen/ragis/data/pism",
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
        default="EPSG:3413"
    )
    parser.add_argument("--n_jobs", help="""Number of parallel jobs.""", type=int, default=4)
    parser.add_argument("PROJECTFILES", nargs="*", help="Project files in toml", default=None)

    options = parser.parse_args()
    crs = options.crs
    n_jobs = options.n_jobs

    project_files = [Path(f) for f in options.PROJECTFILES]
    experiments = [toml.load(f) for f in project_files]

    ensemble_dir = Path(options.ensemble_dir)
    assert ensemble_dir.exists()

    spatial_ds = load_experiments(experiments, data_type="spatial").sel(time=slice(*options.temporal_range))
    spatial_ds.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=True)
    spatial_ds.rio.write_crs("epsg:3413", inplace=True)

    scalar_ds = load_experiments(experiments, data_type="scalar")

    # Load basins
    basin_url = Path(options.basin_url)
    basins = gp.read_file(basin_url).to_crs(crs)
    # Load observations
    mou = load_mouginot(url=Path("/mnt/storstrommen/data/mouginot/pnas.1904242116.sd02.xlsx"), norm_year=1980)
    mou_gis = mou[mou["Basin"] == "GIS"]

    mb_vars = ["ice_mass", "tendency_of_ice_mass", "tendency_of_ice_mass_due_to_surface_mass_flux", "tendency_of_ice_mass_due_to_discharge"]
    

    # domain_mass = spatial_ds["ice_mass"].sum(dim=["x", "y"]).compute()

    from dask.diagnostics import ProgressBar
    start = time.process_time()
    b_sums = []
    for k, basin in basins.iterrows():
        b_sum = spatial_ds[mb_vars].rio.clip([basin.geometry]).sum(dim=["x", "y"])
        b_sum = b_sum.expand_dims("basin")
        b_sum["basin"] = [basin["SUBREGION1"]]
        with ProgressBar():
            b_sum.compute()
        b_sums.append(b_sum)
    basins_sums = xr.concat(b_sums, dim="basin")

    gris_sums = spatial_ds[mb_vars].rio.clip(basins.geometry).sum(dim=["x", "y"])
    with ProgressBar():
        gris_sums.compute()
    domain_sums = spatial_ds[mb_vars].sum(dim=["x", "y"])
    with ProgressBar():
        domain_sums.compute()
    time_elapsed = time.process_time() - start
    print(f"Time elapsed {time_elapsed:.0f}s")

    colorblind_colors = ["#882255", "#AA4499", "#CC6677", "#DDCC77", "#88CCEE", "#44AA99", "#117733"]

    # start = time.process_time()
    # client = Client(n_workers=n_jobs, threads_per_worker=1, memory_limit='16GB')
    # with client:
    #     # domain_mass = spatial_ds["ice_mass"].sum(dim=["x", "y"]).compute()
    #     b_sums = []
    #     for k, basin in basins.iterrows():
    #         b_sum = spatial_ds[mb_vars].rio.clip([basin.geometry]).sum(dim=["x", "y"]).persist()
    #         progress(b_sum)
    #         b_sum = b_sum.expand_dims("basin")
    #         b_sum["basin"] = [basin["SUBREGION1"]]
    #         b_sums.append(b_sum)
  
    #     gris_sums = spatial_ds[mb_vars].rio.clip(basins.geometry).sum(dim=["x", "y"]).persist()
    #     progress(gris_sums)
    #     domain_sums = spatial_ds[mb_vars].persist()
    #     progress(domain_sums)

    # basins_sums = xr.concat(b_sums, dim="basin")
    # time_elapsed = time.process_time() - start
    # print(f"Time elapsed {time_elapsed:.0f}s")

    print(basins_sums)
    gris_sums_normalized = gris_sums - gris_sums.sel(time="1980")
    basin_sums_normalized  = basins_sums - basins_sums.sel(time="1980")
    print(basin_sums_normalized)

    # plt.rc('font', size=6)
    # plt.style.use("tableau-colorblind10")

    # sim_colors = plt.rcParams['axes.prop_cycle'].by_key()['color'][1::]
    # imbie_color = plt.rcParams['axes.prop_cycle'].by_key()['color'][0]
    # sim_colors = ["#882255", "#AA4499", "#CC6677", "#DDCC77", "#88CCEE", "#44AA99", "#117733", "#332288"]
    # imbie_color = "k"
    # imbie_color = "#238b45"
    # mou_color = "#6a51a3"

    # obs = mou_gis
    # obs_color = mou_color

    sigma = 1
    sigma_smb = 1
    sigma_discharge = 1
    obs_alpha = 0.3
    mass_varname = "Cumulative ice sheet mass change (Gt)"
    mass_uncertainty_varname = "Cumulative ice sheet mass change uncertainty (Gt)"
    discharge_varname = "Rate of ice discharge (Gt/yr)"
    discharge_uncertainty_varname = "Rate of ice discharge uncertainty (Gt/yr)"
    smb_varname = "Rate of surface mass balance (Gt/yr)"
    smb_uncertainty_varname = "Rate of surface mass balance uncertainty (Gt/yr)"

    fig, axs = plt.subplots(nrows=3, ncols=1, sharex="col", figsize=(3.6, 3.8), height_ratios=[16, 9, 9])
    fig.subplots_adjust(wspace=0, hspace=0.075, bottom=0.125, top=0.975, left=0.175, right=0.96)

    sim_cis = []
    for k, (exp_id, da) in enumerate(basin_mass_scaled.groupby("ensemble_id")):
        sim_ci = axs[0].fill_between(da.time, da.quantile(0.16, dim="exp_id"), da.quantile(0.84, dim="exp_id"), 
                            alpha=0.10, color=sim_colors[k], lw=0, label=exp_id)
        axs[0].plot(da.time, da.quantile(0.50, dim="exp_id"), lw=0.75, color=sim_colors[k])
        sim_cis.append(sim_ci)

    #scalar_limnsw_scaled.sel(exp_id="BAYES-MEDIAN").plot.line(x="time", hue="ensemble_id", ax=ax, label="limnsw")
    #scalar_mass_scaled.sel(exp_id="BAYES-MEDIAN").plot.line(x="time", ax=ax, hue="ensemble_id", label="scalar_mass")
    obs_ci = axs[0].fill_between(obs["Date"], 
                        (obs[mass_varname] + sigma * obs[mass_uncertainty_varname]), 
                        (obs[mass_varname] - sigma * obs[mass_uncertainty_varname]), 
                        ls="solid", color=obs_color, lw=0, alpha=obs_alpha, label="1-$\sigma$")

    axs[1].fill_between(obs["Date"], 
                        (obs[discharge_varname] + sigma_discharge * obs[discharge_uncertainty_varname]), 
                        (obs[discharge_varname] - sigma_discharge * obs[discharge_uncertainty_varname]), 
                        ls="solid", color=obs_color, lw=0, alpha=obs_alpha)
    axs[2].fill_between(obs["Date"], 
                        (obs[smb_varname] + sigma_smb * obs[smb_uncertainty_varname]), 
                        (obs[smb_varname] - sigma_smb * obs[smb_uncertainty_varname]), 
                        ls="solid", color=obs_color, lw=0, alpha=obs_alpha)

    legend_obs = axs[0].legend(handles=[obs_ci], loc="lower left",
                                   title="Observed")
    legend_obs.get_frame().set_linewidth(0.0)
    legend_obs.get_frame().set_alpha(0.0)
    legend_sim = axs[0].legend(handles=sim_cis, loc="lower center",
                                   title="Simulated")
    legend_sim.get_frame().set_linewidth(0.0)
    legend_sim.get_frame().set_alpha(0.0)

    axs[0].add_artist(legend_obs)
    axs[0].add_artist(legend_sim)
    #scalar_mass.sel(exp_id="BAYES-MEDIAN").plot.line(x="time", ax=ax)
    #axs[-1].set_xlim(pd.to_datetime("1980-1-1"), pd.to_datetime("2000-1-1"))
    #axs[-1].set_ylim(-3000, 2000)
    fig.savefig(fig_dir / "ragis-comp-3_scalar_1980-2000.pdf")
    fig.savefig(fig_dir / "ragis-comp-3_scalar_1980-2000.png", dpi=600)
