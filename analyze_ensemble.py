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
import cartopy.crs as ccrs
import cartopy

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
    mou = load_mouginot(url=Path("/mnt/storstrommen/data/mouginot/pnas.1904242116.sd02.xlsx"), norm_year=1980)
    mou_gis = mou[mou["Basin"] == "GIS"]

    mb_vars = ["ice_mass", "tendency_of_ice_mass", "tendency_of_ice_mass_due_to_surface_mass_flux", "tendency_of_ice_mass_due_to_discharge", "tendency_of_ice_mass_due_to_grounding_line_flux"]
    
    comp = dict(zlib=True, complevel=2)

    urls = result_dir.glob("rignot_b*_sums.nc")
    # urls = ['/mnt/storstrommen/ragis/2024_03_fogss/rignot_basin_CW_ensemble_id_GrIMP_exp_id_3_sums.nc', '/mnt/storstrommen/ragis/2024_03_fogss/rignot_basin_NE_ensemble_id_GrIMP_exp_id_3_sums.nc']


    basins_file = result_dir / "basins_sums.nc"
    gris_file = result_dir / "gris_sums.nc"
    domain_file = result_dir / "domain_sums.nc"
    scalar_file = result_dir / "scalar_sums.nc"

    basins_sums = xr.open_dataset(basins_file, chunks="auto")
    basins_sums["ice_mass"] = basins_sums["ice_mass"] - basins_sums.sel(time="1980-01-01", method="nearest")["ice_mass"]

    gris_sums = xr.open_dataset(gris_file, chunks="auto")
    gris_sums["ice_mass"] = gris_sums["ice_mass"] - gris_sums.sel(time="1980-01-01", method="nearest")["ice_mass"]
    
    domain_sums = xr.open_dataset(domain_file, chunks="auto")
    domain_sums["ice_mass"] = domain_sums["ice_mass"] - domain_sums.sel(time="1980-01-01", method="nearest")["ice_mass"]

    scalar_sums = xr.open_dataset(scalar_file, chunks="auto")
    scalar_sums["ice_mass"] = (scalar_sums["ice_mass"] - scalar_sums.sel(time="1980-01-01", method="nearest")["ice_mass"])
    scalar_sums["limnsw"] = (scalar_sums["limnsw"] - scalar_sums.sel(time="1980-01-01", method="nearest")["limnsw"] ) / 1e12
    
    gris = basins_sums.sum(dim="basin")
    gris.expand_dims("basin")
    gris["basin"] = ["GIS"]
    sums = xr.concat([basins_sums, gris], dim="basin")
    
    plt.rc('font', size=6)
    colorblind_colors = ["#882255", "#DDCC77", "#AA4499", "#CC6677", "#DDCC77", "#88CCEE", "#44AA99", "#117733"]

    sigma = 1
    sigma_smb = 2
    sigma_discharge = 2
    mass_varname = "Cumulative ice sheet mass change (Gt)"
    mass_uncertainty_varname = "Cumulative ice sheet mass change uncertainty (Gt)"
    discharge_varname = "Rate of ice discharge (Gt/yr)"
    discharge_uncertainty_varname = "Rate of ice discharge uncertainty (Gt/yr)"
    smb_varname = "Rate of surface mass balance (Gt/yr)"
    smb_uncertainty_varname = "Rate of surface mass balance uncertainty (Gt/yr)"

    sim_mass_varname = "ice_mass"
    sim_smb_varname = "tendency_of_ice_mass_due_to_surface_mass_flux"
    sim_discharge_varname = "tendency_of_ice_mass_due_to_discharge"

    sim_colors = colorblind_colors
    obs = mou_gis
    obs_color = "#216778"
    obs_alpha = 0.5
    sim_alpha = 0.5


    plt.rc('font', size=9)
    fig, ax = plt.subplots(nrows=1, ncols=1, sharex="col", figsize=(7, 4.2))
    #fig.subplots_adjust(wspace=0, hspace=0.075, bottom=0.125, top=0.975, left=0.175, right=0.96)

    sim_cis = []
    obs_color = "0.5"
    gris = sums.sel(basin="GIS")
    for k, (ens_id, da) in enumerate(gris.groupby("ensemble_id")):
        da = da.load()
        s_med = scalar_sums.sel(ensemble_id=ens_id).quantile(0.50, dim="exp_id")
        d_med = domain_sums.sel(ensemble_id=ens_id).quantile(0.50, dim="exp_id")
        g_med = gris_sums.sel(ensemble_id=ens_id).quantile(0.50, dim="exp_id")

        # sim_ci = axs[0].fill_between(da.time, q_low[sim_mass_varname], q_high[sim_mass_varname],
        #                     alpha=sim_alpha, color=sim_colors[k], lw=0, label=ens_id)
        #l_sim = ax.plot(da.time, q_med[sim_mass_varname], lw=1.5, color=sim_colors[0], label="Ice Sheet")
        g_sim = ax.plot(g_med.time, g_med[sim_mass_varname], lw=2.0, color=sim_colors[3], label="Ice Sheet")
        s_sim = ax.plot(s_med.time, s_med["limnsw"], lw=2.0, color=sim_colors[2], label="Modeling Domain VAF")
        d_sim = ax.plot(d_med.time, d_med[sim_mass_varname], lw=2.0, color=sim_colors[1], label="Modeling Domain")
        sim_cis.append(s_sim[0])
        sim_cis.append(d_sim[0])
        sim_cis.append(g_sim[0])
        
    obs_ci = ax.fill_between(obs["Date"], 
                        (obs[mass_varname] + sigma * obs[mass_uncertainty_varname]), 
                        (obs[mass_varname] - sigma * obs[mass_uncertainty_varname]), 
                        ls="solid", color=obs_color, lw=0, alpha=obs_alpha, label="1-$\sigma$")
    obs_ci_2 = ax.fill_between(obs["Date"], 
                        (obs[mass_varname] + 2 * sigma * obs[mass_uncertainty_varname]), 
                        (obs[mass_varname] - 2 * sigma * obs[mass_uncertainty_varname]), 
                        ls="solid", color=obs_color, lw=0, alpha=obs_alpha / 2, label="2-$\sigma$")



    legend_obs = ax.legend(handles=[obs_ci, obs_ci_2], loc="lower left",
                                   title="Observed\n(Mouginot 2019)")
    legend_obs.get_frame().set_linewidth(0.0)
    legend_obs.get_frame().set_alpha(0.0)

    legend_sim = ax.legend(handles=sim_cis, loc="upper left",
                                   title="Simulated")
    legend_sim.get_frame().set_linewidth(0.0)
    legend_sim.get_frame().set_alpha(0.0)

    ax.add_artist(legend_obs)
    ax.add_artist(legend_sim)
    #scalar_mass.sel(exp_id="BAYES-MEDIAN").plot.line(x="time", ax=ax)
    ax.axhline(0, ls="dotted", color="k", lw=0.5) 
    ax.set_xlim(pd.to_datetime("1980-1-1"), pd.to_datetime("2000-1-1"))
    ax.set_ylim(-2500, 2000)
    ax.set_ylabel("Cumulative Mass Change (Gt)")
    fig.tight_layout()
    fig.savefig(fig_dir / "ice_mass_scalar_1980-2000.pdf")
    fig.savefig(fig_dir / "ice_mass_scalar_1980-2000.png", dpi=600)

    fig, axs = plt.subplots(nrows=3, ncols=1, sharex="col", figsize=(7, 4.2), height_ratios=[16, 9, 9])
    fig.subplots_adjust(wspace=0, hspace=0.005)

    sim_cis = []

    gris = sums.sel(basin="GIS")
    for k, (ens_id, da) in enumerate(gris.groupby("ensemble_id")):
        da = da.load()
        s_med = scalar_sums.sel(ensemble_id=ens_id).quantile(0.50, dim="exp_id")
        d_med = domain_sums.sel(ensemble_id=ens_id).quantile(0.50, dim="exp_id")
        g_med = gris_sums.sel(ensemble_id=ens_id).quantile(0.50, dim="exp_id")
        q_low = da.quantile(0.16, dim="exp_id")
        q_med = da.quantile(0.50, dim="exp_id")
        q_high = da.quantile(0.84, dim="exp_id")

        # sim_ci = axs[0].fill_between(da.time, q_low[sim_mass_varname], q_high[sim_mass_varname],
        #                     alpha=sim_alpha, color=sim_colors[k], lw=0, label=ens_id)
        g_sim = axs[0].plot(g_med.time, g_med[sim_mass_varname], lw=2.0, color=sim_colors[3], label="Ice Sheet")
        d_sim = axs[0].plot(d_med.time, d_med[sim_mass_varname], lw=2.0, color=sim_colors[1], label="Modeling Domain")
        axs[1].plot(g_med.time, g_med.rolling({"time": 13}).mean()[sim_discharge_varname], lw=1.5, color=sim_colors[3], label="Ice Sheet")
        axs[1].plot(d_med.time, d_med.rolling({"time": 13}).mean()[sim_discharge_varname], lw=1.5, color=sim_colors[1], label="Modeling Domain")
        axs[2].plot(g_med.time, g_med.rolling({"time": 13}).mean()[sim_smb_varname], lw=1.5, color=sim_colors[3], label="Ice Sheet")
        axs[2].plot(d_med.time, d_med.rolling({"time": 13}).mean()[sim_smb_varname], lw=1.5, color=sim_colors[1], label="Modeling Domain")
        sim_cis.append(d_sim[0])
        sim_cis.append(g_sim[0])
        

    obs_ci = axs[0].fill_between(obs["Date"], 
                        (obs[mass_varname] + sigma * obs[mass_uncertainty_varname]), 
                        (obs[mass_varname] - sigma * obs[mass_uncertainty_varname]), 
                        ls="solid", color=obs_color, lw=0, alpha=obs_alpha, label="1-$\sigma$")
    obs_ci_2 = axs[0].fill_between(obs["Date"], 
                        (obs[mass_varname] + 2 * sigma * obs[mass_uncertainty_varname]), 
                        (obs[mass_varname] - 2 * sigma * obs[mass_uncertainty_varname]), 
                        ls="solid", color=obs_color, lw=0, alpha=obs_alpha / 2, label="2-$\sigma$")

    axs[1].fill_between(obs["Date"], 
                        (obs[discharge_varname] + sigma * obs[discharge_uncertainty_varname]), 
                        (obs[discharge_varname] - sigma * obs[discharge_uncertainty_varname]), 
                        ls="solid", color=obs_color, lw=0, alpha=obs_alpha, label="1-$\sigma$")
    axs[1].fill_between(obs["Date"], 
                        (obs[discharge_varname] + 2 * sigma * obs[discharge_uncertainty_varname]), 
                        (obs[discharge_varname] - 2* sigma * obs[discharge_uncertainty_varname]), 
                        ls="solid", color=obs_color, lw=0, alpha=obs_alpha / 2, label="2-$\sigma$")
    axs[2].fill_between(obs["Date"], 
                        (obs[smb_varname] + sigma * obs[smb_uncertainty_varname]), 
                        (obs[smb_varname] - sigma * obs[smb_uncertainty_varname]), 
                        ls="solid", color=obs_color, lw=0, alpha=obs_alpha, label="1-$\sigma$")
    axs[2].fill_between(obs["Date"], 
                        (obs[smb_varname] + 2 * sigma * obs[smb_uncertainty_varname]), 
                        (obs[smb_varname] - 2* sigma * obs[smb_uncertainty_varname]), 
                        ls="solid", color=obs_color, lw=0, alpha=obs_alpha / 2, label="2-$\sigma$")
    
    legend_obs = axs[0].legend(handles=[obs_ci], loc="upper left",
                                   title="Observed\n(Mouginot 2019)")
    legend_obs.get_frame().set_linewidth(0.0)
    legend_obs.get_frame().set_alpha(0.0)
    legend_sim = axs[0].legend(handles=sim_cis, loc="lower left",
                                   title="Simulated")
    legend_sim.get_frame().set_linewidth(0.0)
    legend_sim.get_frame().set_alpha(0.0)

    axs[0].add_artist(legend_obs)
    axs[0].add_artist(legend_sim)
    #scalar_mass.sel(exp_id="BAYES-MEDIAN").plot.line(x="time", ax=ax)
    axs[0].set_ylim(-2500, 2000)
    axs[-1].set_xlim(pd.to_datetime("1980-1-1"), pd.to_datetime("2000-1-1"))
    axs[1].set_ylim(-800, 0)
    axs[0].set_ylabel("Cumulative Mass Change\n(Gt)")
    axs[1].set_ylabel("Ice Discharge (Gt/yr)")
    axs[2].set_ylabel("SMB (Gt/yr)")
    fig.tight_layout()
    fig.savefig(fig_dir / "ice_fluxes_scalar_1980-2000.pdf")
    fig.savefig(fig_dir / "ice_fluxes_scalar_1980-2000.png", dpi=600)


    import cartopy
    from cartopy import crs as ccrs
    import numpy as np
    crs = ccrs.NorthPolarStereo(central_longitude=-45, true_scale_latitude=70, globe=None)

    fig = plt.figure(figsize=(3.2, 6.2))
    ax = fig.add_subplot(111, projection=crs)

    basins.plot(color=colorblind_colors[3],ax=ax)
    ax.coastlines(linewidth=0.25, resolution="10m")
    ax.gridlines(

            dms=True,
            xlocs=np.arange(-60, 0, 10),
            ylocs=np.arange(50, 85, 10),
            x_inline=False,
            y_inline=False,
            rotate_labels=20,
            ls="dotted",
            color="0.5",
        )
    fig.tight_layout()
    fig.savefig(fig_dir / "gris_domain_red.pdf")
    fig.savefig(fig_dir / "gris_domain_red.png", dpi=600)

    fig = plt.figure(figsize=(3.2, 6.2))
    ax = fig.add_subplot(111, projection=crs)
    ax.set_facecolor(colorblind_colors[2])
    basins.plot(color=colorblind_colors[0],ax=ax, alpha=0)
    ax.coastlines(linewidth=0.25, resolution="10m")
    ax.gridlines(

            dms=True,
            xlocs=np.arange(-60, 0, 10),
            ylocs=np.arange(50, 85, 10),
            x_inline=False,
            y_inline=False,
            rotate_labels=20,
            ls="dotted",
            color="0.5",
        )
    fig.tight_layout()
    fig.savefig(fig_dir / "modeling_domain_purple.pdf")
    fig.savefig(fig_dir / "modeling_domain_purple.png", dpi=600)
    
    fig = plt.figure(figsize=(3.2, 6.2))
    ax = fig.add_subplot(111, projection=crs)
    ax.set_facecolor(colorblind_colors[1])
    basins.plot(color=colorblind_colors[1],ax=ax, alpha=0)
    ax.coastlines(linewidth=0.25, resolution="10m")
    ax.gridlines(

            dms=True,
            xlocs=np.arange(-60, 0, 10),
            ylocs=np.arange(50, 85, 10),
            x_inline=False,
            y_inline=False,
            rotate_labels=20,
            ls="dotted",
            color="0.5",
        )
    fig.tight_layout()
    fig.savefig(fig_dir / "modeling_domain_yellow.pdf")
    fig.savefig(fig_dir / "modeling_domain_yellow.png", dpi=600)

    fig, ax = plt.subplots(nrows=1, ncols=1, sharex="col", figsize=(8, 1.6))
    #fig.subplots_adjust(wspace=0, hspace=0.075, bottom=0.125, top=0.975, left=0.175, right=0.96)

    sim_cis = []
    sim_colors = ["#216778"]
    obs_alpha = 0.5
    gris = sums.sel(basin="GIS")
    for k, (ens_id, da) in enumerate(gris.groupby("ensemble_id")):
        da = da.load()
        q_low = da.quantile(0.16, dim="exp_id")
        q_med = da.quantile(0.50, dim="exp_id")
        q_high = da.quantile(0.84, dim="exp_id")

        # sim_ci = axs[0].fill_between(da.time, q_low[sim_mass_varname], q_high[sim_mass_varname],
        #                     alpha=sim_alpha, color=sim_colors[k], lw=0, label=ens_id)
        l_sim = ax.plot(da.time, q_med[sim_mass_varname], lw=2.0, color=sim_colors[k])
        sim_cis.append(l_sim[0])
        

    obs_ci = ax.fill_between(obs["Date"], 
                        (obs[mass_varname] + sigma * obs[mass_uncertainty_varname]), 
                        (obs[mass_varname] - sigma * obs[mass_uncertainty_varname]), 
                        ls="solid", color=obs_color, lw=0, alpha=obs_alpha, label="1-$\sigma$")


    ax.set_xlim(pd.to_datetime("1980-1-1"), pd.to_datetime("2000-1-1"))
    ax.set_ylim(-2000, 500)
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(fig_dir / "ice_mass_scalar_plain_1980-2000.pdf")
    fig.savefig(fig_dir / "ice_mass_scalar_plain_1980-2000.png", dpi=600)


    fig, axs = plt.subplots(nrows=3, ncols=1, sharex="col", figsize=(6.2, 2.1))
    fig.subplots_adjust(wspace=0, hspace=0.075, bottom=0.125, top=0.975, left=0.175, right=0.96)
    gris = sums.sel(basin="GIS")
    for k, (ens_id, da) in enumerate(gris.groupby("ensemble_id")):
        da = da.load()
        q_low = da.quantile(0.16, dim="exp_id")
        q_med = da.quantile(0.50, dim="exp_id")
        q_high = da.quantile(0.84, dim="exp_id")

        # sim_ci = axs[0].fill_between(da.time, q_low[sim_mass_varname], q_high[sim_mass_varname],
        #                     alpha=sim_alpha, color=sim_colors[k], lw=0, label=ens_id)
        axs[0].plot(da.time, q_med[sim_mass_varname], lw=0.75, color=sim_colors[k])
        
        # axs[1].fill_between(q_low["time"], q_low.rolling({"time": 13}).mean()[sim_discharge_varname], q_high.rolling({"time": 13}).mean()[sim_discharge_varname],
        #                     alpha=sim_alpha, color=sim_colors[k], lw=0, label=ens_id)
        axs[1].plot(q_med["time"], q_med.rolling({"time": 13}).mean()[sim_discharge_varname], lw=0.75, color=sim_colors[k])
        #axs[1].plot(q_med["time"], q_med.rolling({"time": 13}).mean()["tendency_of_ice_mass_due_to_grounding_line_flux"], lw=0.75, color=sim_colors[k])
        # axs[2].fill_between(da.time, q_low.rolling({"time": 13}).mean()[sim_smb_varname], q_high.rolling({"time": 13}).mean()[sim_smb_varname],
        #                     alpha=sim_alpha, color=sim_colors[k], lw=0, label=ens_id)
        axs[2].plot(da.time, q_med.rolling({"time": 13}).mean()[sim_smb_varname], lw=0.75, color=sim_colors[k])
        #sim_cis.append(sim_ci)

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
    # legend_sim = axs[0].legend(handles=sim_cis, loc="lower center",
    #                                title="Simulated")
    # legend_sim.get_frame().set_linewidth(0.0)
    # legend_sim.get_frame().set_alpha(0.0)

    axs[0].add_artist(legend_obs)
    #axs[0].add_artist(legend_sim)
    #scalar_mass.sel(exp_id="BAYES-MEDIAN").plot.line(x="time", ax=ax)
    axs[-1].set_xlim(pd.to_datetime("1980-1-1"), pd.to_datetime("2000-1-1"))
    axs[1].set_ylim(-750, 0)
    axs[0].set_ylim(-2000, 500)
    fig.savefig(fig_dir / "ragis-comp-3_scalar_1980-2000.pdf")
    fig.savefig(fig_dir / "ragis-comp-3_scalar_1980-2000.png", dpi=600)

