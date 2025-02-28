# Copyright (C) 2024-25 Andy Aschwanden, Constantine Khroulev
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

# pylint: disable=unused-import,too-many-positional-arguments
"""
Analyze RAGIS ensemble.
"""
import json
import time
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from importlib.resources import files
from pathlib import Path

import cartopy.crs as ccrs
import cf_xarray.units  # pylint: disable=unused-import
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pint_xarray  # pylint: disable=unused-import
import rioxarray as rxr
import seaborn as sns
import toml
import xarray as xr
from dask.diagnostics import ProgressBar
from tqdm.auto import tqdm

from pism_ragis.download import save_netcdf
from pism_ragis.filtering import importance_sampling, run_importance_sampling
from pism_ragis.likelihood import log_normal, log_pseudo_huber
from pism_ragis.logger import get_logger
from pism_ragis.processing import filter_by_retreat_method, preprocess_config

xr.set_options(keep_attrs=True)

plt.style.use("tableau-colorblind10")
logger = get_logger("pism_ragis")

sim_alpha = 0.5
sim_cmap = sns.color_palette("crest", n_colors=4).as_hex()[0:3:2]
sim_cmap = ["#a6cee3", "#1f78b4"]
sim_cmap = ["#CC6677", "#882255"]
obs_alpha = 1.0
obs_cmap = ["0.8", "0.7"]
# obs_cmap = ["#88CCEE", "#44AA99"]
hist_cmap = ["#a6cee3", "#1f78b4"]
cartopy_crs = ccrs.NorthPolarStereo(
    central_longitude=-45, true_scale_latitude=70, globe=None
)


def plot_spatial_median(
    sim: xr.DataArray,
    obs: xr.DataArray,
    sim_var: str = "dhdt",
    obs_mean_var: str = "dhdt",
    smb_var="tendency_of_ice_mass_due_to_surface_mass_flux",
    flow_var="tendency_of_ice_mass_due_to_flow",
    fig_dir: Path = Path("figures"),
):
    """
    Plot the spatial median of simulated and observed data.

    Parameters
    ----------
    sim : xr.DataArray
        Simulated data array.
    obs : xr.DataArray
        Observed data array.
    sim_var : str, optional
        Variable name for the simulated data, by default "dhdt".
    obs_mean_var : str, optional
        Variable name for the observed mean data, by default "dhdt".
    smb_var : str, optional
        Variable name for the surface mass balance data, by default "tendency_of_ice_mass_due_to_surface_mass_flux".
    flow_var : str, optional
        Variable name for the flow data, by default "tendency_of_ice_mass_due_to_flow".
    fig_dir : Path, optional
        Directory to save the figures, by default "figures".
    """
    o = obs.sum(dim="time")[obs_mean_var]
    s = sim.median(dim="exp_id").sum(dim="time")[sim_var]
    smb = sim.median(dim="exp_id").sum(dim="time")[smb_var]
    flow = sim.median(dim="exp_id").sum(dim="time")[flow_var]

    fig = plt.figure(figsize=(6.4, 7.2))
    fig.set_dpi(600)
    ax_1 = fig.add_subplot(2, 2, 1, projection=cartopy_crs)
    ax_2 = fig.add_subplot(2, 2, 2, projection=cartopy_crs)
    ax_3 = fig.add_subplot(2, 2, 3, projection=cartopy_crs)
    ax_4 = fig.add_subplot(2, 2, 4, projection=cartopy_crs)
    cb = o.plot(
        ax=ax_1, vmin=-20, vmax=20, cmap="RdBu_r", extend="both", add_colorbar=False
    )
    s.plot(ax=ax_2, vmin=-20, vmax=20, cmap="RdBu_r", extend="both", add_colorbar=False)
    cb2 = smb.plot(
        ax=ax_3, vmin=-0.05, vmax=0.05, cmap="RdBu_r", extend="both", add_colorbar=False
    )
    flow.plot(
        ax=ax_4, vmin=-0.05, vmax=0.05, cmap="RdBu_r", extend="both", add_colorbar=False
    )
    for ax in [ax_1, ax_2, ax_3, ax_4]:
        ax.gridlines(
            draw_labels={"top": "x", "left": "y"},
            dms=True,
            xlocs=np.arange(-60, 60, 10),
            ylocs=np.arange(50, 88, 10),
            x_inline=False,
            y_inline=False,
            rotate_labels=20,
            ls="dotted",
            color="k",
        )
        ax.coastlines()
    ax_1.set_title("Observed dhdt")
    ax_2.set_title("Simulated Median dhdt")
    ax_3.set_title("Simulated Median SMB")
    ax_4.set_title("Simulated Median Flow")

    fig.colorbar(
        cb,
        ax=[ax_1, ax_2],
        location="right",
        orientation="vertical",
        extend="both",
        anchor=(0.5, 0.5),
        pad=0,
        shrink=0.75,
        label="dh/dt (m)",
    )
    fig.colorbar(
        cb2,
        ax=[ax_3, ax_4],
        location="right",
        orientation="vertical",
        extend="both",
        anchor=(0.5, 0.5),
        pad=0,
        shrink=0.75,
        label="mass (Gt)",
    )
    fig.savefig(fig_dir / Path("dhdt.png"), dpi=600)


if __name__ == "__main__":
    __spec__ = None  # type: ignore

    # set up the option parser
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.description = "Compute ensemble statistics."
    parser.add_argument(
        "--result_dir",
        help="""Result directory.""",
        type=str,
        default="./results",
    )
    parser.add_argument(
        "--obs_url",
        help="""Path to "observed" mass balance.""",
        type=str,
        default="data/itslive/ITS_LIVE_GRE_G0240_2018.nc",
    )
    parser.add_argument(
        "--outlier_variable",
        help="""Quantity to filter outliers. Default="grounding_line_flux".""",
        type=str,
        default="grounding_line_flux",
    )
    parser.add_argument(
        "--fudge_factor",
        help="""Observational uncertainty multiplier. Default=3""",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--n_jobs", help="""Number of parallel jobs.""", type=int, default=4
    )
    parser.add_argument(
        "--notebook",
        help="""Use when running in a notebook to display a nicer progress bar. Default=False.""",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--parallel",
        help="""Open dataset in parallel. Default=False.""",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--resampling_frequency",
        help="""Resampling data to resampling_frequency for importance sampling. Default is "MS".""",
        type=str,
        default="YS",
    )
    parser.add_argument(
        "--reference_year",
        help="""Reference year.""",
        type=int,
        default=1986,
    )
    parser.add_argument(
        "--temporal_range",
        help="""Time slice to extract.""",
        type=str,
        nargs=2,
        default=[2003, 2017],
    )
    parser.add_argument(
        "FILES",
        help="""Ensemble netCDF files.""",
        nargs="*",
    )

    options = parser.parse_args()
    spatial_files = options.FILES
    fudge_factor = options.fudge_factor
    notebook = options.notebook
    parallel = options.parallel
    reference_year = options.reference_year
    resampling_frequency = options.resampling_frequency
    result_dir = Path(options.result_dir)
    outlier_variable = options.outlier_variable
    ragis_config_file = Path(
        str(files("pism_ragis.data").joinpath("ragis_config.toml"))
    )
    ragis_config = toml.load(ragis_config_file)
    config = json.loads(json.dumps(ragis_config))
    params_short_dict = config["Parameters"]
    params = list(params_short_dict.keys())

    obs_mean_var = "dhdt"
    obs_std_var = "dhdt_err"
    sim_var = "dhdt"
    sum_dims = ["time", "y", "x"]
    retreat_methods = ["Prescribed"]

    filter_range = ["2003", "2020"]
    # print(m_ds.pism_config)

    print("Loading ensemble.")
    with ProgressBar():
        simulated = xr.open_mfdataset(
            spatial_files,
            preprocess=preprocess_config,
            parallel=True,
            decode_cf=True,
            decode_timedelta=True,
            engine="h5netcdf",
            combine="nested",
            concat_dim="exp_id",
        )

    simulated = simulated.sel(time=slice(*filter_range)).pint.quantify()
    simulated[sim_var] = simulated["dHdt"] * 1000.0 / 910.0
    simulated[sim_var] = simulated[sim_var].pint.to("m year^-1")
    simulated = simulated.pint.dequantify()

    observed = xr.open_mfdataset(
        "~/base/pism-ragis/data/mass_balance/Greenland_dhdt_mass*_1kmgrid_DB.nc",
        # chunks={"time": -1, "x": -1, "y": -1},
        chunks="auto",
    ).sel(time=slice(*filter_range))
    observed_resampled = observed.resample({"time": resampling_frequency}).mean()

    r_normal: list = []
    r_huber: list = []
    for retreat_method in retreat_methods:
        print("-" * 80)
        print(f"Retreat method: {retreat_method}")
        print("-" * 80)

        fig_dir = (
            result_dir / Path(f"retreat_{retreat_method.lower()}") / Path("figures")
        )
        fig_dir.mkdir(parents=True, exist_ok=True)

        simulated_retreat_filtered = filter_by_retreat_method(simulated, retreat_method)

        stats = simulated_retreat_filtered[["pism_config", "run_stats"]]

        simulated_retreat_resampled = (
            simulated_retreat_filtered.drop_vars(["pism_config", "run_stats"])
            .drop_dims(["pism_config_axis", "run_stats_axis"])
            .resample({"time": resampling_frequency})
            .mean(dim="time")
        )

        obs = observed_resampled.interp_like(
            simulated_retreat_resampled
        ).pint.quantify()
        for v in [obs_mean_var, obs_std_var]:
            obs[v] = obs[v].pint.to("m year^-1")
        obs = obs.pint.dequantify()

        obs_dhdt = obs["dhdt"]
        obs_mask = obs_dhdt.isnull()
        obs_mask = obs_mask.any(dim="time")

        sim = simulated_retreat_resampled.where(~obs_mask)
        sim = xr.merge([sim, stats])

        (
            prior_posterior,
            simulated_prior,
            simulated_posterior,
        ) = run_importance_sampling(
            observed=obs,
            simulated=sim,
            obs_mean_vars=[obs_mean_var],
            obs_std_vars=[obs_std_var],
            sim_vars=[sim_var],
            filter_range=filter_range,
            fudge_factor=fudge_factor,
            sum_dims=sum_dims,
            params=params,
        )

        simulated_posterior.to_netcdf("posterior.nc")
        # print(f"Importance sampling using {obs_mean_var}.")
        # f = importance_sampling(
        #     observed=obs,
        #     simulated=sim,
        #     log_likelihood=log_normal,
        #     n_samples=len(sim.exp_id),
        #     fudge_factor=fudge_factor,
        #     obs_mean_var=obs_mean_var,
        #     obs_std_var=obs_std_var,
        #     sim_var=sim_var,
        #     sum_dim=["time", "y", "x"],
        # )
        # with ProgressBar():
        #     sum_filtered_ids = f.compute()
        # r_normal.append(sum_filtered_ids)

        # print(f"Importance sampling using {obs_mean_var}.")
        # f = importance_sampling(
        #     observed=obs,
        #     simulated=sim,
        #     log_likelihood=log_pseudo_huber,
        #     n_samples=len(sim.exp_id),
        #     fudge_factor=fudge_factor,
        #     obs_mean_var=obs_mean_var,
        #     obs_std_var=obs_std_var,
        #     sim_var=sim_var,
        #     sum_dim=["time", "y", "x"],
        # )
        # with ProgressBar():
        #     sum_filtered_ids = f.compute()
        # r_huber.append(sum_filtered_ids)

    # s = sim_ds["velsurf_mag"]
    # o = obs_ds["v"]
    # print(xskillscore.rmse(s, o, dim=["x", "y"], skipna=True).values)

    # fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    # s.median(dim="exp_id").plot(ax=axs[0], vmin=0, vmax=500, label=False)
    # o.plot(ax=axs[1], vmin=0, vmax=500)
    # plt.show()

    # observed = (
    #     xr.open_dataset(options.obs_url, chunks="auto")
    #     .sel({"time": str(sampling_year)})
    #     .mean(dim="time")
    # )
    # obs_file_nc = (
    #     result_dir
    #     / Path(f"retreat_{retreat_method.lower()}")
    #     / Path(f"observed_dhdt_{start_date}_{end_date}.nc")
    # )
    # sim_file_nc = (
    #     result_dir
    #     / Path(f"retreat_{retreat_method.lower()}")
    #     / Path(f"simulated_dhdt_{start_date}_{end_date}.nc")
    # )
    # print(f"Saving observations to {obs_file_nc}")
    # with ProgressBar():
    #     obs.to_netcdf(obs_file_nc)
    # print(f"Saving simulations to {sim_file_nc}")
    # with ProgressBar():
    #     sim.to_netcdf(sim_file_nc)

    # save_netcdf(obs, obs_file_nc)
    # save_netcdf(sim, sim_file_nc)

    # obs_file_zarr = result_dir / Path(f"retreat_{retreat_method.lower()}") / Path(f"observed_dhdt_{start_date}_{end_date}.zarr")
    # sim_file_zarr = result_dir / Path(f"retreat_{retreat_method.lower()}") / Path(f"observed_dhdt_{start_date}_{end_date}.zarr")
    # with ProgressBar():
    #     obs.to_zarr(obs_file_zarr, mode="w")
    #     sim.to_zarr(sim_file_zarr, mode="w")

    # start_time = time.time()

    # plot_spatial_median(
    #     sim,
    #     obs,
    #     sim_var=sim_var,
    #     obs_mean_var=obs_mean_var,
    #     smb_var=smb_var,
    #     flow_var=flow_var,
    #     fig_dir=fig_dir,
    # )
    # end_time = time.time()
    # elapsed_time = end_time - start_time
    # print(f"Plotting finished in {elapsed_time:.2f} seconds")
