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
from itertools import chain
from pathlib import Path
from typing import Callable

import cartopy.crs as ccrs
import cf_xarray.units  # pylint: disable=unused-import
import matplotlib as mpl
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

import pism_ragis.processing as prp
from pism_ragis.download import save_netcdf
from pism_ragis.filtering import importance_sampling, run_importance_sampling
from pism_ragis.likelihood import log_jaccard_score_xr
from pism_ragis.logger import get_logger
from pism_ragis.plotting import plot_prior_posteriors
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

    obs_cmap = config["Plotting"]["obs_cmap"]
    sim_cmap = config["Plotting"]["sim_cmap"]

    result_dir = Path(options.result_dir)
    data_dir = result_dir / Path("retreat_posteriors")
    data_dir.mkdir(parents=True, exist_ok=True)

    column_function_mapping: dict[str, list[Callable]] = {
        "surface.given.file": [prp.simplify_path, prp.simplify_climate],
        "ocean.th.file": [prp.simplify_path, prp.simplify_ocean],
        "geometry.front_retreat.prescribed.file": [prp.simplify_retreat],
    }

    rcparams = {
        "axes.linewidth": 0.25,
        "xtick.direction": "in",
        "xtick.major.size": 2.5,
        "xtick.major.width": 0.25,
        "ytick.direction": "in",
        "ytick.major.size": 2.5,
        "ytick.major.width": 0.25,
        "hatch.linewidth": 0.25,
    }

    mpl.rcParams.update(rcparams)

    filter_range = ["1980", "2019"]
    sum_dims = ["y", "x", "time"]

    obs_mean_var = "land_ice_area_fraction_retreat"
    obs_std_var = "land_ice_area_fraction_retreat_uncertainty"
    sim_var = "land_ice_area_fraction_retreat"

    retreat_methods = ["Free"]

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
        ).sel({"time": slice(*filter_range)})

    observed = xr.open_dataset(
        "/Users/andy/base/pism-ragis/data/front_retreat/pism_g450m_frontretreat_calfin_1972_2019_YM.nc"
    ).sel({"time": slice(*filter_range)})

    bins_dict = config["Posterior Bins"]
    parameter_categories = config["Parameter Categories"]
    params_sorted_by_category: dict = {
        group: [] for group in sorted(parameter_categories.values())
    }
    for param in params:
        prefix = param.split(".")[0]
        if prefix in parameter_categories:
            group = parameter_categories[prefix]
            if param not in params_sorted_by_category[group]:
                params_sorted_by_category[group].append(param)

    params_sorted_list = list(chain(*params_sorted_by_category.values()))
    params_sorted_dict = {k: params_short_dict[k] for k in params_sorted_list}
    short_bins_dict = {
        params_short_dict[key]: bins_dict[key]
        for key in params_short_dict
        if key in bins_dict
    }

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

        s_liafr = (
            xr.where(simulated_retreat_filtered["thk"] > 10, 1, 0)
            .resample({"time": resampling_frequency})
            .mean()
        )
        s_liafr.name = sim_var
        o_liafr = (
            observed["land_ice_area_fraction_retreat"]
            .resample({"time": resampling_frequency})
            .mean()
            .interp_like(s_liafr, method="nearest")
            .fillna(0)
        )
        s_liafr_b = s_liafr.astype(bool)
        o_liafr_b = o_liafr.astype(bool)

        o_liafr_b_uncertainty = xr.ones_like(o_liafr_b)
        o_liafr_b_uncertainty.name = obs_std_var
        obs = xr.merge([o_liafr_b, o_liafr_b_uncertainty])

        sim = xr.merge([s_liafr_b.to_dataset(), stats])

        (prior_posterior, simulated_prior, simulated_posterior, simulated_weights) = (
            run_importance_sampling(
                observed=obs,
                simulated=sim,
                obs_mean_vars=[obs_mean_var],
                obs_std_vars=[obs_std_var],
                sim_vars=[sim_var],
                log_likelihood=log_jaccard_score_xr,
                filter_range=filter_range,
                fudge_factor=fudge_factor,
                sum_dims=sum_dims,
                params=params,
            )
        )

        # Apply the functions to the corresponding columns
        for col, functions in column_function_mapping.items():
            for func in functions:
                prior_posterior[col] = prior_posterior[col].apply(func)

        if "frontal_melt.routing.parameter_a" in prior_posterior.columns:
            prior_posterior["frontal_melt.routing.parameter_a"] *= 10**4
        if "ocean.th.gamma_T" in prior_posterior.columns:
            prior_posterior["ocean.th.gamma_T"] *= 10**4
        if "calving.vonmises_calving.sigma_max" in prior_posterior.columns:
            prior_posterior["calving.vonmises_calving.sigma_max"] *= 10**-3

        prior_posterior["basin"] = "GIS"
        prior_posterior.to_parquet(
            data_dir
            / Path(
                f"""prior_posterior_retreat_{retreat_method}_{filter_range[0]}-{filter_range[1]}.parquet"""
            )
        )

        plot_prior_posteriors(
            prior_posterior.rename(columns=params_sorted_dict),
            x_order=params_sorted_dict.values(),
            fig_dir=fig_dir,
            bins_dict=short_bins_dict,
        )

        # f = importance_sampling(
        #     simulated=sim,
        #     observed=obs,
        #     log_likelihood=log_jaccard_score_xr,
        #     fudge_factor=fudge_factor,
        #     n_samples=sim.sizes["exp_id"],
        #     obs_mean_var=obs_mean_var,
        #     obs_std_var=obs_std_var,
        #     sim_var=sim_var,
        #     sum_dims=sum_dims,
        # )

        # with ProgressBar():
        #     result = f.compute()
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
