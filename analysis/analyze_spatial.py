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

# pylint: disable=unused-import,too-many-positional-arguments,unused-argument
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
from dask.distributed import Client, progress
from tqdm.auto import tqdm

import pism_ragis.processing as prp
from pism_ragis.download import save_netcdf
from pism_ragis.filtering import (
    importance_sampling,
)
from pism_ragis.likelihood import log_jaccard_score_xr, log_normal_xr
from pism_ragis.logger import get_logger
from pism_ragis.plotting import plot_mapplane, plot_prior_posteriors
from pism_ragis.processing import (
    filter_by_retreat_method,
    load_ensemble,
    prepare_dhdt,
    prepare_grace,
    prepare_liafr,
    prepare_v,
    preprocess_config,
)

xr.set_options(
    keep_attrs=True,
    warn_for_unclosed_files=True,
    use_flox=True,
    use_bottleneck=True,
    use_opt_einsum=True,
)

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
cartopy_crs = ccrs.NorthPolarStereo(central_longitude=-45, true_scale_latitude=70, globe=None)


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
        "--filter_var",
        help="""Filter variable. Default='retreat'""",
        type=str,
        choices=["retreat", "dhdt", "speed", "grace"],
        default="land_ice_area_fraction_retreat",
    )
    parser.add_argument(
        "--data_dir",
        help="""Observational uncertainty multiplier. Default=3""",
        type=str,
        default="land_ice_are_fraction_retreat",
    )
    parser.add_argument("--n_jobs", help="""Number of parallel jobs.""", type=int, default=4)
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
        help="""Resampling data to resampling_frequency for importance sampling. Default is "YS".""",
        type=str,
        default="YS",
    )
    parser.add_argument(
        "--engine",
        help="""Engine for xarray. Default="h5netcdf".""",
        type=str,
        default="netcdf4",
    )
    parser.add_argument(
        "FILES",
        help="""Ensemble netCDF files.""",
        nargs="*",
    )

    options = parser.parse_args()
    engine = options.engine
    spatial_files = sorted(options.FILES)
    filter_var = options.filter_var
    fudge_factor = options.fudge_factor
    notebook = options.notebook
    parallel = options.parallel
    input_data_dir = options.data_dir
    resampling_frequency = options.resampling_frequency
    outlier_variable = options.outlier_variable
    ragis_config_file = Path(str(files("pism_ragis.data").joinpath("ragis_config.toml")))
    ragis_config = toml.load(ragis_config_file)
    config = json.loads(json.dumps(ragis_config))
    params_short_dict = config["Parameters"]
    params = list(params_short_dict.keys())

    result_dir = Path(options.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    obs_cmap = config["Plotting"]["obs_cmap"]
    sim_cmap = config["Plotting"]["sim_cmap"]

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

    client = Client()
    print(f"Open client in browser: {client.dashboard_link}")
    start = time.time()

    log_likelihood: Callable
    prepare_input: Callable
    obs_mean_var = "land_ice_area_fraction_retreat"
    obs_std_var = "land_ice_area_fraction_retreat_uncertainty"
    sim_var = "land_ice_area_fraction_retreat"
    filter_range = ["1980", "2019"]
    coarsen: dict | None = None
    sum_dims: list | None = None

    obs_file: str
    if filter_var == "retreat":
        obs_mean_var = "land_ice_area_fraction_retreat"
        obs_std_var = "land_ice_area_fraction_retreat_uncertainty"
        sim_var = "land_ice_area_fraction_retreat"
        filter_range = ["1980", "2019"]
        sum_dims = ["y", "x", "time"]
        obs_file = input_data_dir + "/front_retreat/pism_g450m_frontretreat_calfin_1980_2019_YM.nc"
        log_likelihood = log_jaccard_score_xr
        prepare_input = prepare_liafr
        coarsen = None
        sum_dims = ["y", "x", "time"]
    elif filter_var == "dhdt":
        obs_mean_var = "dhdt"
        obs_std_var = "dhdt_err"
        sim_var = "dhdt"
        filter_range = ["2003", "2020"]
        sum_dims = ["time", "y", "x"]
        obs_file = input_data_dir + "/mass_balance/Greenland_dhdt_mass*_1kmgrid_DB.nc"
        prepare_input = prepare_dhdt
        log_likelihood = log_normal_xr
        coarsen = {"x": 5, "y": 5}
    elif filter_var == "speed":
        obs_mean_var = "v"
        obs_std_var = "v_err"
        sim_var = "velsurf_mag"
        filter_range = ["1985", "2018"]
        sum_dims = ["time", "y", "x"]
        obs_file = input_data_dir + "/itslive/ITS_LIVE_GRE_G0240_*.nc"
        prepare_input = prepare_v
        log_likelihood = log_normal_xr
        coarsen = {"x": 5, "y": 5}
    elif filter_var == "grace":
        obs_mean_var = "mass_balance"
        obs_std_var = "mass_balance_err"
        sim_var = "tendency_of_ice_mass"
        filter_range = ["2002", "2020"]
        sum_dims = ["time", "lat", "lon"]
        obs_file = input_data_dir + "/mass_balance/grace_gsfc_greenland_mass_balance.nc"
        prepare_input = prepare_grace
        log_likelihood = log_normal_xr
        coarsen = None
    else:
        print(f"{filter_var} not supported")

    retreat_methods = ["Free"]

    print("Loading ensemble.")
    simulated = load_ensemble(
        spatial_files,
        preprocess=preprocess_config,
        parallel=True,
        engine=engine,
    )
    simulated = simulated.sel({"time": slice(*filter_range)})

    observed = xr.open_mfdataset(obs_file, chunks={"time": -1})
    observed = observed.sel({"time": slice(*filter_range)})

    stats = simulated[["pism_config", "run_stats"]]

    simulated = (
        simulated.drop_vars(["pism_config", "run_stats"])
        .drop_dims(["pism_config_axis", "run_stats_axis"])
        .resample({"time": resampling_frequency})
        .mean()
    )
    simulated = xr.merge([simulated, stats])

    simulated_all = simulated
    simulated_all["ensemble"] = "All"

    bins_dict = config["Posterior Bins"]
    parameter_categories = config["Parameter Categories"]
    params_sorted_by_category: dict = {group: [] for group in sorted(parameter_categories.values())}
    for param in params:
        prefix = param.split(".")[0]
        if prefix in parameter_categories:
            group = parameter_categories[prefix]
            if param not in params_sorted_by_category[group]:
                params_sorted_by_category[group].append(param)

    params_sorted_list = list(chain(*params_sorted_by_category.values()))
    params_sorted_dict = {k: params_short_dict[k] for k in params_sorted_list}
    short_bins_dict = {params_short_dict[key]: bins_dict[key] for key in params_short_dict if key in bins_dict}
    plot_params = params_sorted_dict.copy()
    del plot_params["geometry.front_retreat.prescribed.file"]

    for retreat_method in retreat_methods:
        print("-" * 80)
        print(f"Retreat method: {retreat_method}")
        print("-" * 80)

        retreat_dir = result_dir / Path(f"retreat_{retreat_method.lower()}")
        retreat_dir.mkdir(parents=True, exist_ok=True)
        data_dir = retreat_dir / Path("posteriors")
        data_dir.mkdir(parents=True, exist_ok=True)
        fig_dir = retreat_dir / Path("figures")
        fig_dir.mkdir(parents=True, exist_ok=True)

        simulated_retreat_filtered = filter_by_retreat_method(simulated, retreat_method)
        stats = simulated_retreat_filtered[["pism_config", "run_stats"]]

        obs, sim = prepare_input(
            observed,
            simulated_retreat_filtered,
            obs_mean_var,
            obs_std_var,
            sim_var,
            coarsen=coarsen,
        )
        # sim = xr.merge([sim, stats])

        # (prior_posterior, simulated_prior, simulated_posterior, simulated_weights) = run_importance_sampling(
        #     observed=obs,
        #     simulated=sim,
        #     obs_mean_vars=[obs_mean_var],
        #     obs_std_vars=[obs_std_var],
        #     sim_vars=[sim_var],
        #     log_likelihood=log_likelihood,
        #     filter_range=filter_range,
        #     fudge_factor=fudge_factor,
        #     sum_dims=sum_dims,
        #     params=params,
        # )

        # # Apply the functions to the corresponding columns
        # for col, functions in column_function_mapping.items():
        #     for func in functions:
        #         prior_posterior[col] = prior_posterior[col].apply(func)

        # prior_posterior["basin"] = "GIS"
        # posterior = prior_posterior[prior_posterior["ensemble"] == "Posterior"].copy()
        # posterior["fudge_factor"] = fudge_factor

        # posterior.to_parquet(
        #     data_dir / Path(f"""posterior_retreat_filtered_by_{sim_var}_{filter_range[0]}-{filter_range[1]}.parquet""")
        # )

        # plot_prior_posteriors(
        #     prior_posterior.rename(columns=plot_params),
        #     x_order=plot_params.values(),
        #     fig_dir=fig_dir,
        #     bins_dict=short_bins_dict,
        # )

        # prior_nc = data_dir / Path(
        #     f"""simulated_prior_retreat_filtered_by_{sim_var}_{filter_range[0]}-{filter_range[1]}.nc"""
        # )
        # print(f"Writing {prior_nc}")
        # save_netcdf(simulated_prior.chunk("auto"), prior_nc)

        # simulated_posterior["fudge_factor"] = fudge_factor
        # posterior_nc = data_dir / Path(
        #     f"""simulated_posterior_retreat_filtered_by_{sim_var}_{filter_range[0]}-{filter_range[1]}.nc"""
        # )
        # print(f"Writing {posterior_nc}")
        # save_netcdf(simulated_posterior.chunk("auto"), posterior_nc)

        # simulated_weights = simulated_weights.to_dataset()
        # simulated_weights["fudge_factor"] = fudge_factor
        # save_netcdf(
        #     simulated_weights,
        #     data_dir
        #     / Path(f"""simulated_weights_retreat_filtered_by_{sim_var}_{filter_range[0]}-{filter_range[1]}.nc"""),
        # )

    client.close()
    end = time.time()
    time_elapsed = end - start
    print(f"Time elapsed {time_elapsed:.0f}s")
