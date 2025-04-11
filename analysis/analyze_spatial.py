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
from pism_ragis.filtering import (
    importance_sampling,
    run_importance_sampling,
)
from pism_ragis.likelihood import log_jaccard_score_xr, log_normal_xr
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


def prepare_liafr(
    obs_ds: xr.Dataset,
    sim_ds: xr.Dataset,
    obs_mean_var,
    obs_std_var,
    sim_var: str,
) -> tuple[xr.Dataset, xr.Dataset]:
    """
    Prepare land ice area fraction retreat data for analysis.

    Parameters
    ----------
    obs_ds : xr.Dataset
        The observed dataset.
    sim_ds : xr.Dataset
        The simulated dataset.
    obs_mean_var : str
        The variable name for the observed mean data.
    obs_std_var : str
        The variable name for the observed standard deviation data.
    sim_var : str
        The variable name for the simulated data.

    Returns
    -------
    tuple[xr.Dataset, xr.Dataset]
        A tuple containing the prepared observed and simulated datasets.
    """
    s_liafr = xr.where(sim_ds["thk"] > 10, 1, 0).resample({"time": "YS"}).mean()
    s_liafr.name = sim_var
    o_liafr = (
        obs_ds[obs_mean_var]
        .resample({"time": "YS"})
        .mean()
        .interp_like(s_liafr, method="nearest")
        .fillna(0)
    )
    s_liafr_b = s_liafr.astype(bool)
    o_liafr_b = o_liafr.astype(bool)
    sim = s_liafr_b.to_dataset()

    o_liafr_b_uncertainty = xr.ones_like(o_liafr_b)
    o_liafr_b_uncertainty.name = obs_std_var
    obs = xr.merge([o_liafr_b, o_liafr_b_uncertainty])

    return obs, sim


def prepare_dhdt(
    obs_ds: xr.Dataset,
    sim_ds: xr.Dataset,
    obs_mean_var,
    obs_std_var,
    sim_var: str,
    coarsen: dict | None = None,
) -> tuple[xr.Dataset, xr.Dataset]:
    """
    Prepare dh/dt data for analysis.

    Parameters
    ----------
    obs_ds : xr.Dataset
        The observed dataset.
    sim_ds : xr.Dataset
        The simulated dataset.
    obs_mean_var : str
        The variable name for the observed mean data.
    obs_std_var : str
        The variable name for the observed standard deviation data.
    sim_var : str
        The variable name for the simulated data.
    coarsen : dict or None, optional
        Dictionary specifying the dimensions and factors for coarsening the simulated data, by default None.

    Returns
    -------
    tuple[xr.Dataset, xr.Dataset]
        A tuple containing the prepared observed and simulated datasets.
    """
    sim_ds = sim_ds.pint.quantify()
    sim_ds[sim_var] = sim_ds["dHdt"] * 1000.0 / 910.0
    sim_ds[sim_var] = sim_ds[sim_var].pint.to("m year^-1")
    sim_ds = sim_ds.pint.dequantify()

    sim_retreat_resampled = (
        sim_ds.drop_vars(["pism_config", "run_stats"])
        .drop_dims(["pism_config_axis", "run_stats_axis"])
        .resample({"time": "YS"})
        .mean(dim="time")
    )

    obs = obs_ds.interp_like(sim_retreat_resampled).pint.quantify()
    for v in [obs_mean_var, obs_std_var]:
        obs[v] = obs[v].pint.to("m year^-1")
    obs = obs.pint.dequantify()

    obs_dhdt = obs[obs_mean_var]
    obs_mask = obs_dhdt.isnull()
    obs_mask = obs_mask.any(dim="time")

    sim = sim_retreat_resampled.where(~obs_mask)[["dhdt"]]

    if coarsen is not None:
        sim = sim.coarsen(coarsen).mean()
        obs = obs.interp_like(sim)

    return obs, sim


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
        help="""Filter variable. Default='land_ice_area_retreat'""",
        type=str,
        choices=["land_ice_area_fraction_retreat", "dhdt"],
        default="land_ice_area_fraction_retreat",
    )
    parser.add_argument(
        "--data_dir",
        help="""Observational uncertainty multiplier. Default=3""",
        type=str,
        default="land_ice_are_fraction_retreat",
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
    spatial_files = options.FILES
    filter_var = options.filter_var
    fudge_factor = options.fudge_factor
    notebook = options.notebook
    parallel = options.parallel
    input_data_dir = options.data_dir
    resampling_frequency = options.resampling_frequency
    outlier_variable = options.outlier_variable
    ragis_config_file = Path(
        str(files("pism_ragis.data").joinpath("ragis_config.toml"))
    )
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

    log_likelihood: Callable
    prepare_input: Callable
    obs_mean_var = "land_ice_area_fraction_retreat"
    obs_std_var = "land_ice_area_fraction_retreat_uncertainty"
    sim_var = "land_ice_area_fraction_retreat"
    filter_range = ["1980", "2019"]
    sum_dims = ["y", "x", "time"]
    obs_file: str
    if filter_var == "land_ice_area_fraction_retreat":
        obs_mean_var = "land_ice_area_fraction_retreat"
        obs_std_var = "land_ice_area_fraction_retreat_uncertainty"
        sim_var = "land_ice_area_fraction_retreat"
        filter_range = ["1980", "2019"]
        sum_dims = ["y", "x", "time"]
        obs_file = (
            input_data_dir
            + "/front_retreat/pism_g450m_frontretreat_calfin_1980_2019_YM.nc"
        )
        log_likelihood = log_jaccard_score_xr
        prepare_input = prepare_liafr
    elif filter_var == "dhdt":
        obs_mean_var = "dhdt"
        obs_std_var = "dhdt_err"
        sim_var = "dhdt"
        filter_range = ["2003", "2020"]
        sum_dims = ["time", "y", "x"]
        obs_file = input_data_dir + "/mass_balance/Greenland_dhdt_mass*_1kmgrid_DB.nc"
        prepare_input = prepare_dhdt
        log_likelihood = log_normal_xr
    else:
        print(f"{filter_var} not supported")

    retreat_methods = ["Free"]

    print("Loading ensemble.")
    with ProgressBar():
        simulated = xr.open_mfdataset(
            spatial_files,
            preprocess=preprocess_config,
            parallel=True,
            decode_cf=True,
            decode_timedelta=True,
            engine=engine,
            combine="nested",
            concat_dim="exp_id",
        ).sel({"time": slice(*filter_range)})

    observed = xr.open_mfdataset(obs_file).sel({"time": slice(*filter_range)})

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
            coarsen={"x": 5, "y": 5},
        )
        sim = xr.merge([sim, stats])

        (prior_posterior, simulated_prior, simulated_posterior, simulated_weights) = (
            run_importance_sampling(
                observed=obs,
                simulated=sim,
                obs_mean_vars=[obs_mean_var],
                obs_std_vars=[obs_std_var],
                sim_vars=[sim_var],
                log_likelihood=log_likelihood,
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

        prior_posterior["basin"] = "GIS"
        posterior = prior_posterior[prior_posterior["ensemble"] == "Posterior"].copy()
        posterior["fudge_factor"] = fudge_factor

        posterior.to_parquet(
            data_dir
            / Path(
                f"""posterior_retreat_filtered_by_{sim_var}_{filter_range[0]}-{filter_range[1]}.parquet"""
            )
        )

        plot_prior_posteriors(
            prior_posterior.rename(columns=plot_params),
            x_order=plot_params.values(),
            fig_dir=fig_dir,
            bins_dict=short_bins_dict,
        )

        prior_nc = data_dir / Path(
            f"""simulated_prior_retreat_filtered_by_{sim_var}_{filter_range[0]}-{filter_range[1]}.nc"""
        )
        print(f"Writing {prior_nc}")
        save_netcdf(simulated_prior.chunk("auto"), prior_nc)

        simulated_posterior["fudge_factor"] = fudge_factor
        posterior_nc = data_dir / Path(
            f"""simulated_posterior_retreat_filtered_by_{sim_var}_{filter_range[0]}-{filter_range[1]}.nc"""
        )
        print(f"Writing {posterior_nc}")
        save_netcdf(simulated_posterior.chunk("auto"), posterior_nc)

        simulated_weights = simulated_weights.to_dataset()
        simulated_weights["fudge_factor"] = fudge_factor
        save_netcdf(
            simulated_weights,
            data_dir
            / Path(
                f"""simulated_weights_retreat_filtered_by_{sim_var}_{filter_range[0]}-{filter_range[1]}.nc"""
            ),
        )
