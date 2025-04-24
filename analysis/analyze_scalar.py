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

# pylint: disable=too-many-positional-arguments

"""
Analyze RAGIS ensemble.
"""

import json
import warnings
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from importlib.resources import files
from itertools import chain
from pathlib import Path
from typing import Callable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import toml
import xarray as xr

import pism_ragis.processing as prp
from pism_ragis.analyze import run_sensitivity_analysis
from pism_ragis.filtering import filter_outliers, run_importance_sampling
from pism_ragis.logger import get_logger
from pism_ragis.plotting import (
    plot_basins,
    plot_prior_posteriors,
    plot_sensitivity_indices,
    plot_timeseries,
)

logger = get_logger("pism_ragis")

# mpl.use("Agg")
xr.set_options(keep_attrs=True)
plt.style.use("tableau-colorblind10")
# Ignore specific RuntimeWarnings
warnings.filterwarnings(
    "ignore", category=RuntimeWarning, message="overflow encountered in exp"
)
warnings.filterwarnings(
    "ignore", category=RuntimeWarning, message="invalid value encountered in divide"
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
        "--observed_url",
        help="""Path to "observed" mass balance.""",
        type=str,
        default="data/mass_balance/mankoff_greenland_mass_balance.nc",
    )
    parser.add_argument(
        "--engine",
        help="""Engine for xarray. Default="h5netcdf".""",
        type=str,
        default="netcdf4",
    )
    parser.add_argument(
        "--filter_range",
        help="""Time slice used for Importance Sampling, needs an integer year. Default="1986 2019". """,
        type=int,
        nargs=2,
        default=[1990, 2019],
    )
    parser.add_argument(
        "--ci",
        help="""Credibility Interval percentiles. Default="0.025 0.0975". """,
        type=float,
        nargs=2,
        default=[0.025, 0.975],
    )
    parser.add_argument(
        "--valid_range",
        help="""Ensemble members outside this range are removed. Default="-50000 0". """,
        type=float,
        nargs=2,
        default=[-2500.0, -100.0],
    )
    parser.add_argument(
        "--outlier_variable",
        help="""Quantity to filter outliers. Default="grounding_line_flux".""",
        type=str,
        default="grounding_line_flux",
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
        "--reference_date",
        help="""Reference date.""",
        type=str,
        default="1986-01-1",
    )
    parser.add_argument(
        "--n_jobs",
        help="""Number of parallel jobs. Default=8.""",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--temporal_range",
        help="""Time slice to extract.""",
        type=str,
        nargs=2,
        default=None,
    )
    parser.add_argument(
        "FILES",
        help="""Ensemble netCDF files.""",
        nargs="*",
    )

    parser.add_argument(
        "--log",
        default="WARNING",
        help="Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    )

    print("================================================================")
    print("Analyze RAGIS Scalars")
    print("================================================================\n\n")

    options, unknown = parser.parse_known_args()
    basin_files = options.FILES
    ci = options.ci
    engine = options.engine
    filter_range = options.filter_range
    notebook = options.notebook
    parallel = options.parallel
    reference_date = options.reference_date
    resampling_frequency = options.resampling_frequency
    outlier_variable = options.outlier_variable
    valid_range = options.valid_range
    ragis_config_file = Path(
        str(files("pism_ragis.data").joinpath("ragis_config.toml"))
    )
    ragis_config = toml.load(ragis_config_file)
    config = json.loads(json.dumps(ragis_config))
    params_short_dict = config["Parameters"]
    params = list(params_short_dict.keys())
    obs_cmap = config["Plotting"]["obs_cmap"]
    sim_cmap = config["Plotting"]["sim_cmap"]
    fudge_factor = config["Importance Sampling"]["mankoff_fudge_factor"]
    retreat_methods = ["All", "Free", "Prescribed"]

    if not notebook:
        backend = "Agg"

    result_dir = Path(options.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    data_dir = result_dir / Path("posteriors")
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

    simulated = prp.prepare_simulations(
        basin_files, config, reference_date, parallel=parallel, engine=engine
    )
    simulated = simulated.dropna(dim="exp_id")
    observed = prp.prepare_observations(
        options.observed_url,
        config,
        reference_date,
        engine=engine,
    )

    simulated_all = simulated
    simulated_all["ensemble"] = "All"
    fig_dir = result_dir / Path("figures")
    fig_dir.mkdir(parents=True, exist_ok=True)
    plot_timeseries(
        observed.sel({"basin": "GIS"}),
        sim_prior=simulated_all.sel({"basin": "GIS"}),
        fig_dir=fig_dir,
        figsize=(4.8, 3.6),
        fontsize=6,
        plot_vars=["cumulative_mass_flux", "grounding_line_flux"],
        config=config,
        reference_date=reference_date,
        y_lim=[[-20_000, 4000], [-2000, 0]],
        add_lineplot=True,
        add_median=True,
    )

    da = observed.sel(time=slice(f"{filter_range[0]}", f"{filter_range[-1]}"))[
        "grounding_line_flux"
    ].mean(dim="time")
    posterior_basins_sorted = observed.basin.sortby(da).values
    print(
        "Mean Grounding Line Flux by basin:\n",
        " \n".join(
            [
                f"""{basin}: {np.round(flux, decimals=0)} {da.attrs["units"]}"""
                for basin, flux in zip(da.basin.values, da.values)
            ]
        ),
    )

    discharge_var = ragis_config["Flux Variables"]["grounding_line_flux"]
    discharge_uncertainty_var = ragis_config["Flux Uncertainty Variables"][
        "grounding_line_flux_uncertainty"
    ]

    gis_obs_discharge = observed.sel({"basin": "GIS", "time": slice("1980", "2019")})[
        [discharge_var, discharge_uncertainty_var]
    ]
    gis_obs_discharge_mean = gis_obs_discharge.mean(dim="time").compute()
    gis_sim_discharge = (
        simulated.sel({"basin": "GIS", "time": slice("1980", "2019")})[discharge_var]
        .mean(dim="time")
        .compute()
    )
    gis_sim_discharge_median = gis_sim_discharge.median(dim="exp_id").compute()
    gis_sim_discharge_ci = gis_sim_discharge.quantile(ci, dim="exp_id").compute()
    print(
        f"""Observed {discharge_var} mean: {gis_obs_discharge_mean[discharge_var].values.round()} std: {gis_obs_discharge_mean[discharge_uncertainty_var].values.round()} {gis_obs_discharge_mean[discharge_var].attrs["units"]}"""
    )
    print(
        f"""Simulated {discharge_var} median {gis_sim_discharge_median.values.round()}  {gis_sim_discharge_ci.values.round()} {ci} {gis_sim_discharge_median.attrs["units"]}"""
    )

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
    plot_params = params_sorted_dict.copy()
    del plot_params["geometry.front_retreat.prescribed.file"]

    short_bins_dict = {
        params_short_dict[key]: bins_dict[key]
        for key in params_short_dict
        if key in bins_dict
    }
    pp_retreat_list: list[pd.DataFrame] = []
    posterior_ds_dict: dict[str, xr.Dataset] = {}
    for retreat_method in retreat_methods:
        print("-" * 8)
        print(f"Retreat method: {retreat_method}")
        print("-" * 80)

        retreat_dir = result_dir / Path(f"retreat_{retreat_method.lower()}")
        retreat_dir.mkdir(parents=True, exist_ok=True)
        data_dir = retreat_dir / Path("posteriors")
        data_dir.mkdir(parents=True, exist_ok=True)
        fig_dir = retreat_dir / Path("figures")
        fig_dir.mkdir(parents=True, exist_ok=True)

        simulated_retreat_filtered = prp.filter_by_retreat_method(
            simulated, retreat_method, compute=True
        )

        simulated_valid, simulated_outliers = filter_outliers(
            simulated_retreat_filtered,
            valid_range=valid_range,
            outlier_variable=outlier_variable,
            subset={"basin": "GIS"},
        )
        stats = simulated_valid[["pism_config", "run_stats"]]

        outliers_config = prp.filter_config(simulated_outliers["pism_config"], params)
        outliers_df = prp.config_to_dataframe(outliers_config, ensemble="Outliers")

        valid_config = prp.filter_config(simulated_valid["pism_config"], params)
        valid_df = prp.config_to_dataframe(valid_config, ensemble="Valid")

        obs_basins = set(observed.basin.values)
        sim_basins = set(simulated.basin.values)

        intersection = list(sim_basins.intersection(obs_basins))

        observed_basins = observed.sel({"basin": intersection})
        simulated_basins = simulated_valid.sel({"basin": intersection})

        observed_basins_resampled = observed_basins.resample(
            {"time": resampling_frequency}
        ).mean()
        simulated_basins_resampled = (
            simulated_basins.drop_vars(["pism_config", "run_stats"])
            .drop_dims(["pism_config_axis", "run_stats_axis"])
            .resample({"time": resampling_frequency})
            .mean()
        )
        simulated_basins_resampled = xr.merge(
            [simulated_basins_resampled, stats.sel({"basin": intersection})]
        )

        obs_mean_vars: list[str] = [
            "grounding_line_flux",
            "mass_balance",
        ]
        obs_std_vars: list[str] = [
            "grounding_line_flux_uncertainty",
            "mass_balance_uncertainty",
        ]
        sim_vars: list[str] = [
            "grounding_line_flux",
            "mass_balance",
        ]

        sim_plot_vars = (
            [ragis_config["Cumulative Variables"]["cumulative_mass_flux"]]
            + list(ragis_config["Flux Variables"].values())
            + ["ensemble"]
        )

        (prior_posterior, simulated_prior, simulated_posterior, simulated_weights) = (
            run_importance_sampling(
                observed=observed_basins_resampled,
                simulated=simulated_basins_resampled,
                obs_mean_vars=obs_mean_vars,
                obs_std_vars=obs_std_vars,
                sim_vars=sim_vars,
                filter_range=filter_range,
                fudge_factor=fudge_factor,
                params=params,
            )
        )

        for filter_var in obs_mean_vars:
            plot_basins(
                observed_basins_resampled.load(),
                simulated_prior[sim_plot_vars].load(),
                simulated_posterior.sel({"filtered_by": filter_var})[
                    sim_plot_vars
                ].load(),
                filter_var=filter_var,
                filter_range=filter_range,
                figsize=(3.2, 2.6),
                fig_dir=fig_dir,
                fontsize=5,
                fudge_factor=fudge_factor,
                percentiles=ci,
                plot_vars=["cumulative_mass_flux", "grounding_line_flux"],
                reference_date=reference_date,
                config=config,
            )

        Path(fig_dir).mkdir(exist_ok=True)
        plot_dir = fig_dir / Path("basin_cumulative_violins")
        plot_dir.mkdir(parents=True, exist_ok=True)
        pdf_dir = plot_dir / Path("pdfs")
        pdf_dir.mkdir(parents=True, exist_ok=True)
        png_dir = plot_dir / Path("pngs")
        png_dir.mkdir(parents=True, exist_ok=True)

        # Apply the functions to the corresponding columns
        for col, functions in column_function_mapping.items():
            for func in functions:
                prior_posterior[col] = prior_posterior[col].apply(func)

        prior_posterior.to_parquet(
            data_dir
            / Path(
                f"""prior_posterior_retreat_{retreat_method}_{filter_range[0]}-{filter_range[1]}.parquet"""
            )
        )

        if "frontal_melt.routing.parameter_a" in prior_posterior.columns:
            prior_posterior["frontal_melt.routing.parameter_a"] *= 10**4
        if "ocean.th.gamma_T" in prior_posterior.columns:
            prior_posterior["ocean.th.gamma_T"] *= 10**4
        if "calving.vonmises_calving.sigma_max" in prior_posterior.columns:
            prior_posterior["calving.vonmises_calving.sigma_max"] *= 10**-3

        for col in [
            "basin",
            "geometry.front_retreat.prescribed.file",
            "ocean.th.file",
            "surface.given.file",
            "ensemble",
            "filtered_by",
            "retreat_method",
        ]:
            if col in prior_posterior.columns:
                prior_posterior[col] = prior_posterior[col].astype("category")

        plot_prior_posteriors(
            prior_posterior.rename(columns=plot_params),
            x_order=plot_params.values(),
            fig_dir=fig_dir,
            bins_dict=short_bins_dict,
        )

        p_df = prior_posterior
        p_df["retreat_method"] = retreat_method
        pp_retreat_list.append(p_df)
        simulated_posterior["retreat_method"] = [retreat_method]
        posterior_ds_dict[retreat_method] = simulated_posterior

    retreat_df = pd.concat(pp_retreat_list).reset_index(drop=True)

    # Sensitivity Analysis
    prior_config = prp.filter_config(simulated["pism_config"], params)
    prior_df = prp.config_to_dataframe(prior_config, ensemble="Prior")
    params_df = prp.convert_category_to_integer(prior_df).drop(columns=["aux_id"])

    sensitivity_indices_list = []
    for basin_group, intersection, filtering_vars in zip(
        [simulated_basins],
        [intersection],
        [["mass_balance", "grounding_line_flux"]],
    ):
        sobol_response = basin_group
        sobol_input_df = params_df[params_df["basin"].isin(intersection)]

        sensitivity_indices_list.append(
            run_sensitivity_analysis(
                sobol_input_df,
                sobol_response,
                filtering_vars,
                notebook=notebook,
            )
        )

    sensitivity_indices = xr.concat(sensitivity_indices_list, dim="basin")
    si_dir = result_dir / Path("sensitivity_indices")
    si_dir.mkdir(parents=True, exist_ok=True)
    sensitivity_indices.to_netcdf(si_dir / Path("sensitivity_indices.nc"))

    sensitivity_indices = prp.add_prefix_coord(
        sensitivity_indices, parameter_categories
    )

    # Group by the new coordinate and compute the sum for each group
    indices_vars = [v for v in sensitivity_indices.data_vars if "_conf" not in v]
    aggregated_indices = (
        sensitivity_indices[indices_vars].groupby("sensitivity_indices_group").sum()
    )
    # Group by the new coordinate and compute the sum the squares for each group
    # then take the root.
    indices_conf = [v for v in sensitivity_indices.data_vars if "_conf" in v]
    aggregated_conf = (
        sensitivity_indices[indices_conf]
        .apply(np.square)
        .groupby("sensitivity_indices_group")
        .sum()
        .apply(np.sqrt)
    )
    aggregated = xr.merge([aggregated_indices, aggregated_conf])
    aggregated.to_netcdf(si_dir / Path("aggregated_sensitivity_indices.nc"))

    for indices_var, indices_conf_var in zip(indices_vars, indices_conf):
        for basin in aggregated.basin.values:
            for filter_var in aggregated.filtered_by.values:
                plot_sensitivity_indices(
                    aggregated.sel(basin=basin, filtered_by=filter_var)
                    .rolling({"time": 13})
                    .mean()
                    .sel(time=slice("1980-01-01", "2020-01-01")),
                    indices_var=indices_var,
                    indices_conf_var=indices_conf_var,
                    basin=basin,
                    filter_var=filter_var,
                    fig_dir=fig_dir,
                )
