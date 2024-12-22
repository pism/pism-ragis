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

# pylint: disable=unused-import,too-many-positional-arguments

"""
Analyze RAGIS ensemble.
"""

import copy
import json
import time
import warnings
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from functools import wraps
from importlib.resources import files
from itertools import chain
from pathlib import Path
from typing import Any, Callable, Dict, Hashable, List, Mapping, Union

import dask
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import toml
import xarray as xr
from dask.diagnostics import ProgressBar
from dask.distributed import Client, progress
from joblib import Parallel, delayed
from tqdm.auto import tqdm

import pism_ragis.processing as prp
from pism_ragis.analyze import delta_analysis, sobol_analysis
from pism_ragis.decorators import profileit, timeit
from pism_ragis.filtering import filter_outliers, run_importance_sampling
from pism_ragis.logger import get_logger
from pism_ragis.plotting import (
    plot_basins,
    plot_outliers,
    plot_prior_posteriors,
    plot_sensitivity_indices,
)
from pism_ragis.processing import config_to_dataframe, filter_config

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


def sort_columns(df: pd.DataFrame, sorted_columns: list) -> pd.DataFrame:
    """
    Sort columns of a DataFrame.

    This function sorts the columns of a DataFrame such that the columns specified in
    `sorted_columns` appear in the specified order, while all other columns appear before
    the sorted columns in their original order.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame to be sorted.
    sorted_columns : list
        A list of column names to be sorted.

    Returns
    -------
    pd.DataFrame
        The DataFrame with columns sorted as specified.
    """
    # Identify columns that are not in the list
    other_columns = [col for col in df.columns if col not in sorted_columns]

    # Concatenate other columns with the sorted columns
    new_column_order = other_columns + sorted_columns

    # Reindex the DataFrame
    return df.reindex(columns=new_column_order)


def add_prefix_coord(
    sensitivity_indices: xr.Dataset, parameter_groups: Dict
) -> xr.Dataset:
    """
    Add prefix coordinates to an xarray Dataset.

    This function extracts the prefix from each coordinate value in the 'pism_config_axis'
    and adds it as a new coordinate. It also maps the prefixes to their corresponding
    sensitivity indices groups.

    Parameters
    ----------
    sensitivity_indices : xr.Dataset
        The input dataset containing sensitivity indices.
    parameter_groups : Dict
        A dictionary mapping parameter names to their corresponding groups.

    Returns
    -------
    xr.Dataset
        The dataset with added prefix coordinates and sensitivity indices groups.
    """
    prefixes = [
        name.split(".")[0] for name in sensitivity_indices.pism_config_axis.values
    ]

    sensitivity_indices = sensitivity_indices.assign_coords(
        prefix=("pism_config_axis", prefixes)
    )
    si_prefixes = [parameter_groups[name] for name in sensitivity_indices.prefix.values]

    sensitivity_indices = sensitivity_indices.assign_coords(
        sensitivity_indices_group=("pism_config_axis", si_prefixes)
    )
    return sensitivity_indices


def prepare_input(
    df: pd.DataFrame,
    params: List[str] = [
        "surface.given.file",
        "ocean.th.file",
        "calving.rate_scaling.file",
        "geometry.front_retreat.prescribed.file",
    ],
) -> pd.DataFrame:
    """
    Prepare the input DataFrame by converting columns to numeric and mapping unique values to integers.

    This function processes the input DataFrame by converting specified columns to numeric values,
    dropping specified columns, and mapping unique values in the specified parameters to integers.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame to be processed.
    params : List[str], optional
        A list of column names to be processed. Unique values in these columns will be mapped to integers.
        By default, the list includes:
        ["surface.given.file", "ocean.th.file", "calving.rate_scaling.file", "geometry.front_retreat.prescribed.file"].

    Returns
    -------
    pd.DataFrame
        The processed DataFrame with specified columns converted to numeric and unique values mapped to integers.

    Examples
    --------
    >>> df = pd.DataFrame({
    ...     "surface.given.file": ["file1", "file2", "file1"],
    ...     "ocean.th.file": ["fileA", "fileB", "fileA"],
    ...     "calving.rate_scaling.file": ["fileX", "fileY", "fileX"],
    ...     "geometry.front_retreat.prescribed.file": ["fileM", "fileN", "fileM"],
    ...     "ensemble": [1, 2, 3],
    ...     "exp_id": [101, 102, 103]
    ... })
    >>> prepare_input(df)
       surface.given.file  ocean.th.file  calving.rate_scaling.file  geometry.front_retreat.prescribed.file
    0                   0              0                          0                                      0
    1                   1              1                          1                                      1
    2                   0              0                          0                                      0
    """
    df = df.apply(prp.convert_column_to_numeric).drop(
        columns=["ensemble", "exp_id"], errors="ignore"
    )

    for param in params:
        m_dict: Dict[str, int] = {v: k for k, v in enumerate(df[param].unique())}
        df[param] = df[param].map(m_dict)

    return df


@timeit
def prepare_simulations(
    filenames: List[Union[Path, str]],
    config: Dict,
    reference_date: str,
    parallel: bool = True,
    engine: str = "h5netcdf",
) -> xr.Dataset:
    """
    Prepare simulations by loading and processing ensemble datasets.

    This function loads ensemble datasets from the specified filenames, processes them
    according to the provided configuration, and returns the processed dataset. The
    processing steps include sorting, converting byte strings to strings, dropping NaNs,
    standardizing variable names, calculating cumulative variables, and normalizing
    cumulative variables.

    Parameters
    ----------
    filenames : List[Union[Path, str]]
        A list of file paths to the ensemble datasets.
    config : Dict
        A dictionary containing configuration settings for processing the datasets.
    reference_date : str
        The reference date for normalizing cumulative variables.
    parallel : bool, optional
        Whether to load the datasets in parallel, by default True.
    engine : str, optional
        The engine to use for loading the datasets, by default "h5netcdf".

    Returns
    -------
    xr.Dataset
        The processed xarray dataset.

    Examples
    --------
    >>> filenames = ["file1.nc", "file2.nc"]
    >>> config = {
    ...     "PISM Spatial": {...},
    ...     "Cumulative Variables": {
    ...         "cumulative_grounding_line_flux": "cumulative_gl_flux",
    ...         "cumulative_smb": "cumulative_smb_flux"
    ...     },
    ...     "Flux Variables": {
    ...         "grounding_line_flux": "gl_flux",
    ...         "smb_flux": "smb_flux"
    ...     }
    ... }
    >>> reference_date = "2000-01-01"
    >>> ds = prepare_simulations(filenames, config, reference_date)
    """
    ds = prp.load_ensemble(filenames, parallel=parallel, engine=engine).sortby("basin")
    ds = xr.apply_ufunc(np.vectorize(convert_bstrings_to_str), ds, dask="parallelized")

    ds = prp.standardize_variable_names(ds, config["PISM Spatial"])
    ds[config["Cumulative Variables"]["cumulative_grounding_line_flux"]] = ds[
        config["Flux Variables"]["grounding_line_flux"]
    ].cumsum() / len(ds.time)
    ds[config["Cumulative Variables"]["cumulative_smb"]] = ds[
        config["Flux Variables"]["smb_flux"]
    ].cumsum() / len(ds.time)
    ds = prp.normalize_cumulative_variables(
        ds,
        list(config["Cumulative Variables"].values()),
        reference_date=reference_date,
    )
    return ds


@timeit
def prepare_observations(
    basin_url: Union[Path, str],
    grace_url: Union[Path, str],
    config: Dict,
    reference_date: str,
    engine: str = "h5netcdf",
) -> tuple:
    """
    Prepare observation datasets by normalizing cumulative variables.

    This function loads observation datasets from the specified URLs, sorts them by basin,
    normalizes the cumulative variables, and returns the processed datasets.

    Parameters
    ----------
    basin_url : Union[Path, str]
        The URL or path to the basin observation dataset.
    grace_url : Union[Path, str]
        The URL or path to the GRACE observation dataset.
    config : Dict
        A dictionary containing configuration settings for processing the datasets.
    reference_date : str
        The reference date for normalizing cumulative variables.
    engine : str, optional
        The engine to use for loading the datasets, by default "h5netcdf".

    Returns
    -------
    tuple
        A tuple containing the processed basin and GRACE observation datasets.

    Examples
    --------
    >>> config = {
    ...     "Cumulative Variables": {"cumulative_mass_balance": "mass_balance"},
    ...     "Cumulative Uncertainty Variables": {"cumulative_mass_balance_uncertainty": "mass_balance_uncertainty"}
    ... }
    >>> prepare_observations("basin.nc", "grace.nc", config, "2000-01-1")
    (<xarray.Dataset>, <xarray.Dataset>)
    """
    obs_basin = xr.open_dataset(basin_url, engine=engine, chunks=-1)
    obs_basin = obs_basin.sortby("basin")

    cumulative_vars = config["Cumulative Variables"]
    cumulative_uncertainty_vars = config["Cumulative Uncertainty Variables"]

    obs_basin = prp.normalize_cumulative_variables(
        obs_basin,
        list(cumulative_vars.values()) + list(cumulative_uncertainty_vars.values()),
        reference_date,
    )

    obs_grace = xr.open_dataset(grace_url, engine=engine, chunks=-1)
    obs_grace = obs_grace.sortby("basin")

    cumulative_vars = config["Cumulative Variables"]["cumulative_mass_balance"]
    cumulative_uncertainty_vars = config["Cumulative Uncertainty Variables"][
        "cumulative_mass_balance_uncertainty"
    ]

    obs_grace = prp.normalize_cumulative_variables(
        obs_grace,
        [cumulative_vars] + [cumulative_uncertainty_vars],
        reference_date,
    )

    return obs_basin, obs_grace


def convert_bstrings_to_str(element: Any) -> Any:
    """
    Convert byte strings to regular strings.

    Parameters
    ----------
    element : Any
        The element to be checked and potentially converted. If the element is a byte string,
        it will be converted to a regular string. Otherwise, the element will be returned as is.

    Returns
    -------
    Any
        The converted element if it was a byte string, otherwise the original element.
    """
    if isinstance(element, bytes):
        return element.decode("utf-8")
    return element


@timeit
def run_sensitivity_analysis(
    input_df: pd.DataFrame,
    response_ds: xr.Dataset,
    filter_vars: List[str],
    group_dim: str = "basin",
    iter_dim: str = "time",
    notebook: bool = False,
) -> xr.Dataset:
    """
    Run delta sensitivity analysis on the given dataset.

    This function calculates sensitivity indices for each basin in the dataset,
    filtered by the specified variables. It uses Dask for parallel processing
    to improve performance.

    Parameters
    ----------
    input_df : pd.DataFrame
        DataFrame containing ensemble information, with a 'basin' column to group by.
    response_ds : xr.Dataset
        The input dataset containing the data to be analyzed.
    filter_vars : List[str]
        List of variables to filter by for sensitivity analysis.
    group_dim : str, optional
        The dimension to group by, by default "basin".
    iter_dim : str, optional
        The dimension to iterate over, by default "time".
    notebook : bool, optional
        Whether to display a nicer progress bar when running in a notebook, by default False.

    Returns
    -------
    xr.Dataset
        A dataset containing the calculated sensitivity indices for each basin and filter variable.

    Notes
    -----
    It is imperative to load the dataset before starting the Dask client,
    to avoid each Dask worker loading the dataset separately, which would
    significantly slow down the computation.
    """

    print("Calculating Sensitivity Indices")
    print("===============================")

    client = Client()
    print(f"Open client in browser: {client.dashboard_link}")
    sensitivity_indices_list = []
    for gdim, df in input_df.groupby(by=group_dim):
        df = df.drop(columns=[group_dim])
        problem = {
            "num_vars": len(df.columns),
            "names": df.columns,  # Parameter names
            "bounds": zip(
                df.min().values,
                df.max().values,
            ),  # Parameter bounds
        }
        for filter_var in filter_vars:
            print(
                f"  ...sensitivity indices for basin {gdim} filtered by {filter_var} ",
            )

            responses = response_ds.sel({"basin": gdim})[filter_var].load()
            responses_scattered = client.scatter(
                [
                    responses.isel({"time": k}).to_numpy()
                    for k in range(len(responses[iter_dim]))
                ]
            )

            futures = client.map(
                delta_analysis,
                responses_scattered,
                X=df.to_numpy(),
                problem=problem,
            )
            progress(futures, notebook=notebook)
            result = client.gather(futures)

            sensitivity_indices = xr.concat(
                [r.expand_dims(iter_dim) for r in result], dim=iter_dim
            )
            sensitivity_indices[iter_dim] = responses[iter_dim]
            sensitivity_indices = sensitivity_indices.expand_dims(group_dim, axis=1)
            sensitivity_indices[group_dim] = [gdim]
            sensitivity_indices = sensitivity_indices.expand_dims("filtered_by", axis=2)
            sensitivity_indices["filtered_by"] = [filter_var]
            sensitivity_indices_list.append(sensitivity_indices)

    all_sensitivity_indices: xr.Dataset = xr.merge(sensitivity_indices_list)
    client.close()

    return all_sensitivity_indices


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
        "--mankoff_url",
        help="""Path to "observed" Mankoff mass balance.""",
        type=str,
        default="data/mass_balance/mankoff_greenland_mass_balance.nc",
    )
    parser.add_argument(
        "--grace_url",
        help="""Path to "observed" GRACE mass balance.""",
        type=str,
        default="data/mass_balance/grace_greenland_mass_balance.nc",
    )
    parser.add_argument(
        "--engine",
        help="""Engine for xarray. Default="h5netcdf".""",
        type=str,
        default="h5netcdf",
    )
    parser.add_argument(
        "--filter_range",
        help="""Time slice used for Importance Sampling, needs an integer year. Default="1986 2019". """,
        type=int,
        nargs=2,
        default=[1990, 2019],
    )
    parser.add_argument(
        "--outlier_range",
        help="""Ensemble members outside this range are removed. Default="-1_250 250". """,
        type=float,
        nargs=2,
        default=[-10000.0, 0.0],
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
        type=float,
        default=3.0,
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
        default="2020-01-1",
    )
    parser.add_argument(
        "--retreat_method",
        help="""Sub-select retreat method. Default='all'.""",
        type=str,
        choices=["all", "free", "prescribed"],
        default="all",
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
    engine = options.engine
    filter_range = options.filter_range
    fudge_factor = options.fudge_factor
    notebook = options.notebook
    parallel = options.parallel
    reference_date = options.reference_date
    resampling_frequency = options.resampling_frequency
    retreat_method = options.retreat_method
    outlier_variable = options.outlier_variable
    outlier_range = options.outlier_range
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
    data_dir = result_dir / Path("posteriors")
    data_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = result_dir / Path("figures")
    fig_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = fig_dir / Path("basin_timeseries")
    plot_dir.mkdir(parents=True, exist_ok=True)
    pdf_dir = plot_dir / Path("pdfs")
    pdf_dir.mkdir(parents=True, exist_ok=True)
    png_dir = plot_dir / Path("pngs")
    png_dir.mkdir(parents=True, exist_ok=True)

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

    plt.rcParams.update(rcparams)

    simulated_ds = prepare_simulations(
        basin_files, config, reference_date, parallel=parallel, engine=engine
    )

    observed_mankoff_ds, observed_grace_ds = prepare_observations(
        options.mankoff_url,
        options.grace_url,
        config,
        reference_date,
        engine=engine,
    )

    # Select the relevant pism_config_axis
    retreat = simulated_ds.sel(
        pism_config_axis="geometry.front_retreat.prescribed.file"
    ).compute()

    if retreat_method == "free":
        retreat_exp_ids = retreat.where(
            retreat["pism_config"] == "false", drop=True
        ).exp_id.values
    elif retreat_method == "prescribed":
        retreat_exp_ids = retreat.where(
            retreat["pism_config"] != "false", drop=True
        ).exp_id.values
    else:
        retreat_exp_ids = simulated_ds.exp_id

    # Select the Dataset with the filtered exp_ids
    simulated_ds = simulated_ds.sel(exp_id=retreat_exp_ids)

    filtered_ds, outliers_ds = filter_outliers(
        simulated_ds,
        outlier_range=outlier_range,
        outlier_variable=outlier_variable,
        subset={"basin": "GIS"},
    )

    plot_outliers(
        filtered_ds.sel(basin="GIS")[outlier_variable],
        outliers_ds.sel(basin="GIS")[outlier_variable],
        Path(pdf_dir) / Path(f"{outlier_variable}_filtering.pdf"),
    )

    outliers_config = filter_config(outliers_ds, params)
    outliers_df = config_to_dataframe(outliers_config, ensemble="Outliers")

    filtered_config = filter_config(filtered_ds, params)
    filtered_df = config_to_dataframe(filtered_config, ensemble="Filtered")

    obs_mankoff_basins = set(observed_mankoff_ds.basin.values)
    obs_grace_basins = set(observed_grace_ds.basin.values)

    simulated = filtered_ds

    sim_basins = set(simulated.basin.values)
    sim_grace = set(simulated.basin.values)

    intersection_mankoff = list(sim_basins.intersection(obs_mankoff_basins))
    intersection_grace = list(sim_grace.intersection(obs_grace_basins))

    observed_mankoff_basins_ds = observed_mankoff_ds.sel(
        {"basin": intersection_mankoff}
    )
    simulated_mankoff_basins_ds = simulated.sel({"basin": intersection_mankoff})

    observed_mankoff_basins_resampled_ds = observed_mankoff_basins_ds.resample(
        {"time": resampling_frequency}
    ).mean()
    simulated_mankoff_basins_resampled_ds = simulated_mankoff_basins_ds.resample(
        {"time": resampling_frequency}
    ).mean()

    observed_grace_basins_ds = observed_grace_ds.sel({"basin": intersection_grace})
    simulated_grace_basins_ds = simulated.sel({"basin": intersection_grace})

    observed_grace_basins_resampled_ds = observed_grace_basins_ds.resample(
        {"time": resampling_frequency}
    ).mean()
    simulated_grace_basins_resampled_ds = simulated_grace_basins_ds.resample(
        {"time": resampling_frequency}
    ).mean()

    obs_mean_vars_mankoff: List[str] = ["grounding_line_flux", "mass_balance"]
    obs_std_vars_mankoff: List[str] = [
        "grounding_line_flux_uncertainty",
        "mass_balance_uncertainty",
    ]
    sim_vars_mankoff: List[str] = ["grounding_line_flux", "mass_balance"]

    sim_plot_vars = (
        [ragis_config["Cumulative Variables"]["cumulative_mass_balance"]]
        + list(ragis_config["Flux Variables"].values())
        + ["ensemble"]
    )

    prior_posterior_mankoff, simulated_prior_mankoff, simulated_posterior_mankoff = (
        run_importance_sampling(
            observed=observed_mankoff_basins_resampled_ds,
            simulated=simulated_mankoff_basins_resampled_ds,
            obs_mean_vars=obs_mean_vars_mankoff,
            obs_std_vars=obs_std_vars_mankoff,
            sim_vars=sim_vars_mankoff,
            filter_range=filter_range,
            fudge_factor=fudge_factor,
            params=params,
        )
    )

    for filter_var in obs_mean_vars_mankoff:
        plot_basins(
            observed_mankoff_basins_resampled_ds,
            simulated_prior_mankoff[sim_plot_vars],
            simulated_posterior_mankoff.sel({"filtered_by": filter_var})[sim_plot_vars],
            filter_var=filter_var,
            filter_range=filter_range,
            fig_dir=fig_dir,
            config=config,
        )

    obs_mean_vars_grace: List[str] = ["mass_balance"]
    obs_std_vars_grace: List[str] = [
        "mass_balance_uncertainty",
    ]
    sim_vars_grace: List[str] = ["mass_balance"]

    prior_posterior_grace, simulated_prior_grace, simulated_posterior_grace = (
        run_importance_sampling(
            observed=observed_grace_basins_resampled_ds,
            simulated=simulated_grace_basins_resampled_ds,
            obs_mean_vars=obs_mean_vars_grace,
            obs_std_vars=obs_std_vars_grace,
            sim_vars=sim_vars_grace,
            fudge_factor=fudge_factor,
            filter_range=filter_range,
            params=params,
        )
    )

    for filter_var in obs_mean_vars_grace:
        plot_basins(
            observed_grace_basins_resampled_ds,
            simulated_prior_grace[sim_plot_vars],
            simulated_posterior_grace.sel({"filtered_by": filter_var})[sim_plot_vars],
            filter_var=filter_var,
            filter_range=filter_range,
            fig_dir=fig_dir,
            config=config,
        )

    prior_posterior = pd.concat(
        [prior_posterior_mankoff, prior_posterior_grace]
    ).reset_index(drop=True)

    column_function_mapping: Dict[str, List[Callable]] = {
        "surface.given.file": [prp.simplify_path, prp.simplify_climate],
        "ocean.th.file": [prp.simplify_path, prp.simplify_ocean],
        "calving.rate_scaling.file": [prp.simplify_path, prp.simplify_calving],
        "geometry.front_retreat.prescribed.file": [prp.simplify_retreat],
    }

    # Apply the functions to the corresponding columns
    for col, functions in column_function_mapping.items():
        for func in functions:
            prior_posterior[col] = prior_posterior[col].apply(func)

    prior_posterior.to_parquet(
        data_dir
        / Path(f"""prior_posterior_{filter_range[0]}-{filter_range[1]}.parquet""")
    )

    bins_dict = config["Posterior Bins"]
    parameter_catetories = config["Parameter Categories"]

    params_sorted_by_category: dict = {
        group: [] for group in sorted(parameter_catetories.values())
    }
    for param in params:
        prefix = param.split(".")[0]
        if prefix in parameter_catetories:
            group = parameter_catetories[prefix]
            if param not in params_sorted_by_category[group]:
                params_sorted_by_category[group].append(param)

    params_sorted_list = list(chain(*params_sorted_by_category.values()))
    if "frontal_melt.routing.parameter_a" in prior_posterior.columns:
        prior_posterior["frontal_melt.routing.parameter_a"] *= 10**4
    if "ocean.th.gamma_T" in prior_posterior.columns:
        prior_posterior["ocean.th.gamma_T"] *= 10**5
    if "calving.vonmises_calving.sigma_max" in prior_posterior.columns:
        prior_posterior["calving.vonmises_calving.sigma_max"] *= 10**-3
    prior_posterior_sorted = sort_columns(prior_posterior, params_sorted_list)

    params_sorted_dict = {k: params_short_dict[k] for k in params_sorted_list}
    plot_prior_posteriors(
        prior_posterior_sorted.rename(columns=params_sorted_dict),
        fig_dir=fig_dir,
    )

    prior_config = filter_config(simulated.isel({"time": 0}), params)
    prior_df = config_to_dataframe(prior_config, ensemble="Prior")
    params_df = prepare_input(prior_df)

    sensitivity_indices_list = []
    for basin_group, intersection, filtering_vars in zip(
        [simulated_grace_basins_ds, simulated_mankoff_basins_ds],
        [intersection_grace, intersection_mankoff],
        [["mass_balance"], ["mass_balance", "grounding_line_flux"]],
    ):
        sobol_response_ds = basin_group
        sobol_input_df = params_df[params_df["basin"].isin(intersection)]

        sensitivity_indices_list.append(
            run_sensitivity_analysis(
                sobol_input_df,
                sobol_response_ds,
                filtering_vars,
                notebook=notebook,
            )
        )

    sensitivity_indices = xr.concat(sensitivity_indices_list, dim="basin")
    si_dir = result_dir / Path("sensitivity_indices")
    si_dir.mkdir(parents=True, exist_ok=True)
    sensitivity_indices.to_netcdf(si_dir / Path("sensitivity_indices.nc"))

    sensitivity_indices = add_prefix_coord(sensitivity_indices, parameter_catetories)

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
    aggregated_ds = xr.merge([aggregated_indices, aggregated_conf])
    aggregated_ds.to_netcdf(si_dir / Path("aggregated_sensitivity_indices.nc"))

    for indices_var, indices_conf_var in zip(indices_vars, indices_conf):
        for basin in aggregated_ds.basin.values:
            for filter_var in aggregated_ds.filtered_by.values:
                plot_sensitivity_indices(
                    aggregated_ds.sel(basin=basin, filtered_by=filter_var)
                    .rolling({"time": 13})
                    .mean()
                    .sel(time=slice("1980-01-01", "2020-01-01")),
                    indices_var=indices_var,
                    indices_conf_var=indices_conf_var,
                    basin=basin,
                    filter_var=filter_var,
                    fig_dir=fig_dir,
                )
