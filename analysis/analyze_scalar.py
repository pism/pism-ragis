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
from pathlib import Path
from typing import Any, Callable, Dict, Hashable, List, Mapping, Union

import dask
import matplotlib
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
from pism_ragis.filtering import filter_outliers, importance_sampling
from pism_ragis.likelihood import log_normal
from pism_ragis.logger import get_logger
from pism_ragis.plotting import (
    plot_basins,
    plot_outliers,
    plot_prior_posteriors,
    plot_sensitivity_indices,
)

logger = get_logger("pism_ragis")

matplotlib.use("Agg")
xr.set_options(keep_attrs=True)
plt.style.use("tableau-colorblind10")
# Ignore specific RuntimeWarnings
warnings.filterwarnings(
    "ignore", category=RuntimeWarning, message="overflow encountered in exp"
)
warnings.filterwarnings(
    "ignore", category=RuntimeWarning, message="invalid value encountered in divide"
)


def add_prefix_coord(sensitivity_indices: xr.Dataset, parameter_groups: Dict):
    """
    Add prefix coordinates to xr.Dataset.
    """
    # Extract the prefix from each coordinate value
    prefixes = [
        name.split(".")[0] for name in sensitivity_indices.pism_config_axis.values
    ]

    # Add the prefixes as a new coordinate
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
        ["surface.given.file", "ocean.th.file", "calving.rate_scaling.file", "geometry.front_retreat.prescribed.file"]

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
    # Convert columns to numeric and drop specified columns
    df = df.apply(prp.convert_column_to_numeric).drop(
        columns=["ensemble", "exp_id"], errors="ignore"
    )

    # Map unique values in specified parameters to integers
    for param in params:
        m_dict: Dict[str, int] = {v: k for k, v in enumerate(df[param].unique())}
        df[param] = df[param].map(m_dict)

    return df


@timeit
def run_sampling(
    observed: xr.Dataset,
    simulated: xr.Dataset,
    obs_mean_vars: List[str] = ["grounding_line_flux", "mass_balance"],
    obs_std_vars: List[str] = [
        "grounding_line_flux_uncertainty",
        "mass_balance_uncertainty",
    ],
    sim_vars: List[str] = ["grounding_line_flux", "mass_balance"],
    filter_range: List[int] = [1990, 2019],
    fudge_factor: float = 3.0,
    fig_dir: Union[str, Path] = "figures",
    params: List[str] = [],
    config: Dict = {},
) -> pd.DataFrame:
    """
    Run sampling to process observed and simulated datasets.

    This function performs importance sampling using the specified observed and simulated datasets,
    processes the results, and returns a DataFrame with the prior and posterior configurations.

    Parameters
    ----------
    observed : xr.Dataset
        The observed dataset.
    simulated : xr.Dataset
        The simulated dataset.
    obs_mean_vars : List[str], optional
        A list of variable names for the observed mean values, by default ["grounding_line_flux", "mass_balance"].
    obs_std_vars : List[str], optional
        A list of variable names for the observed standard deviation values, by default ["grounding_line_flux_uncertainty", "mass_balance_uncertainty"].
    sim_vars : List[str], optional
        A list of variable names for the simulated values, by default ["grounding_line_flux", "mass_balance"].
    filter_range : List[int], optional
        A list containing the start and end years for filtering, by default [1990, 2019].
    fudge_factor : float, optional
        A fudge factor for the importance sampling, by default 3.0.
    fig_dir : Union[str, Path], optional
        The directory where figures will be saved, by default "figures".
    params : List[str], optional
        A list of parameter names to be used for filtering configurations, by default [].
    config : Dict, optional
        A dictionary containing configuration settings for the RAGIS model, by default {}.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the prior and posterior configurations.

    Examples
    --------
    >>> observed = xr.Dataset(...)
    >>> simulated = xr.Dataset(...)
    >>> result = run_sampling(observed, simulated)
    """
    filter_start_year, filter_end_year = filter_range

    simulated_prior = simulated
    simulated_prior["ensemble"] = "Prior"

    prior_config = filter_config(simulated.isel({"time": 0}), params)
    prior_df = config_to_dataframe(prior_config, ensemble="Prior")
    prior_posterior_list = []

    for obs_mean_var, obs_std_var, sim_var in zip(
        obs_mean_vars, obs_std_vars, sim_vars
    ):
        print(f"Importance sampling using {obs_mean_var}")
        f = importance_sampling(
            simulated=simulated.sel(
                time=slice(str(filter_start_year), str(filter_end_year))
            ),
            observed=observed.sel(
                time=slice(str(filter_start_year), str(filter_end_year))
            ),
            log_likelihood=log_normal,
            fudge_factor=fudge_factor,
            n_samples=len(simulated.exp_id),
            obs_mean_var=obs_mean_var,
            obs_std_var=obs_std_var,
            sim_var=sim_var,
        )

        with ProgressBar() as pbar:
            result = f.compute()
            logger.info(
                "Importance Sampling: Finished in %2.2f seconds", pbar.last_duration
            )

        importance_sampled_ids = result["exp_id_sampled"]
        importance_sampled_ids["basin"] = importance_sampled_ids["basin"].astype(str)

        simulated_posterior = simulated.sel(exp_id=importance_sampled_ids)
        simulated_posterior["ensemble"] = "Posterior"

        posterior_config = filter_config(simulated_posterior.isel({"time": 0}), params)
        posterior_df = config_to_dataframe(posterior_config, ensemble="Posterior")

        prior_posterior_f = pd.concat([prior_df, posterior_df]).reset_index(drop=True)
        prior_posterior_f["filtered_by"] = obs_mean_var
        prior_posterior_list.append(prior_posterior_f)

        plot_basins(
            observed,
            simulated_prior,
            simulated_posterior,
            obs_mean_var,
            filter_range=filter_range,
            fig_dir=fig_dir,
            config=config,
        )
    prior_posterior = pd.concat(prior_posterior_list).reset_index(drop=True)
    prior_posterior = prior_posterior.apply(prp.convert_column_to_numeric)
    return prior_posterior


def filter_config(ds: xr.Dataset, params: List[str]) -> xr.DataArray:
    """
    Filter the configuration parameters from the dataset.

    This function selects the specified configuration parameters from the dataset
    and returns them as a DataArray.

    Parameters
    ----------
    ds : xr.Dataset
        The input dataset containing the configuration parameters.
    params : List[str]
        A list of configuration parameter names to be selected.

    Returns
    -------
    xr.DataArray
        The selected configuration parameters as a DataArray.

    Examples
    --------
    >>> ds = xr.Dataset({'pism_config': (('pism_config_axis',), [1, 2, 3])},
                        coords={'pism_config_axis': ['param1', 'param2', 'param3']})
    >>> filter_config(ds, ['param1', 'param3'])
    <xarray.DataArray 'pism_config' (pism_config_axis: 2)>
    array([1, 3])
    Coordinates:
      * pism_config_axis  (pism_config_axis) <U6 'param1' 'param3'
    """
    config = ds.sel(pism_config_axis=params).pism_config
    return config


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


@timeit
def config_to_dataframe(
    config: xr.DataArray, ensemble: Union[str, None] = None
) -> pd.DataFrame:
    """
    Convert an xarray DataArray configuration to a pandas DataFrame.

    This function converts the input DataArray containing configuration data into a
    pandas DataFrame. The dimensions of the DataArray (excluding 'pism_config_axis')
    are used as the index, and the 'pism_config_axis' values are used as columns.

    Parameters
    ----------
    config : xr.DataArray
        The input DataArray containing the configuration data.
    ensemble : Union[str, None], optional
        An optional string to add as a column named 'ensemble' in the DataFrame, by default None.

    Returns
    -------
    pd.DataFrame
        A DataFrame where the dimensions of the DataArray (excluding 'pism_config_axis')
        are used as the index, and the 'pism_config_axis' values are used as columns.

    Examples
    --------
    >>> config = xr.DataArray(
    ...     data=[[1, 2, 3], [4, 5, 6]],
    ...     dims=["time", "pism_config_axis"],
    ...     coords={"time": [0, 1], "pism_config_axis": ["param1", "param2", "param3"]}
    ... )
    >>> df = config_to_dataframe(config)
    >>> print(df)
    pism_config_axis  time  param1  param2  param3
    0                   0       1       2       3
    1                   1       4       5       6

    >>> df = config_to_dataframe(config, ensemble="ensemble1")
    >>> print(df)
    pism_config_axis  time  param1  param2  param3   ensemble
    0                   0       1       2       3  ensemble1
    1                   1       4       5       6  ensemble1
    """
    dims = [dim for dim in config.dims if dim != "pism_config_axis"]
    df = config.to_dataframe().reset_index()
    df = df.pivot(index=dims, columns="pism_config_axis", values="pism_config")
    df.reset_index(inplace=True)
    if ensemble:
        df["ensemble"] = ensemble
    return df


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
    __spec__ = None

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

    plt.rcParams["font.size"] = 6

    flux_vars = config["Flux Variables"]
    flux_uncertainty_vars = {
        k + "_uncertainty": v + "_uncertainty" for k, v in flux_vars.items()
    }

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

    prior_posterior_mankoff = run_sampling(
        observed=observed_mankoff_basins_resampled_ds,
        simulated=simulated_mankoff_basins_resampled_ds,
        filter_range=filter_range,
        fudge_factor=fudge_factor,
        params=params,
        config=config,
        fig_dir=fig_dir,
    )

    prior_posterior_grace = run_sampling(
        observed=observed_grace_basins_resampled_ds,
        simulated=simulated_grace_basins_resampled_ds,
        obs_mean_vars=["mass_balance"],
        obs_std_vars=["mass_balance_uncertainty"],
        sim_vars=["mass_balance"],
        fudge_factor=10,
        filter_range=filter_range,
        params=params,
        config=config,
        fig_dir=fig_dir,
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
    plot_prior_posteriors(
        prior_posterior.rename(columns=params_short_dict),
        bins_dict,
        fig_dir=fig_dir,
        config=config,
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
        sobol_response_ds = basin_group.sel(time=slice("1980-01-01", "1990-01-01"))
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

    parameter_groups = config["Parameter Groups"]
    sensitivity_indices = add_prefix_coord(sensitivity_indices, parameter_groups)

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
                    aggregated_ds.sel(filtered_by=filter_var)
                    .rolling({"time": 13})
                    .mean(),
                    indices_var=indices_var,
                    indices_conf_var=indices_conf_var,
                    fig_dir=fig_dir,
                )
