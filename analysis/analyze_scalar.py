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

import time
import warnings
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from functools import wraps
from importlib.resources import files
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, Hashable, List, Mapping, Union

import dask
import numpy as np
import pandas as pd
import pylab as plt
import seaborn as sns
import toml
import xarray as xr
from dask.diagnostics import ProgressBar
from dask.distributed import Client, progress
from tqdm.auto import tqdm

import pism_ragis.processing as prp
from pism_ragis.analysis import delta_analysis
from pism_ragis.decorators import profileit, timeit
from pism_ragis.filtering import filter_outliers, importance_sampling
from pism_ragis.likelihood import log_normal
from pism_ragis.logger import get_logger

logger = get_logger("pism_ragis")


xr.set_options(keep_attrs=True)
plt.style.use("tableau-colorblind10")


sim_alpha = 0.5
sim_cmap = sns.color_palette("crest", n_colors=4).as_hex()[0:3:2]
sim_cmap = ["#a6cee3", "#1f78b4"]
sim_cmap = ["#CC6677", "#882255"]
obs_alpha = 1.0
obs_cmap = ["0.8", "0.7"]
# obs_cmap = ["#88CCEE", "#44AA99"]
hist_cmap = ["#a6cee3", "#1f78b4"]


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
    filenames: List[Path | str],
    config: Dict,
    reference_date: str,
    parallel: bool = True,
    engine: str = "netcdf4",
) -> xr.Dataset:
    """
    Prepare simulations by loading and processing ensemble datasets.

    This function loads ensemble datasets from the specified filenames, processes them
    according to the provided configuration, and returns the processed dataset. The
    processing steps include sorting, dropping NaNs, standardizing variable names,
    calculating cumulative variables, and normalizing cumulative variables.

    Parameters
    ----------
    filenames : List[Union[Path, str]]
        A list of file paths to the ensemble datasets.
    config : Dict
        A dictionary containing configuration settings for processing the datasets.
    parallel : bool, optional
        Whether to load the datasets in parallel, by default True.
    engine : str, optional
        The engine to use for loading the datasets, by default "netcdf4".

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
    >>> ds = prepare_simulations(filenames, config)
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
def config_to_dataframe(config: xr.DataArray, ensemble: str | None = None):
    """
    Convert an xarray DataArray configuration to a pandas DataFrame.

    Parameters
    ----------
    config : xr.DataArray
        The input DataArray containing the configuration data.

    Returns
    -------
    pd.DataFrame
        A DataFrame where the dimensions of the DataArray (excluding 'pism_config_axis')
        are used as the index, and the 'pism_config_axis' values are used as columns.
    """
    dims = [dim for dim in config.dims if not dim in ["pism_config_axis"]]
    df = config.to_dataframe().reset_index()
    df = df.pivot(index=dims, columns="pism_config_axis", values="pism_config")
    df.reset_index(inplace=True)
    if ensemble:
        df["Ensemble"] = ensemble
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


def plot_outliers(
    filtered_da: xr.DataArray, outliers_da: xr.DataArray, filename: Path | str
):
    """
    Plot outliers.
    """
    fig, ax = plt.subplots(1, 1)
    if filtered_da.size > 0:
        filtered_da.plot(hue="exp_id", color="k", add_legend=False, ax=ax, lw=0.5)
    if outliers_da.size > 0:
        outliers_da.plot(hue="exp_id", color="r", add_legend=False, ax=ax, lw=0.5)
    fig.savefig(filename)


@timeit
def run_delta_analysis(
    ds: xr.Dataset,
    ensemble_df: pd.DataFrame,
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
    ds : xr.Dataset
        The input dataset containing the data to be analyzed.
    ensemble_df : pd.DataFrame
        DataFrame containing ensemble information, with a 'basin' column to group by.
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

    ds = ds.load()
    client = Client()
    print(f"Open client in browser: {client.dashboard_link}")
    all_delta_indices_list = []
    for gdim, df in ensemble_df.groupby(by=group_dim):
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

            responses = ds.sel({"basin": gdim})[filter_var]
            responses_scattered = client.scatter(
                [
                    responses.isel({"time": k}).to_numpy()
                    for k in range(len(responses[iter_dim]))
                ]
            )

            futures = client.map(
                delta_analysis,
                responses_scattered,
                problem=problem,
                ensemble_df=df,
            )
            progress(futures, notebook=notebook)
            result = client.gather(futures)

            delta_indices = xr.concat(
                [r.expand_dims(iter_dim) for r in result], dim=iter_dim
            )
            delta_indices[iter_dim] = responses[iter_dim]
            delta_indices = delta_indices.expand_dims(group_dim, axis=1)
            delta_indices[group_dim] = [gdim]
            delta_indices = delta_indices.expand_dims("filtered_by", axis=2)
            delta_indices["filtered_by"] = [filter_var]
            all_delta_indices_list.append(delta_indices)

    all_delta_indices: xr.Dataset = xr.merge(all_delta_indices_list)
    client.close()

    return all_delta_indices


def plot_obs_sims(
    obs: xr.Dataset,
    sim_prior: xr.Dataset,
    sim_posterior: xr.Dataset,
    config: dict,
    filtering_var: str,
    filter_range: List[int] = [1990, 2019],
    fig_dir: Union[str, Path] = "figures",
    reference_date: str = "1986-01-1",
    sim_alpha: float = 0.4,
    obs_alpha: float = 1.0,
    sigma: float = 2,
    percentiles: List[float] = [0.025, 0.975],
    fontsize: float = 6,
) -> None:
    """
    Plot figure with cumulative mass balance and grounding line flux and climatic
    mass balance fluxes.

    Parameters
    ----------
    obs : xr.Dataset
        Observational dataset.
    sim_prior : xr.Dataset
        Prior simulation dataset.
    sim_posterior : xr.Dataset
        Posterior simulation dataset.
    config : dict
        Configuration dictionary containing variable names.
    filtering_var : str
        Variable used for filtering.
    filter_range : List[int], optional
        Range of years for filtering, by default [1990, 2019].
    fig_dir : Union[str, Path], optional
        Directory to save the figures, by default "figures".
    sim_alpha : float, optional
        Alpha value for simulation plots, by default 0.4.
    obs_alpha : float, optional
        Alpha value for observation plots, by default 1.0.
    """

    import pism_ragis.processing  # pylint: disable=import-outside-toplevel,reimported
    import matplotlib  # pylint: disable=import-outside-toplevel,reimported
    matplotlib.use('Agg')
    
    Path(fig_dir).mkdir(exist_ok=True)

    percentile_range = (percentiles[1] - percentiles[0]) * 100

    basin = obs.basin.values
    mass_cumulative_varname = config["Cumulative Variables"]["cumulative_mass_balance"]
    mass_cumulative_uncertainty_varname = mass_cumulative_varname + "_uncertainty"
    grounding_line_flux_varname = config["Flux Variables"]["grounding_line_flux"]
    grounding_line_flux_uncertainty_varname = (
        grounding_line_flux_varname + "_uncertainty"
    )

    sim_prior = sim_prior[
        [mass_cumulative_varname, grounding_line_flux_varname, "Ensemble"]
    ].load()
    sim_posterior = sim_posterior[
        [mass_cumulative_varname, grounding_line_flux_varname, "Ensemble"]
    ].load()
    plt.rcParams["font.size"] = fontsize

    fig, axs = plt.subplots(2, 1, sharex=True, figsize=(6.2, 3.2), height_ratios=[2, 1])
    fig.subplots_adjust(hspace=0.05, wspace=0.05)

    obs_ci = axs[0].fill_between(
        obs["time"],
        obs[mass_cumulative_varname] - sigma * obs[mass_cumulative_uncertainty_varname],
        obs[mass_cumulative_varname] + sigma * obs[mass_cumulative_uncertainty_varname],
        color=obs_cmap[0],
        alpha=obs_alpha,
        lw=0,
        label=f"Observed ({sigma}-$\sigma$ uncertainty)",
    )

    axs[1].fill_between(
        obs["time"],
        obs[grounding_line_flux_varname]
        - sigma * obs[grounding_line_flux_uncertainty_varname],
        obs[grounding_line_flux_varname]
        + sigma * obs[grounding_line_flux_uncertainty_varname],
        color=obs_cmap[0],
        alpha=obs_alpha,
        lw=0,
    )

    sim_cis = []
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", r"All-NaN (slice|axis) encountered")
        quantiles = {}
        for q in [percentiles[0], 0.5, percentiles[1]]:
            quantiles[q] = sim_prior.utils.drop_nonnumeric_vars().quantile(
                q, dim="exp_id", skipna=True
            )

    for k, m_var in enumerate([mass_cumulative_varname, grounding_line_flux_varname]):
        sim_ci = axs[k].fill_between(
            quantiles[0.5].time,
            quantiles[percentiles[0]][m_var],
            quantiles[percentiles[1]][m_var],
            alpha=sim_alpha,
            color=sim_cmap[0],
            label=f"""{sim_prior["Ensemble"].values} ({percentile_range:.0f}% credibility interval)""",
            lw=0,
        )
        if k == 0:
            sim_cis.append(sim_ci)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", r"All-NaN (slice|axis) encountered")
        quantiles = {}
        for q in [percentiles[0], 0.5, percentiles[1]]:
            quantiles[q] = sim_posterior.utils.drop_nonnumeric_vars().quantile(
                q, dim="exp_id", skipna=True
            )

    for k, m_var in enumerate([mass_cumulative_varname, grounding_line_flux_varname]):
        sim_ci = axs[k].fill_between(
            quantiles[0.5].time,
            quantiles[percentiles[0]][m_var],
            quantiles[percentiles[1]][m_var],
            alpha=sim_alpha,
            color=sim_cmap[1],
            label=f"""{sim_posterior["Ensemble"].values} ({percentile_range:.0f}% credibility interval)""",
            lw=0,
        )
        if k == 0:
            sim_cis.append(sim_ci)
        axs[k].plot(
            quantiles[0.5].time, quantiles[0.5][m_var], lw=0.75, color=sim_cmap[1]
        )

    y_min, y_max = axs[1].get_ylim()
    scaler = y_min + (y_max - y_min) * 0.05
    obs_filtered = obs.sel({"time": slice(f"{filter_range[0]}", f"{filter_range[-1]}")})
    filter_range_ds = obs_filtered[mass_cumulative_varname]
    filter_range_ds *= 0
    filter_range_ds += scaler
    _ = filter_range_ds.plot(
        ax=axs[1], lw=1, ls="solid", color="k", label="Filtering Range"
    )
    x_s = (
        filter_range_ds.time.values[0]
        + (filter_range_ds.time.values[-1] - filter_range_ds.time.values[0]) / 2
    )
    y_s = scaler
    axs[1].text(
        x_s, y_s, "Filtering Range", horizontalalignment="center", fontweight="medium"
    )
    legend = axs[0].legend(
        handles=[obs_ci, *sim_cis],
    )
    legend.get_frame().set_linewidth(0.0)
    legend.get_frame().set_alpha(0.0)

    axs[0].add_artist(legend)

    axs[0].xaxis.set_tick_params(labelbottom=False)

    axs[0].set_ylabel(f"Cumulative mass\nchange since {reference_date} (Gt)")
    axs[0].set_xlabel("")
    axs[0].set_title(f"{basin} filtered by {filtering_var}")
    axs[1].set_title("")
    axs[1].set_ylabel("Grounding Line\nFlux (Gt/yr)")
    axs[-1].set_xlim(np.datetime64("1980-01-01"), np.datetime64("2021-01-01"))
    fig.tight_layout()
    fig.savefig(
        fig_dir / Path(f"{basin}_mass_accounting_filtered_by_{filtering_var}.pdf")
    )
    plt.close()


def plot_obs_sims_3(
    obs: xr.Dataset,
    sim_prior: xr.Dataset,
    sim_posterior: xr.Dataset,
    config: dict,
    filtering_var: str,
    filter_range: List[int] = [1990, 2019],
    fig_dir: Union[str, Path] = "figures",
    reference_date: str = "1986-01-01",
    sim_alpha: float = 0.4,
    obs_alpha: float = 1.0,
    sigma: float = 2,
    percentiles: List[float] = [0.025, 0.975],
) -> None:
    """
    Plot figure with cumulative mass balance and grounding line flux and climatic
    mass balance fluxes.

    Parameters
    ----------
    obs : xr.Dataset
        Observational dataset.
    sim_prior : xr.Dataset
        Prior simulation dataset.
    sim_posterior : xr.Dataset
        Posterior simulation dataset.
    config : dict
        Configuration dictionary containing variable names.
    filtering_var : str
        Variable used for filtering.
    filter_range : List[int], optional
        Range of years for filtering, by default [1990, 2019].
    fig_dir : Union[str, Path], optional
        Directory to save the figures, by default "figures".
    sim_alpha : float, optional
        Alpha value for simulation plots, by default 0.4.
    obs_alpha : float, optional
        Alpha value for observation plots, by default 1.0.
    """

    import pism_ragis.processing  # pylint: disable=import-outside-toplevel,reimported

    Path(fig_dir).mkdir(exist_ok=True)

    percentile_range = (percentiles[1] - percentiles[0]) * 100

    basin = obs.basin.values
    mass_cumulative_varname = config["Cumulative Variables"]["cumulative_mass_balance"]
    mass_cumulative_uncertainty_varname = mass_cumulative_varname + "_uncertainty"
    grounding_line_flux_varname = config["Flux Variables"]["grounding_line_flux"]
    grounding_line_flux_uncertainty_varname = (
        grounding_line_flux_varname + "_uncertainty"
    )
    smb_flux_varname = config["Flux Variables"]["smb_flux"]
    smb_flux_uncertainty_varname = smb_flux_varname + "_uncertainty"

    plt.rcParams["font.size"] = 6

    fig, axs = plt.subplots(
        3, 1, sharex=True, figsize=(6.2, 4.2), height_ratios=[2, 1, 1]
    )
    fig.subplots_adjust(hspace=0.05, wspace=0.05)

    obs_ci = axs[0].fill_between(
        obs["time"],
        obs[mass_cumulative_varname] - sigma * obs[mass_cumulative_uncertainty_varname],
        obs[mass_cumulative_varname] + sigma * obs[mass_cumulative_uncertainty_varname],
        color=obs_cmap[0],
        alpha=obs_alpha,
        lw=0,
        label=f"Observed ({sigma}-$\sigma$ uncertainty)",
    )

    axs[1].fill_between(
        obs["time"],
        obs[grounding_line_flux_varname]
        - sigma * obs[grounding_line_flux_uncertainty_varname],
        obs[grounding_line_flux_varname]
        + sigma * obs[grounding_line_flux_uncertainty_varname],
        color=obs_cmap[0],
        alpha=obs_alpha,
        lw=0,
    )

    axs[2].fill_between(
        obs["time"],
        obs[smb_flux_varname] - sigma * obs[smb_flux_uncertainty_varname],
        obs[smb_flux_varname] + sigma * obs[smb_flux_uncertainty_varname],
        color=obs_cmap[0],
        alpha=obs_alpha,
        lw=0,
    )

    sim_cis = []
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", r"All-NaN (slice|axis) encountered")
        quantiles = {}
        for q in [percentiles[0], 0.5, percentiles[1]]:
            quantiles[q] = sim_prior.utils.drop_nonnumeric_vars().quantile(
                q, dim="exp_id", skipna=True
            )

    for k, m_var in enumerate(
        [mass_cumulative_varname, grounding_line_flux_varname, smb_flux_varname]
    ):
        sim_posterior[m_var].plot(
            hue="exp_id",
            color=sim_cmap[1],
            ax=axs[k],
            lw=0.1,
            alpha=sim_alpha,
            add_legend=False,
        )
        sim_ci = axs[k].fill_between(
            quantiles[0.5].time,
            quantiles[percentiles[0]][m_var],
            quantiles[percentiles[1]][m_var],
            alpha=sim_alpha,
            color=sim_cmap[0],
            label=f"""{sim_prior["Ensemble"].values} ({percentile_range:.0f}% c.i.)""",
            lw=0,
        )
        if k == 0:
            sim_cis.append(sim_ci)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", r"All-NaN (slice|axis) encountered")
        quantiles = {}
        for q in [percentiles[0], 0.5, percentiles[1]]:
            quantiles[q] = sim_posterior.utils.drop_nonnumeric_vars().quantile(
                q, dim="exp_id", skipna=True
            )

    for k, m_var in enumerate(
        [mass_cumulative_varname, grounding_line_flux_varname, smb_flux_varname]
    ):
        sim_ci = axs[k].fill_between(
            quantiles[0.5].time,
            quantiles[percentiles[0]][m_var],
            quantiles[percentiles[1]][m_var],
            alpha=sim_alpha,
            color=sim_cmap[1],
            label=f"""{sim_posterior["Ensemble"].values} ({percentile_range:.0f}% c.i.)""",
            lw=0,
        )
        if k == 0:
            sim_cis.append(sim_ci)
        axs[k].plot(
            quantiles[0.5].time, quantiles[0.5][m_var], lw=0.75, color=sim_cmap[1]
        )

    y_min, y_max = axs[0].get_ylim()
    scaler = y_min + (y_max - y_min) * 0.05
    obs_filtered = obs.sel(time=slice(filter_range[0], filter_range[-1]))
    filter_range_ds = obs_filtered[mass_cumulative_varname]
    filter_range_ds *= 0
    filter_range_ds += scaler
    _ = filter_range_ds.plot(
        ax=axs[0], lw=1, ls="solid", color="k", label="Filtering Range"
    )
    x_s = (
        filter_range_ds.time.values[0]
        + (filter_range_ds.time.values[-1] - filter_range_ds.time.values[0]) / 2
    )
    y_s = scaler
    axs[0].text(
        x_s, y_s, "Filtering Range", horizontalalignment="center", fontweight="medium"
    )
    legend = axs[0].legend(
        handles=[obs_ci, *sim_cis],
    )
    legend.get_frame().set_linewidth(0.0)
    legend.get_frame().set_alpha(0.0)

    axs[0].add_artist(legend)

    # axs[0].set_ylim(-10_000, 5_000)
    # axs[1].set_ylim(-750, -250)
    # axs[2].set_ylim(-500, 1_000)

    axs[0].xaxis.set_tick_params(labelbottom=False)
    axs[1].xaxis.set_tick_params(labelbottom=False)

    axs[0].set_ylabel(f"Cumulative mass\nloss since {reference_date} (Gt)")
    axs[0].set_xlabel("")
    axs[1].set_xlabel("")
    axs[0].set_title(f"{basin} filtered by {filtering_var}")
    axs[1].set_title("")
    axs[2].set_title("")
    axs[1].set_ylabel("Grounding Line\nFlux (Gt/yr)")
    axs[2].set_ylabel("Climatic Mass\nBalance (Gt/yr)")
    axs[-1].set_xlim(np.datetime64("1980-01-01"), np.datetime64("2021-01-01"))
    fig.tight_layout()
    fig.savefig(
        fig_dir / Path(f"{basin}_mass_accounting_filtered_by_{filtering_var}.pdf")
    )
    plt.close()


# def plot_filtered_hist():
#     outliers_filtered_df = pd.concat([outliers_df, filtered_df]).reset_index(drop=True)
#     # Apply the conversion function to each column
#     outliers_filtered_df = outliers_filtered_df.apply(prp.convert_column_to_numeric)
#     for col in ["surface.given.file", "ocean.th.file", "calving.rate_scaling.file"]:
#         outliers_filtered_df[col] = outliers_filtered_df[col].apply(prp.simplify_path)
#     outliers_filtered_df["surface.given.file"] = outliers_filtered_df[
#         "surface.given.file"
#     ].apply(prp.simplify_climate)
#     outliers_filtered_df["ocean.th.file"] = outliers_filtered_df["ocean.th.file"].apply(
#         prp.simplify_ocean
#     )
#     outliers_filtered_df["calving.rate_scaling.file"] = outliers_filtered_df[
#         "calving.rate_scaling.file"
#     ].apply(prp.simplify_calving)

#     df = outliers_filtered_df.rename(columns=params_short_dict)

#     n_params = len(params_short_dict)
#     plt.rcParams["font.size"] = 4
#     fig, axs = plt.subplots(
#         5,
#         3,
#         sharey=True,
#         figsize=[6.2, 6.2],
#     )
#     fig.subplots_adjust(hspace=1.0, wspace=0.1)
#     for k, v in enumerate(params_short_dict.values()):
#         legend = bool(k == 0)
#         try:
#             sns.histplot(
#                 data=df,
#                 x=v,
#                 hue="Ensemble",
#                 hue_order=["Filtered", "Outliers"],
#                 palette=sim_cmap,
#                 common_norm=False,
#                 stat="probability",
#                 multiple="dodge",
#                 alpha=0.8,
#                 linewidth=0.2,
#                 ax=axs.ravel()[k],
#                 legend=legend,
#             )
#         except:
#             pass
#     for ax in axs.flatten():
#         ax.set_ylabel("")
#         ticklabels = ax.get_xticklabels()
#         for tick in ticklabels:
#             tick.set_rotation(45)
#     fn = result_dir / Path("figures") / Path("outliers_hist.pdf")
#     fig.savefig(fn)
#     plt.close()


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
        help="""Engine for xarray. Default="netcdf4".""",
        type=str,
        default="netcdf4",
    )
    parser.add_argument(
        "--filter_range",
        help="""Time slice used for Importance Sampling, needs an integer year. Default="1986 2019". """,
        type=int,
        nargs=2,
        default=[1986, 2019],
    )
    parser.add_argument(
        "--outlier_range",
        help="""Ensemble members outside this range are removed. Default="-1_250 250". """,
        type=float,
        nargs=2,
        default=[-1250.0, -250.0],
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
        default="2004-01-1",
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

    options, unknown = parser.parse_known_args()
    basin_files = options.FILES
    engine = options.engine
    filter_start_year, filter_end_year = options.filter_range
    fudge_factor = options.fudge_factor
    notebook = options.notebook
    parallel = options.parallel
    reference_date = options.reference_date
    resampling_frequency = options.resampling_frequency
    outlier_variable = options.outlier_variable
    outlier_range = options.outlier_range
    ragis_config_file = Path(
        str(files("pism_ragis.data").joinpath("ragis_config.toml"))
    )
    ragis_config = toml.load(ragis_config_file)
    all_params_dict = ragis_config["Parameters"]
    params = [
        "calving.vonmises_calving.sigma_max",
        "calving.rate_scaling.file",
        "geometry.front_retreat.prescribed.file",
        "ocean.th.gamma_T",
        "surface.given.file",
        "ocean.th.file",
        "frontal_melt.routing.parameter_a",
        "frontal_melt.routing.parameter_b",
        "frontal_melt.routing.power_alpha",
        "frontal_melt.routing.power_beta",
        "stress_balance.sia.enhancement_factor",
        "stress_balance.ssa.Glen_exponent",
        "basal_resistance.pseudo_plastic.q",
        "basal_yield_stress.mohr_coulomb.till_effective_fraction_overburden",
        "basal_yield_stress.mohr_coulomb.topg_to_phi.phi_min",
        "basal_yield_stress.mohr_coulomb.topg_to_phi.phi_max",
        "basal_yield_stress.mohr_coulomb.topg_to_phi.topg_min",
        "basal_yield_stress.mohr_coulomb.topg_to_phi.topg_max",
    ]
    params_short_dict = {key: all_params_dict[key] for key in params}

    result_dir = Path(options.result_dir)
    fig_dir = result_dir / Path("figures")
    fig_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams["font.size"] = 6

    flux_vars = ragis_config["Flux Variables"]
    flux_uncertainty_vars = {
        k + "_uncertainty": v + "_uncertainty" for k, v in flux_vars.items()
    }

    simulated_ds = prepare_simulations(
        basin_files, ragis_config, reference_date, parallel=parallel, engine=engine
    )

    observed_mankoff_ds, observed_grace_ds = prepare_observations(
        options.mankoff_url,
        options.grace_url,
        ragis_config,
        reference_date,
        engine=engine,
    )

    # fig, ax = plt.subplots(1, 1)
    # ds.sel(time=slice(str(filter_start_year), str(filter_end_year))).sel(
    #     basin="GIS", ensemble_id=ensemble
    # ).grounding_line_flux.plot(hue="exp_id", add_legend=False, ax=ax, lw=0.5)
    # fig.savefig("grounding_line_flux_unfiltered.pdf")

    filtered_ds, outliers_ds = filter_outliers(
        simulated_ds,
        outlier_range=outlier_range,
        outlier_variable=outlier_variable,
        subset={"basin": "GIS"},
    )

    # plot_outliers(
    #     filtered_ds.sel(basin="GIS", ensemble_id="RAGIS")[outlier_variable],
    #     outliers_ds.sel(basin="GIS", ensemble_id="RAGIS")[outlier_variable],
    #     Path(fig_dir) / Path(f"{outlier_variable}_filtering.pdf"),
    # )


    prior_config = simulated_ds.sel(pism_config_axis=params).pism_config
    prior_df = config_to_dataframe(prior_config, ensemble="Prior")

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

    filtered_all = {}
    prior_posterior_list = []
    for obs_mean_var, obs_std_var, sim_var in zip(
        list(flux_vars.values())[1:2],
        list(flux_uncertainty_vars.values())[1:2],
        list(flux_vars.values())[1:2],
    ):
        print(f"Importance sampling using {obs_mean_var}")
        f = importance_sampling(
            simulated=simulated_mankoff_basins_resampled_ds.sel(
                time=slice(str(filter_start_year), str(filter_end_year))
            ),
            observed=observed_mankoff_basins_resampled_ds.sel(
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

        posterior_config = filter_config(simulated_mankoff_basins_ds, params)
        posterior_df = config_to_dataframe(posterior_config, ensemble="Posterior")

        prior_posterior_f = pd.concat([prior_df, posterior_df]).reset_index(drop=True)
        prior_posterior_f["filtered_by"] = obs_mean_var
        prior_posterior_list.append(prior_posterior_f)

        filtered_all[obs_mean_var] = pd.concat([prior_df, posterior_df]).rename(
            columns=params_short_dict
        )

        simulated_mankoff_basins_filtered_ds = (
            simulated_mankoff_basins_resampled_ds.sel(exp_id=importance_sampled_ids)
        )
        simulated_mankoff_basins_filtered_ds["Ensemble"] = "Posterior"

        sim_prior = simulated_mankoff_basins_resampled_ds
        sim_prior["Ensemble"] = "Prior"
        sim_posterior = simulated_mankoff_basins_filtered_ds
        sim_posterior["Ensemble"] = "Posterior"

        with prp.tqdm_joblib(
            tqdm(
                desc="Plotting basins",
                total=len(observed_mankoff_basins_resampled_ds.basin),
            )
        ) as progress_bar:

            with ThreadPoolExecutor(max_workers=options.n_jobs) as executor:
                futures = []
                for basin in observed_mankoff_basins_resampled_ds.basin:
                    futures.append(executor.submit(plot_obs_sims,
                        observed_mankoff_basins_resampled_ds.sel(basin=basin),
                        sim_prior.sel(basin=basin),
                        sim_posterior.sel(basin=basin),
                        config=ragis_config,
                        filtering_var=obs_mean_var,
                        filter_range=[filter_start_year, filter_end_year],
                        fig_dir=fig_dir,
                        obs_alpha=obs_alpha,
                        sim_alpha=sim_alpha,
    ))
                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception as e:
                        print(f"An error occurred: {e}")

            

    prior_posterior = pd.concat(prior_posterior_list).reset_index()
    prior_posterior = prior_posterior.apply(prp.convert_column_to_numeric)
    # Define a mapping of columns to their corresponding functions

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

    for (basin, filtering_var), df in prior_posterior.rename(
        columns=params_short_dict
    ).groupby(by=["basin", "filtered_by"]):
        n_params = len(params_short_dict)
        plt.rcParams["font.size"] = 4
        fig, axs = plt.subplots(
            4,
            4,
            sharey=True,
            figsize=[6.2, 6.2],
        )
        fig.subplots_adjust(hspace=1.0, wspace=0.1)
        for k, v in enumerate(params_short_dict.values()):
            legend = bool(k == 0)
            try:
                sns.histplot(
                    data=df,
                    x=v,
                    hue="Ensemble",
                    hue_order=["Prior", "Posterior"],
                    palette=sim_cmap,
                    common_norm=False,
                    stat="probability",
                    multiple="dodge",
                    alpha=0.8,
                    linewidth=0.2,
                    ax=axs.ravel()[k],
                    legend=legend,
                )
            except:
                pass
        for ax in axs.flatten():
            ax.set_ylabel("")
            ticklabels = ax.get_xticklabels()
            for tick in ticklabels:
                tick.set_rotation(45)
        fn = (
            result_dir
            / Path("figures")
            / Path(f"{basin}_prior_posterior_filtered_by_{filtering_var}.pdf")
        )
        fig.savefig(fn)
        plt.close()

    ensemble_df = prior_df.apply(prp.convert_column_to_numeric).drop(
        columns=["Ensemble", "exp_id"], errors="ignore"
    )
    climate_dict = {
        v: k for k, v in enumerate(ensemble_df["surface.given.file"].unique())
    }
    ensemble_df["surface.given.file"] = ensemble_df["surface.given.file"].map(
        climate_dict
    )
    ocean_dict = {v: k for k, v in enumerate(ensemble_df["ocean.th.file"].unique())}
    ensemble_df["ocean.th.file"] = ensemble_df["ocean.th.file"].map(ocean_dict)
    calving_dict = {
        v: k for k, v in enumerate(ensemble_df["calving.rate_scaling.file"].unique())
    }
    ensemble_df["calving.rate_scaling.file"] = ensemble_df[
        "calving.rate_scaling.file"
    ].map(calving_dict)

    retreat_dict = {
        v: k for k, v in enumerate(ensemble_df["geometry.front_retreat.prescribed.file"].unique())
    }

    ensemble_df["geometry.front_retreat.prescribed.file"] = ensemble_df[
        "geometry.front_retreat.prescribed.file"
    ].map(retreat_dict)
    to_analyze = simulated_ds.sel(time=slice("1980-01-01", "2020-01-01"))
    all_delta_indices = run_delta_analysis(
        to_analyze, ensemble_df, list(flux_vars.values())[1:2], notebook=notebook
    )

    # Extract the prefix from each coordinate value
    prefixes = [
        name.split(".")[0] for name in all_delta_indices.pism_config_axis.values
    ]

    # Add the prefixes as a new coordinate
    all_delta_indices = all_delta_indices.assign_coords(
        prefix=("pism_config_axis", prefixes)
    )

    parameter_groups = ragis_config["Parameter Groups"]

    si_prefixes = [parameter_groups[name] for name in all_delta_indices.prefix.values]
    all_delta_indices = all_delta_indices.assign_coords(
        sensitivity_indices_group=("pism_config_axis", si_prefixes)
    )
    # Group by the new coordinate and compute the sum for each group
    aggregated_data = (
        all_delta_indices.groupby("sensitivity_indices_group")
        .sum()
        .rolling(time=13)
        .mean()
    )

    for index in ["S1", "delta"]:
        for basin in aggregated_data.basin.values:
            for filter_var in aggregated_data.filtered_by.values:
                fig, ax = plt.subplots(1, 1)
                aggregated_data.sel(filtered_by=filter_var, basin=basin)[index].plot(
                    hue="sensitivity_indices_group", ax=ax
                )
                ax.set_title(f"S1 for {basin} filtered by {filter_var}")
                fn = (
                    result_dir
                    / Path("figures")
                    / Path(f"{basin}_{index}_filtered_by_{filter_var}.pdf")
                )
                fig.savefig(fn)
                plt.close()
