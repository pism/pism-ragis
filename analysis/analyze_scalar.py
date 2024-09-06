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

# pylint: disable=unused-import
"""
Analyze RAGIS ensemble.
"""

import time
import warnings
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from importlib.resources import files
from pathlib import Path
from typing import Any, Dict, Hashable, List, Mapping, Union

import dask
import numpy as np
import pandas as pd
import pylab as plt
import seaborn as sns
import toml
import xarray as xr
from dask.diagnostics import ProgressBar
from dask.distributed import Client, LocalCluster, progress
from joblib import Parallel, delayed
from tqdm.auto import tqdm

import pism_ragis.processing as prp
from pism_ragis.analysis import delta_analysis
from pism_ragis.filtering import importance_sampling
from pism_ragis.likelihood import log_normal

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


def filter_outliers(
    ds: xr.Dataset,
    outlier_range: List[float],
    outlier_variable: str,
    subset: Dict = {"basin": "GIS", "ensemble_id": "RAGIS"},
) -> Dict[str, xr.Dataset]:
    """
    Filter outliers from a dataset based on a specified variable and range.

    This function filters out ensemble members from the dataset `ds` where the values of
    `outlier_variable` fall outside the specified `outlier_range`. The filtering is done
    for the specified subset of the dataset.

    Parameters
    ----------
    ds : xr.Dataset
        The input dataset containing the data to be filtered.
    outlier_range : List[float]
        A list containing the lower and upper bounds for the outlier range.
    outlier_variable : str
        The variable in the dataset to be used for outlier detection.
    subset : Dict, optional
        A dictionary specifying the subset of the dataset to apply the filter on, by default {"basin": "GIS", "ensemble_id": "RAGIS"}.

    Returns
    -------
    Dict[str, xr.Dataset]
        A dictionary with two keys:
        - "filtered": The dataset with outliers removed.
        - "outliers": The dataset containing only the outliers.
    """
    lower_bound, upper_bound = outlier_range
    print(f"Filtering outliers [{lower_bound}, {upper_bound}] for {outlier_variable}")
    days_in_month = ds.time.dt.days_in_month
    wgts = days_in_month.groupby("time.year") / days_in_month.groupby("time.year").sum()

    outlier_filter = ds.sel(subset).utils.drop_nonnumeric_vars()[outlier_variable]
    outlier_filter = (outlier_filter * wgts).resample(time="YS").sum(dim="time")
    mask = (outlier_filter >= lower_bound) & (outlier_filter <= upper_bound)
    mask = ~(~mask).any(dim="time")
    filtered_ds = outlier_filter.sel(exp_id=mask)
    filtered_exp_ids = filtered_ds.exp_id.values
    outlier_ds = outlier_filter.sel(exp_id=~mask)
    outlier_exp_ids = outlier_ds.exp_id.values
    n_members = len(ds.exp_id)
    n_members_filtered = len(filtered_exp_ids)
    print(f"Ensemble size: {n_members}, outlier-filtered size: {n_members_filtered}")
    filtered_ds = ds.sel(exp_id=filtered_exp_ids)
    outliers_ds = ds.sel(exp_id=outlier_exp_ids)
    return {"filtered": filtered_ds, "outliers": outliers_ds}


def run_delta_analysis(
    ds: xr.Dataset,
    ensemble_df: pd.DataFrame,
    filter_vars: List[str],
    group_dim: str = "basin",
    iter_dim: str = "time",
    ensemble: str = "RAGIS",
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
    ensemble_id : str, optional
        The ensemble ID to select from the dataset, by default "RAGIS".

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

    start_dask = time.time()
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
            start = time.time()

            responses = ds.sel(basin=gdim, ensemble_id=ensemble)[filter_var]
            responses_scattered = client.scatter(
                [
                    responses.isel(time=k).to_numpy()
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

            end = time.time()
            time_elapsed = end - start
            print(f"  ...took {time_elapsed:.0f}s")

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

    end_dask = time.time()
    dask_time_elapsed = end_dask - start_dask
    print(f"  ...took {dask_time_elapsed:.0f}s")

    return all_delta_indices


def plot_obs_sims(
    obs: xr.Dataset,
    sim_prior: xr.Dataset,
    sim_posterior: xr.Dataset,
    config: dict,
    filtering_var: str,
    filter_range: List[int] = [1990, 2019],
    fig_dir: Union[str, Path] = "figures",
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
    obs_filtered = obs.sel(time=slice(filter_range[0], filter_range[-1]))
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

    axs[0].set_ylabel(f"Cumulative mass\nloss since {reference_year} (Gt)")
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

    axs[0].set_ylabel(f"Cumulative mass\nloss since {reference_year} (Gt)")
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
        "--obs_url",
        help="""Path to "observed" mass balance.""",
        type=str,
        default="data/mankoff/mankoff_mass_balance.nc",
    )
    parser.add_argument(
        "--engine",
        help="""Engine for xarray. Default="netcdf4".""",
        type=str,
        default="h5netcdf",
    )
    parser.add_argument(
        "--filter_range",
        help="""Time slice used for Importance Sampling. Default="1990 2019". """,
        type=str,
        nargs=2,
        default="1986 2019",
    )
    parser.add_argument(
        "--outlier_range",
        help="""Ensemble members outside this range are removed. Default="-1_250 250". """,
        type=str,
        nargs=2,
        default="-1250 -250",
    )
    parser.add_argument(
        "--outlier_variable",
        help="""Quantity to filter outliers. Default="grounding_line_flux".""",
        type=str,
        default="grounding_line_flux",
    )
    parser.add_argument(
        "--ensemble",
        help="""Name of the ensemble. Default=RAGIS.""",
        type=str,
        default="RAGIS",
    )
    parser.add_argument(
        "--fudge_factor",
        help="""Observational uncertainty multiplier. Default=3""",
        type=int,
        default=3,
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
        default="MS",
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
        default=None,
    )
    parser.add_argument(
        "FILES",
        help="""Ensemble netCDF files.""",
        nargs="*",
    )

    options = parser.parse_args()
    basin_files = options.FILES
    ensemble = options.ensemble
    engine = options.engine
    filter_start_year, filter_end_year = options.filter_range.split(" ")
    fudge_factor = options.fudge_factor
    notebook = options.notebook
    parallel = options.parallel
    reference_year = options.reference_year
    resampling_frequency = options.resampling_frequency
    outlier_variable = options.outlier_variable
    outlier_range = [float(v) for v in options.outlier_range.split(" ")]
    ragis_config_file = Path(
        str(files("pism_ragis.data").joinpath("ragis_config.toml"))
    )
    ragis_config = toml.load(ragis_config_file)
    all_params_dict = ragis_config["Parameters"]

    params = [
        "calving.vonmises_calving.sigma_max",
        "calving.rate_scaling.file",
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
    cumulative_vars = ragis_config["Cumulative Variables"]
    cumulative_uncertainty_vars = {
        k + "_uncertainty": v + "_uncertainty" for k, v in cumulative_vars.items()
    }

    ds = prp.load_ensemble(basin_files, parallel=parallel, engine=engine).sortby(
        "basin"
    )
    for v in ds.data_vars:
        if ds[v].dtype.kind == "S":
            ds[v] = ds[v].astype(str)
    for c in ds.coords:
        if ds[c].dtype.kind == "S":
            ds.coords[c] = ds.coords[c].astype(str)

    start = time.time()
    ds = xr.apply_ufunc(np.vectorize(convert_bstrings_to_str), ds, dask="parallelized")
    ds = ds.dropna(dim="exp_id")
    end = time.time()
    time_elapsed = end - start
    print(f"Preps  ...took {time_elapsed:.0f}s")

    start = time.time()
    ds = prp.standardize_variable_names(ds, ragis_config["PISM Spatial"])
    ds[ragis_config["Cumulative Variables"]["cumulative_grounding_line_flux"]] = ds[
        ragis_config["Flux Variables"]["grounding_line_flux"]
    ].cumsum() / len(ds.time)
    ds[ragis_config["Cumulative Variables"]["cumulative_smb"]] = ds[
        ragis_config["Flux Variables"]["smb_flux"]
    ].cumsum() / len(ds.time)
    ds = prp.normalize_cumulative_variables(
        ds,
        list(ragis_config["Cumulative Variables"].values()),
        reference_year=reference_year,
    )
    end = time.time()
    time_elapsed = end - start
    print(f"Normalization  ...took {time_elapsed:.0f}s")

    fig, ax = plt.subplots(1, 1)
    ds.sel(time=slice(str(filter_start_year), str(filter_end_year))).sel(
        basin="GIS", ensemble_id=ensemble
    ).grounding_line_flux.plot(hue="exp_id", add_legend=False, ax=ax, lw=0.5)
    fig.savefig("grounding_line_flux_unfiltered.pdf")

    start = time.time()
    result = filter_outliers(
        ds, outlier_range=outlier_range, outlier_variable=outlier_variable
    )
    filtered_ds = result["filtered"]
    outliers_ds = result["outliers"]
    end = time.time()
    time_elapsed = end - start
    print(f"Filter outliers  ...took {time_elapsed:.0f}s")

    fig, ax = plt.subplots(1, 1)
    ds.sel(time=slice(str(filter_start_year), str(filter_end_year))).sel(
        basin="GIS", ensemble_id=ensemble
    ).grounding_line_flux.plot(hue="exp_id", add_legend=False, ax=ax, lw=0.5)
    fig.savefig("grounding_line_flux_filtered.pdf")

    start = time.time()
    prior_config = ds.sel(pism_config_axis=params).pism_config
    dims = [dim for dim in prior_config.dims if not dim in ["pism_config_axis"]]
    end = time.time()
    time_elapsed = end - start
    print(f"Select dims  ...took {time_elapsed:.0f}s")
    start = time.time()
    prior_df = prior_config.to_dataframe().reset_index()
    end = time.time()
    time_elapsed = end - start
    print(f"Conversion to prior dataframe  ...took {time_elapsed:.0f}s")
    start = time.time()
    prior = prior_df.pivot(index=dims, columns="pism_config_axis", values="pism_config")
    end = time.time()
    time_elapsed = end - start
    print(f"Pivot table  ...took {time_elapsed:.0f}s")
    prior.reset_index(inplace=True)
    prior["Ensemble"] = "Prior"
    end = time.time()
    time_elapsed = end - start
    print(f"Conversion to prior dataframe  ...took {time_elapsed:.0f}s")

    start = time.time()
    outlier_config = outliers_ds.sel(pism_config_axis=params).pism_config
    dims = [dim for dim in outlier_config.dims if not dim in ["pism_config_axis"]]
    outlier_df = outlier_config.to_dataframe().reset_index()
    outlier_df = outlier_df.pivot(
        index=dims, columns="pism_config_axis", values="pism_config"
    )
    outlier_df.reset_index(inplace=True)
    outlier_df["Ensemble"] = "Outliers"
    end = time.time()
    time_elapsed = end - start
    print(f"Conversion to outlier dataframe  ...took {time_elapsed:.0f}s")

    start = time.time()
    filtered_config = filtered_ds.sel(pism_config_axis=params).pism_config
    dims = [dim for dim in filtered_config.dims if not dim in ["pism_config_axis"]]
    filtered_df = filtered_config.to_dataframe().reset_index()
    filtered_df = filtered_df.pivot(
        index=dims, columns="pism_config_axis", values="pism_config"
    )
    filtered_df.reset_index(inplace=True)
    filtered_df["Ensemble"] = "Filtered"
    end = time.time()
    time_elapsed = end - start
    print(f"Conversion to filtered dataframe  ...took {time_elapsed:.0f}s")

    outliers_filtered_df = pd.concat([outlier_df, filtered_df]).reset_index(drop=True)
    # Apply the conversion function to each column
    outliers_filtered_df = outliers_filtered_df.apply(prp.convert_column_to_numeric)
    for col in ["surface.given.file", "ocean.th.file", "calving.rate_scaling.file"]:
        outliers_filtered_df[col] = outliers_filtered_df[col].apply(prp.simplify)
    outliers_filtered_df["surface.given.file"] = outliers_filtered_df[
        "surface.given.file"
    ].apply(prp.simplify_climate)
    outliers_filtered_df["ocean.th.file"] = outliers_filtered_df["ocean.th.file"].apply(
        prp.simplify_ocean
    )
    outliers_filtered_df["calving.rate_scaling.file"] = outliers_filtered_df[
        "calving.rate_scaling.file"
    ].apply(prp.simplify_calving)

    df = outliers_filtered_df.rename(columns=params_short_dict)

    end = time.time()
    time_elapsed = end - start
    print(f"Conversion to dataframe  ...took {time_elapsed:.0f}s")

    n_params = len(params_short_dict)
    plt.rcParams["font.size"] = 4
    fig, axs = plt.subplots(
        5,
        3,
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
                hue_order=["Filtered", "Outliers"],
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
    fn = result_dir / Path("figures") / Path("outliers_hist.pdf")
    fig.savefig(fn)
    plt.close()

    observed = xr.open_dataset(options.obs_url, engine=engine, chunks="auto").sel(
        time=slice("1980", "2022")
    )
    observed = observed.sortby("basin")
    observed = prp.normalize_cumulative_variables(
        observed,
        list(cumulative_vars.values()) + list(cumulative_uncertainty_vars.values()),
        reference_year,
    )

    observed_days_in_month = observed["time"].dt.days_in_month

    observed_wgts = 1 / (observed_days_in_month)
    observed_resampled = (
        (observed * observed_wgts)
        .resample(time=resampling_frequency)
        .sum(dim="time")
        .rolling(time=13)
        .mean()
    )

    simulated = filtered_ds
    simulated_resampled = (
        simulated.drop_vars(["pism_config", "run_stats"], errors="ignore")
        .resample(time=resampling_frequency)
        .sum(dim="time")
        .rolling(time=13)
        .mean()
    )
    simulated_resampled["pism_config"] = simulated["pism_config"]

    start = time.time()
    filtered_all = {}
    prior_posterior_list = []
    for obs_mean_var, obs_std_var, sim_var in zip(
        list(flux_vars.values())[:2],
        list(flux_uncertainty_vars.values())[:2],
        list(flux_vars.values())[:2],
    ):
        print(f"Importance sampling using {obs_mean_var}")

        f = importance_sampling(
            simulated=simulated_resampled.sel(
                time=slice(str(filter_start_year), str(filter_end_year))
            ),
            observed=observed_resampled.sel(
                time=slice(str(filter_start_year), str(filter_end_year))
            ),
            log_likelihood=log_normal,
            fudge_factor=fudge_factor,
            n_samples=len(simulated.exp_id),
            obs_mean_var=obs_mean_var,
            obs_std_var=obs_std_var,
            sim_var=sim_var,
        )
        with ProgressBar():
            result = f.compute()
        filtered_ids = result["exp_id_sampled"]
        filtered_ids["basin"] = filtered_ids["basin"].astype("<U3")

        posterior_config = (
            ds.sel(pism_config_axis=params).sel(exp_id=filtered_ids).pism_config
        )
        dims = [dim for dim in prior_config.dims if not dim in ["pism_config_axis"]]
        posterior_df = posterior_config.to_dataframe().reset_index()
        posterior = posterior_df.pivot(
            index=dims, columns="pism_config_axis", values="pism_config"
        )
        posterior.reset_index(inplace=True)
        posterior["Ensemble"] = "Posterior"

        prior_posterior_f = pd.concat([prior, posterior]).reset_index(drop=True)
        prior_posterior_f["filtered_by"] = obs_mean_var
        prior_posterior_list.append(prior_posterior_f)

        filtered_all[obs_mean_var] = pd.concat([prior, posterior]).rename(
            columns=params_short_dict
        )

        simulated_filtered = simulated_resampled.sel(exp_id=filtered_ids)
        simulated_filtered["Ensemble"] = "Posterior"

        sim_prior = simulated_resampled
        sim_prior["Ensemble"] = "Prior"
        sim_posterior = simulated_filtered
        sim_posterior["Ensemble"] = "Posterior"

        with prp.tqdm_joblib(
            tqdm(desc="Plotting basins", total=len(observed_resampled.basin))
        ) as progress_bar:
            result = Parallel(n_jobs=options.n_jobs)(
                delayed(plot_obs_sims)(
                    observed_resampled.sel(basin=basin),
                    sim_prior.sel(basin=basin, ensemble_id=ensemble),
                    sim_posterior.sel(basin=basin, ensemble_id=ensemble),
                    config=ragis_config,
                    filtering_var=obs_mean_var,
                    filter_range=[filter_start_year, filter_end_year],
                    fig_dir=result_dir / Path("figures"),
                    obs_alpha=obs_alpha,
                    sim_alpha=sim_alpha,
                )
                for basin in observed_resampled.basin
            )
    end = time.time()
    time_elapsed = end - start
    print(f"IS Plotting  ...took {time_elapsed:.0f}s")

    prior_posterior = pd.concat(prior_posterior_list).reset_index()
    prior_posterior = prior_posterior.apply(prp.convert_column_to_numeric)
    for col in ["surface.given.file", "ocean.th.file", "calving.rate_scaling.file"]:
        prior_posterior[col] = prior_posterior[col].apply(prp.simplify)
    prior_posterior["surface.given.file"] = prior_posterior["surface.given.file"].apply(
        prp.simplify_climate
    )
    prior_posterior["ocean.th.file"] = prior_posterior["ocean.th.file"].apply(
        prp.simplify_ocean
    )
    prior_posterior["calving.rate_scaling.file"] = prior_posterior[
        "calving.rate_scaling.file"
    ].apply(prp.simplify_calving)

    for (basin, filtering_var), df in prior_posterior.rename(
        columns=params_short_dict
    ).groupby(by=["basin", "filtered_by"]):
        n_params = len(params_short_dict)
        plt.rcParams["font.size"] = 4
        fig, axs = plt.subplots(
            5,
            3,
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

    ensemble_df = prior.apply(prp.convert_column_to_numeric).drop(
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

    to_analyze = ds.sel(time=slice("1980-01-01", "2020-01-01"))
    all_delta_indices = run_delta_analysis(
        to_analyze, ensemble_df, list(flux_vars.values())[:2], notebook=notebook
    )

    # Extract the prefix from each coordinate value
    prefixes = [
        name.split(".")[0] for name in all_delta_indices.pism_config_axis.values
    ]

    # Add the prefixes as a new coordinate
    all_delta_indices = all_delta_indices.assign_coords(
        prefix=("pism_config_axis", prefixes)
    )

    sensitivity_indices_groups = {
        "surface": "Climate",
        "atmosphere": "Climate",
        "ocean": "Ocean",
        "calving": "Calving",
        "frontal_melt": "Frontal Melt",
        "basal_resistance": "Flow",
        "basal_yield_stress": "Flow",
        "stress_balance": "Flow",
    }
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
