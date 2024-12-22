# Copyright (C) 2024 Andy Aschwanden
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

# pylint: disable=unused-import,too-many-positional-arguments

"""
Module for data plotting.
"""
import json
import warnings
from importlib.resources import files
from pathlib import Path
from typing import Dict, List, Union

import matplotlib as mpl
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import seaborn as sns
import toml
import xarray as xr
from tqdm.auto import tqdm

from pism_ragis.decorators import profileit, timeit

ragis_config_file = Path(str(files("pism_ragis.data").joinpath("ragis_config.toml")))
ragis_config = toml.load(ragis_config_file)
config = json.loads(json.dumps(ragis_config))

obs_alpha = config["Plotting"]["obs_alpha"]
obs_cmap = config["Plotting"]["obs_cmap"]
sim_alpha = config["Plotting"]["sim_alpha"]
sim_cmap = config["Plotting"]["sim_cmap"]


@timeit
def plot_prior_posteriors(
    df: pd.DataFrame,
    fig_dir: Union[str, Path] = "figures",
    fontsize: float = 4,
):
    """
    Plot histograms of prior and posterior distributions.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the data to plot.
    fig_dir : Union[str, Path], optional
        Directory to save the figures, by default "figures".
    fontsize : float, optional
        Font size for the plot, by default 4.
    """

    plot_dir = fig_dir / Path("basin_histograms")
    plot_dir.mkdir(parents=True, exist_ok=True)
    pdf_dir = plot_dir / Path("pdfs")
    pdf_dir.mkdir(parents=True, exist_ok=True)
    png_dir = plot_dir / Path("pngs")
    png_dir.mkdir(parents=True, exist_ok=True)

    group_columns = ["basin", "filtered_by"]

    rc_params = {
        "font.size": fontsize,
        # Add other rcParams settings if needed
    }

    with mpl.rc_context(rc=rc_params):
        for (basin, filter_var), m_df in df.groupby(by=group_columns):
            m_df = m_df.drop(columns=group_columns + ["exp_id"])
            fig, axs = plt.subplots(
                3,
                6,
                sharey=False,
                figsize=[6.2, 3.2],
            )
            fig.subplots_adjust(hspace=0.5, wspace=0.22)
            for k, v in enumerate(m_df.drop(columns=["ensemble"]).columns):
                legend = bool(k == 1)
                try:
                    _ = sns.histplot(
                        data=m_df,
                        x=v,
                        hue="ensemble",
                        hue_order=["Prior", "Posterior"],
                        palette=sim_cmap,
                        common_norm=False,
                        stat="count",
                        multiple="dodge",
                        alpha=0.8,
                        linewidth=0.2,
                        ax=axs.ravel()[k],
                        legend=legend,
                    )
                except:
                    pass
                if legend:
                    axs.ravel()[k].get_legend().set_title(None)
                    axs.ravel()[k].get_legend().get_frame().set_linewidth(0.0)
                    axs.ravel()[k].get_legend().get_frame().set_alpha(0.0)

            for ax in axs.flatten():
                ax.set_ylabel("")
                # ax.set_ylim(0, 1)
                ticklabels = ax.get_xticklabels()
                for tick in ticklabels:
                    tick.set_rotation(15)
            fn = pdf_dir / Path(f"{basin}_prior_posterior_filtered_by_{filter_var}.pdf")
            fig.savefig(fn)
            fn = png_dir / Path(f"{basin}_prior_posterior_filtered_by_{filter_var}.png")
            fig.savefig(fn, dpi=300)
            plt.close()
            del fig


@timeit
def plot_basins(
    observed: xr.Dataset,
    prior: xr.Dataset,
    posterior: xr.Dataset,
    filter_var: str,
    filter_range: List[int] = [1990, 2019],
    fig_dir: Union[str, Path] = "figures",
    plot_range: List[int] = [1980, 2020],
    config: Dict = {},
):
    """
    Plot basins using observed, prior, and posterior datasets.

    This function plots the observed, prior, and posterior datasets for each basin,
    and saves the plots to the specified directory. It also displays a progress bar
    to indicate the progress of the plotting process.

    Parameters
    ----------
    observed : xr.Dataset
        The observed dataset.
    prior : xr.Dataset
        The prior dataset.
    posterior : xr.Dataset
        The posterior dataset.
    filter_var : str
        The variable used for filtering.
    filter_range : List[int], optional
        A list containing the start and end years for filtering, by default [1990, 2019].
    fig_dir : Union[str, Path], optional
        The directory where figures will be saved, by default "figures".
    plot_range : List[int], optional
        A list containing the start and end years for plotting, by default [1980, 2020].
    config : Dict, optional
        Configuration dictionary, by default {}.

    Examples
    --------
    >>> observed = xr.Dataset(...)
    >>> prior = xr.Dataset(...)
    >>> posterior = xr.Dataset(...)
    >>> plot_basins(observed, prior, posterior, "grounding_line_flux")
    """

    with tqdm(
        desc="Plotting basins",
        total=len(observed.basin),
    ) as progress_bar:
        for basin in observed.basin:
            plot_obs_sims(
                observed.sel(basin=basin).sel(
                    {"time": slice(str(plot_range[0]), str(plot_range[1]))}
                ),
                prior.sel(basin=basin).sel(
                    {"time": slice(str(plot_range[0]), str(plot_range[1]))}
                ),
                posterior.sel(basin=basin).sel(
                    {"time": slice(str(plot_range[0]), str(plot_range[1]))}
                ),
                config=config,
                filter_var=filter_var,
                filter_range=filter_range,
                fig_dir=fig_dir,
                obs_alpha=obs_alpha,
                sim_alpha=sim_alpha,
            )
            progress_bar.update()


@timeit
def plot_sensitivity_indices(
    ds: xr.Dataset,
    indices_var: str = "S1",
    indices_conf_var: str = "S1_conf",
    basin: str = "",
    filter_var: str = "",
    fig_dir: Union[str, Path] = "figures",
    fontsize: float = 6,
):
    """
    Plot sensitivity indices with confidence intervals.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset containing sensitivity indices and confidence intervals.
    indices_var : str, optional
        The variable name for sensitivity indices in the dataset, by default "S1".
    indices_conf_var : str, optional
        The variable name for confidence intervals of sensitivity indices in the dataset, by default "S1_conf".
    basin : str
        The basin parameter to be used in the plot.
    filter_var : str, optional
        The variable used for filtering, by default "".
    fig_dir : Union[str, Path], optional
        The directory where the figures will be saved, by default "figures".
    fontsize : float, optional
        The font size for the plot, by default 6.
    """
    plot_dir = Path(fig_dir) / "sensitivity_indices"
    plot_dir.mkdir(parents=True, exist_ok=True)
    pdf_dir = plot_dir / "pdfs"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    png_dir = plot_dir / "pngs"
    png_dir.mkdir(parents=True, exist_ok=True)

    with mpl.rc_context({"font.size": fontsize}):

        fig, ax = plt.subplots(1, 1, figsize=(6.2, 3.6))
        for g in ds.sensitivity_indices_group:
            indices_da = ds[indices_var].sel(sensitivity_indices_group=g)
            conf_da = ds[indices_conf_var].sel(sensitivity_indices_group=g)
            ax.fill_between(
                indices_da.time,
                (indices_da - conf_da),
                (indices_da + conf_da),
                alpha=0.25,
            )
            indices_da.plot(
                hue="sensitivity_indices_group", ax=ax, lw=0.75, label=g.values
            )
        legend = ax.legend(loc="upper left")
        legend.get_frame().set_linewidth(0.0)
        legend.get_frame().set_alpha(0.0)
        ax.set_title(f"{indices_var} for basin {basin} for {filter_var}")
        fn = pdf_dir / f"basin_{basin}_{indices_var}_for_{filter_var}.pdf"
        fig.savefig(fn)
        plt.close()


@timeit
def plot_obs_sims(
    obs: xr.Dataset,
    sim_prior: xr.Dataset,
    sim_posterior: xr.Dataset,
    config: dict,
    filter_var: str,
    filter_range: List[int] = [1990, 2019],
    fig_dir: Union[str, Path] = "figures",
    reference_year: float = 1986.0,
    sim_alpha: float = 0.4,
    obs_alpha: float = 1.0,
    sigma: float = 2,
    percentiles: List[float] = [0.025, 0.975],
    fontsize: float = 6,
) -> None:
    """
    Plot cumulative mass balance and grounding line flux.

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
    filter_var : str
        Variable used for filtering.
    filter_range : List[int], optional
        Range of years for filtering, by default [1990, 2019].
    fig_dir : Union[str, Path], optional
        Directory to save the figures, by default "figures".
    reference_year : float, optional
        Reference year for cumulative mass balance, by default 1986.0.
    sim_alpha : float, optional
        Alpha value for simulation plots, by default 0.4.
    obs_alpha : float, optional
        Alpha value for observation plots, by default 1.0.
    sigma : float, optional
        Sigma value for uncertainty, by default 2.
    percentiles : List[float], optional
        Percentiles for credibility interval, by default [0.025, 0.975].
    fontsize : float, optional
        Font size for the plot, by default 6.
    """

    import pism_ragis.processing  # pylint: disable=import-outside-toplevel,reimported

    Path(fig_dir).mkdir(exist_ok=True)
    plot_dir = fig_dir / Path("basin_timeseries")
    plot_dir.mkdir(parents=True, exist_ok=True)
    pdf_dir = plot_dir / Path("pdfs")
    pdf_dir.mkdir(parents=True, exist_ok=True)
    png_dir = plot_dir / Path("pngs")
    png_dir.mkdir(parents=True, exist_ok=True)

    percentile_range = (percentiles[1] - percentiles[0]) * 100

    basin = obs.basin.values
    mass_cumulative_varname = config["Cumulative Variables"]["cumulative_mass_balance"]
    mass_cumulative_uncertainty_varname = config["Cumulative Uncertainty Variables"][
        "cumulative_mass_balance_uncertainty"
    ]
    grounding_line_flux_varname = config["Flux Variables"]["grounding_line_flux"]
    grounding_line_flux_uncertainty_varname = config["Flux Uncertainty Variables"][
        "grounding_line_flux_uncertainty"
    ]
    mass_flux_varname = config["Flux Variables"]["mass_flux"]
    mass_flux_uncertainty_varname = config["Flux Uncertainty Variables"][
        "mass_flux_uncertainty"
    ]

    with mpl.rc_context({"font.size": fontsize}):

        fig, axs = plt.subplots(
            3,
            1,
            sharex=True,
            figsize=(6.2, 3.6),
            height_ratios=[2, 1, 1],
        )
        fig.subplots_adjust(hspace=0.05, wspace=0.05)

        obs_ci = axs[0].fill_between(
            obs["time"],
            obs[mass_cumulative_varname]
            - sigma * obs[mass_cumulative_uncertainty_varname],
            obs[mass_cumulative_varname]
            + sigma * obs[mass_cumulative_uncertainty_varname],
            color=obs_cmap[0],
            alpha=obs_alpha,
            lw=0,
            label=f"Observed ({sigma}-$\sigma$ uncertainty)",
        )

        if mass_flux_varname in obs.data_vars:
            axs[1].fill_between(
                obs["time"],
                obs[mass_flux_varname] - sigma * obs[mass_flux_uncertainty_varname],
                obs[mass_flux_varname] + sigma * obs[mass_flux_uncertainty_varname],
                color=obs_cmap[0],
                alpha=obs_alpha,
                lw=0,
            )

        if grounding_line_flux_varname in obs.data_vars:
            axs[-1].fill_between(
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
        if sim_prior is not None:
            sim_prior = sim_prior[
                [
                    mass_cumulative_varname,
                    mass_flux_varname,
                    grounding_line_flux_varname,
                    "ensemble",
                ]
            ].load()
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", r"All-NaN (slice|axis) encountered")
                quantiles = {}
                for q in [percentiles[0], 0.5, percentiles[1]]:
                    quantiles[q] = sim_prior.utils.drop_nonnumeric_vars().quantile(
                        q, dim="exp_id", skipna=True
                    )

            for k, m_var in enumerate(
                [
                    mass_cumulative_varname,
                    mass_flux_varname,
                    grounding_line_flux_varname,
                ]
            ):
                sim_ci = axs[k].fill_between(
                    quantiles[0.5].time,
                    quantiles[percentiles[0]][m_var],
                    quantiles[percentiles[1]][m_var],
                    alpha=sim_alpha,
                    color=sim_cmap[0],
                    label=f"""{sim_prior["ensemble"].values} ({percentile_range:.0f}% credibility interval)""",
                    lw=0,
                )
                if k == 0:
                    sim_cis.append(sim_ci)
        if sim_posterior is not None:
            sim_posterior = sim_posterior[
                [
                    mass_cumulative_varname,
                    mass_flux_varname,
                    grounding_line_flux_varname,
                    "ensemble",
                ]
            ].load()
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", r"All-NaN (slice|axis) encountered")
                quantiles = {}
                for q in [percentiles[0], 0.5, percentiles[1]]:
                    quantiles[q] = sim_posterior.utils.drop_nonnumeric_vars().quantile(
                        q, dim="exp_id", skipna=True
                    )

            for k, m_var in enumerate(
                [
                    mass_cumulative_varname,
                    mass_flux_varname,
                    grounding_line_flux_varname,
                ]
            ):
                sim_ci = axs[k].fill_between(
                    quantiles[0.5].time,
                    quantiles[percentiles[0]][m_var],
                    quantiles[percentiles[1]][m_var],
                    alpha=sim_alpha,
                    color=sim_cmap[1],
                    label=f"""{sim_posterior["ensemble"].values} ({percentile_range:.0f}% credibility interval)""",
                    lw=0,
                )
                if k == 0:
                    sim_cis.append(sim_ci)
                axs[k].plot(
                    quantiles[0.5].time,
                    quantiles[0.5][m_var],
                    lw=0.75,
                    color=sim_cmap[1],
                )

        if sim_posterior is not None:
            y_min, y_max = axs[-1].get_ylim()
            scaler = y_min + (y_max - y_min) * 0.05
            obs_filtered = obs.sel(
                time=slice(f"{filter_range[0]}", f"{filter_range[-1]}")
            )
            filter_range_ds = obs_filtered[mass_cumulative_varname]
            filter_range_ds *= 0
            filter_range_ds += scaler
            _ = filter_range_ds.plot(
                ax=axs[-1], lw=1, ls="solid", color="k", label="Filtering Range"
            )
            x_s = (
                filter_range_ds.time.values[0]
                + (filter_range_ds.time.values[-1] - filter_range_ds.time.values[0]) / 2
            )
            y_s = scaler
            axs[-1].text(
                x_s,
                y_s,
                "Filtering Range",
                horizontalalignment="center",
                fontweight="medium",
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
        if sim_posterior is not None:
            axs[0].set_title(f"{basin} filtered by {filter_var}")
        else:
            axs[0].set_title(f"{basin}")
        axs[1].set_title("")
        axs[1].set_ylabel("Mass balance\n (Gt/yr)")
        axs[-1].set_title("")
        axs[-1].set_ylabel("Grounding Line\nFlux (Gt/yr)")
        axs[-1].set_xlim(np.datetime64("1980-01-01"), np.datetime64("2020-01-01"))
        fig.tight_layout()

        if sim_prior is not None:
            prior_str = "prior"
        else:
            prior_str = ""
        if sim_posterior is not None:
            posterior_str = "_posterior"
        else:
            posterior_str = ""
        prior_posterior_str = prior_str + posterior_str

        fig.savefig(
            pdf_dir
            / Path(
                f"{basin}_mass_accounting_{prior_posterior_str}_filtered_by_{filter_var}.pdf"
            )
        )
        fig.savefig(
            png_dir
            / Path(
                f"{basin}_mass_accounting_{prior_posterior_str}_filtered_by_{filter_var}.png",
                dpi=300,
            )
        )
        plt.close()
        del fig


def plot_outliers(
    filtered_da: xr.DataArray,
    outliers_da: xr.DataArray,
    filename: Union[Path, str],
    fontsize: int = 6,
):
    """
    Plot outliers in the given DataArrays and save the plot to a file.

    This function creates a plot with the filtered data and outliers, and saves the plot
    to the specified filename. The filtered data is plotted in black, and the outliers
    are plotted in red.

    Parameters
    ----------
    filtered_da : xr.DataArray
        The DataArray containing the filtered data.
    outliers_da : xr.DataArray
        The DataArray containing the outliers.
    filename : Union[Path, str]
        The path or filename where the plot will be saved.
    fontsize : int, optional
        The font size for the plot, by default 6.

    Examples
    --------
    >>> filtered_da = xr.DataArray(
    ...     data=[[1, 2, 3], [4, 5, 6]],
    ...     dims=["time", "exp_id"],
    ...     coords={"time": [0, 1], "exp_id": [0, 1, 2]}
    ... )
    >>> outliers_da = xr.DataArray(
    ...     data=[[7, 8, 9], [10, 11, 12]],
    ...     dims=["time", "exp_id"],
    ...     coords={"time": [0, 1], "exp_id": [0, 1, 2]}
    ... )
    >>> plot_outliers(filtered_da, outliers_da, "outliers_plot.png")
    """
    with mpl.rc_context({"font.size": fontsize}):
        fig, ax = plt.subplots(1, 1)
        if outliers_da.size > 0:
            outliers_da.plot(
                hue="exp_id", color=sim_cmap[0], add_legend=False, ax=ax, lw=0.25
            )
        if filtered_da.size > 0:
            filtered_da.plot(
                hue="exp_id", color=sim_cmap[1], add_legend=False, ax=ax, lw=0.25
            )
        fig.savefig(filename)
