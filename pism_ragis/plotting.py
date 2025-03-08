# Copyright (C) 2024-25 Andy Aschwanden
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

import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import seaborn as sns
import toml
import xarray as xr
from dask.distributed import Client, progress
from tqdm.auto import tqdm

from pism_ragis.decorators import profileit, timeit

mpl.use("Agg")
rcparams = {
    "axes.linewidth": 0.15,
    "xtick.major.size": 2.0,
    "xtick.major.width": 0.15,
    "ytick.major.size": 2.0,
    "ytick.major.width": 0.15,
    "hatch.linewidth": 0.15,
}

mpl.rcParams.update(rcparams)
ragis_config_file = Path(str(files("pism_ragis.data").joinpath("ragis_config.toml")))
ragis_config = toml.load(ragis_config_file)
config = json.loads(json.dumps(ragis_config))

obs_alpha = config["Plotting"]["obs_alpha"]
obs_cmap = config["Plotting"]["obs_cmap"]
sim_alpha = config["Plotting"]["sim_alpha"]
sim_cmap = config["Plotting"]["sim_cmap"]


@timeit
def plot_posteriors(
    df: pd.DataFrame,
    x_order: list[str],
    y_order: list[str] | None = None,
    hue: str | None = "filtered_by",
    figsize: tuple[float, float] | None = (6.4, 5.2),
    fig_dir: str | Path = "figures",
    fontsize: float = 4,
):
    """
    Plot violin plots of posterior distributions.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the data to plot.
    x_order : list[str]
        Order of the variables for the x-axis.
    y_order : list[str] or None, optional
        Order of the basins for the y-axis, by default None.
    hue : str or None, optional
        Variable name for the hue, by default "filtered_by".
    figsize : tuple[float, float] or None, optional
        Size of the figure, by default (6.4, 5.2).
    fig_dir : str or Path, optional
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

    rc_params = {
        "font.size": fontsize,
        "font.family": "DejaVu Sans",
        # Add other rcParams settings if needed
    }

    with mpl.rc_context(rc=rc_params):
        fig, axs = plt.subplots(
            4,
            4,
            sharey=True,
            figsize=figsize,
        )
        fig.subplots_adjust(hspace=0.75, wspace=0.1)
        for k, v in enumerate(x_order):
            legend = bool(k == 0)
            ax = axs.ravel()[k]
            try:
                _ = sns.violinplot(
                    data=df,
                    x=v,
                    y="basin",
                    order=y_order,
                    linewidth=0.25,
                    cut=0,
                    gap=0.1,
                    split=True,
                    inner="quart",
                    hue=hue,
                    orient="h",
                    palette=["#DDCC77", "#CC6677"],
                    ax=ax,
                    legend=legend,
                )
            except:
                pass

            if legend:
                ax.get_legend().remove()

            if k > len(x_order):
                ax.set_visible(False)

        # Create a legend outside the figure at the bottom middle
        handles, labels = axs[0, 0].get_legend_handles_labels()
        legend_main = fig.legend(
            handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=2
        )
        legend_main.set_title(None)
        legend_main.get_frame().set_linewidth(0.0)
        legend_main.get_frame().set_alpha(0.0)

        # Adjust layout to make room for the legend
        fig.tight_layout()
        fig.subplots_adjust(bottom=0.1)

        fn = pdf_dir / Path("posteriors_violinplots.pdf")
        fig.savefig(fn)
        fn = png_dir / Path("posteriors_violinplots.png")
        fig.savefig(fn, dpi=300)
        plt.close()
        del fig


@timeit
def plot_prior_posteriors(
    df: pd.DataFrame,
    figsize: tuple[float, float] | None = (6.4, 3.2),
    fig_dir: str | Path = "figures",
    fontsize: float = 4,
    x_order: list[str] = [],
    bins_dict: dict | None = None,
    group_columns: list = ["basin", "filtered_by"],
):
    """
    Plot histograms of prior and posterior distributions.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the data to plot.
    figsize : tuple[float, float] or None, optional
        Size of the figure, by default (6.4, 3.2).
    fig_dir : str or Path, optional
        Directory to save the figures, by default "figures".
    fontsize : float, optional
        Font size for the plot, by default 4.
    x_order : list[str], optional
        Order of the variables for the x-axis, by default [].
    bins_dict : dict, optional
        Dictionary specifying the number of bins for each variable, by default None.
    group_columns : list, optional
        List of columns to group by, by default ["basin", "filtered_by"].
    """

    plot_dir = fig_dir / Path("basin_histograms")
    plot_dir.mkdir(parents=True, exist_ok=True)
    pdf_dir = plot_dir / Path("pdfs")
    pdf_dir.mkdir(parents=True, exist_ok=True)
    png_dir = plot_dir / Path("pngs")
    png_dir.mkdir(parents=True, exist_ok=True)

    rc_params = {
        "font.size": fontsize,
        "font.family": "DejaVu Sans",
        # Add other rcParams settings if needed
    }

    with mpl.rc_context(rc=rc_params):
        with tqdm(
            desc="Plotting prior and posterior histograms",
            total=len(df.groupby(by=group_columns)),
        ) as progress_bar:
            for (basin, filter_var), m_df in df.groupby(by=group_columns):
                fig, axs = plt.subplots(
                    4,
                    4,
                    sharey=True,
                    figsize=figsize,
                )
                # fig.subplots_adjust(hspace=0.1, wspace=0.1)
                for k, v in enumerate(x_order):
                    if bins_dict is not None:
                        bins = bins_dict.get(v, "auto")
                    else:
                        bins = None
                    legend = bool(k == 1)
                    ax = axs.ravel()[k]
                    try:

                        _ = sns.histplot(
                            data=m_df,
                            x=v,
                            hue="ensemble",
                            hue_order=["Prior", "Posterior"],
                            palette=sim_cmap,
                            bins=bins,
                            common_norm=False,
                            stat="probability",
                            multiple="dodge",
                            alpha=0.8,
                            linewidth=0.2,
                            ax=axs.ravel()[k],
                            legend=legend,
                        )
                        ax.plot([], [])
                    except:
                        pass

                    if legend:
                        ax.get_legend().set_title(None)
                        ax.get_legend().get_frame().set_linewidth(0.0)
                        ax.get_legend().get_frame().set_alpha(0.0)

                # for ax in axs.flat:
                #     if not ax.lines:  # Check if the subplot has any lines plotted
                #         ax.remove()

                for ax in axs.flatten():
                    ax.set_ylabel("")
                    ax.set_ylim(0, 1)
                    ticklabels = ax.get_xticklabels()
                    for tick in ticklabels:
                        tick.set_rotation(15)

                fig.tight_layout()
                fig.set_dpi(600)
                fn = pdf_dir / Path(
                    f"{basin}_prior_posterior_filtered_by_{filter_var}.pdf"
                )
                fig.savefig(fn)
                fn = png_dir / Path(
                    f"{basin}_prior_posterior_filtered_by_{filter_var}.png"
                )
                fig.savefig(fn)
                plt.close()
                del fig
                progress_bar.update()


@timeit
def plot_basins(
    observed: xr.Dataset,
    prior: xr.Dataset,
    posterior: xr.Dataset,
    filter_var: str,
    figsize: tuple[float, float] | None = None,
    filter_range: list[int] = [1990, 2019],
    fig_dir: str | Path = "figures",
    fontsize: int = 6,
    fudge_factor: float = 3.0,
    percentiles: list[float] = [0.025, 0.975],
    reference_date: str = "2020-01-01",
    plot_range: list[int] = [1980, 2020],
    level: int = 2,
    reduced: bool = False,
    config: dict = {},
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
    figsize : tuple[float, float] or None, optional
        Size of the figure, by default None.
    filter_range : list[int], optional
        A list containing the start and end years for filtering, by default [1990, 2019].
    fig_dir : str or Path, optional
        The directory where figures will be saved, by default "figures".
    fontsize : float, optional
        Font size for the plot, by default 6.
    fudge_factor : float, optional
        A multiplicative factor applied to the observed standard deviation to widen the likelihood function,
        allowing for greater tolerance in the matching process, by default 3.0.
    percentiles : list[float], optional
        Percentiles for credibility interval, by default [0.025, 0.975].
    reference_date : str, optional
        The reference date for cumulative mass change, by default "2020-01-01".
    plot_range : list[int], optional
        A list containing the start and end years for plotting, by default [1980, 2020].
    level : int, optional
        The level of detail for the plots, by default 2.
    reduced : bool, optional
        Whether to produce a reduced version of the plots, by default False.
    config : dict, optional
        Configuration dictionary, by default {}.

    Examples
    --------
    >>> observed = xr.Dataset(...)
    >>> prior = xr.Dataset(...)
    >>> posterior = xr.Dataset(...)
    >>> plot_basins(observed, prior, posterior, "grounding_line_flux")
    """

    client = Client()
    observed_scattered = client.scatter(
        [
            observed.sel(basin=basin).sel(
                {"time": slice(str(plot_range[0]), str(plot_range[1]))}
            )
            for basin in observed.basin
        ]
    )
    prior_scattered = client.scatter(
        [
            prior.sel(basin=basin).sel(
                {"time": slice(str(plot_range[0]), str(plot_range[1]))}
            )
            for basin in prior.basin
        ]
    )
    posterior_scattered = client.scatter(
        [
            posterior.sel(basin=basin).sel(
                {"time": slice(str(plot_range[0]), str(plot_range[1]))}
            )
            for basin in posterior.basin
        ]
    )

    futures = client.map(
        plot_obs_sims,
        observed_scattered,
        prior_scattered,
        posterior_scattered,
        reference_date=reference_date,
        config=config,
        filter_var=filter_var,
        filter_range=filter_range,
        figsize=figsize,
        fig_dir=fig_dir,
        fontsize=fontsize,
        fudge_factor=fudge_factor,
        level=level,
        percentiles=percentiles,
        reduced=reduced,
        obs_alpha=obs_alpha,
        sim_alpha=sim_alpha,
    )

    progress(futures)
    client.close()


@timeit
def plot_sensitivity_indices(
    ds: xr.Dataset,
    dim: str = "sensitivity_indices_group",
    indices_var: str = "S1",
    indices_conf_var: str = "S1_conf",
    basin: str = "",
    figsize: tuple[float, float] | None = (3.2, 1.8),
    filter_var: str = "",
    fig_dir: str | Path = "figures",
    fontsize: float = 6,
):
    """
    Plot sensitivity indices with confidence intervals.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset containing sensitivity indices and confidence intervals.
    dim : str, optional
        The dimension name for sensitivity indices groups, by default "sensitivity_indices_group".
    indices_var : str, optional
        The variable name for sensitivity indices in the dataset, by default "S1".
    indices_conf_var : str, optional
        The variable name for confidence intervals of sensitivity indices in the dataset, by default "S1_conf".
    basin : str, optional
        The basin parameter to be used in the plot, by default "".
    figsize : tuple[float, float] or None, optional
        Size of the figure, by default (3.2, 1.8).
    filter_var : str, optional
        The variable used for filtering, by default "".
    fig_dir : str or Path, optional
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

    with mpl.rc_context({"font.size": fontsize, "font.family": "DejaVu Sans"}):

        fig, ax = plt.subplots(1, 1, figsize=figsize)
        for g in ds[dim]:
            indices_da = ds[indices_var].sel({dim: g})
            conf_da = ds[indices_conf_var].sel({dim: g})
            ax.fill_between(
                indices_da.time,
                (indices_da - conf_da),
                (indices_da + conf_da),
                alpha=0.25,
            )
            indices_da.plot(
                hue="sensitivity_indices_group", ax=ax, lw=0.5, label=g.values
            )
        legend = ax.legend(loc="upper left")
        legend.get_frame().set_linewidth(0.0)
        legend.get_frame().set_alpha(0.0)
        ax.set_title(f"{indices_var} for basin {basin} for {filter_var}")
        fn = pdf_dir / f"basin_{basin}_{indices_var}_for_{filter_var}.pdf"
        fig.savefig(fn)
        plt.close()


def plot_obs_sims(
    obs: xr.Dataset,
    sim_prior: xr.Dataset,
    sim_posterior: xr.Dataset,
    config: dict,
    filter_var: str,
    filter_range: list[int] = [1990, 2019],
    figsize: tuple[float, float] | None = None,
    fig_dir: str | Path = "figures",
    fudge_factor: float = 3.0,
    level: int = 4,
    reduced: bool = False,
    reference_date: str = "2020-01-01",
    sim_alpha: float = 0.4,
    obs_alpha: float = 1.0,
    sigma: float = 2,
    percentiles: list[float] = [0.025, 0.975],
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
    filter_range : list[int], optional
        Range of years for filtering, by default [1990, 2019].
    figsize : tuple[float, float] or None, optional
        Size of the figure, by default None.
    fig_dir : str or Path, optional
        Directory to save the figures, by default "figures".
    fudge_factor : float, optional
        A multiplicative factor applied to the observed standard deviation to widen the likelihood function,
        allowing for greater tolerance in the matching process, by default 3.0.
    level : int, optional
        The level of detail for the plots, by default 4.
    reduced : bool, optional
        Whether to produce a reduced version of the plots, by default False.
    reference_date : str, optional
        The reference date for cumulative mass change, by default "2020-01-01".
    sim_alpha : float, optional
        Alpha value for simulation plots, by default 0.4.
    obs_alpha : float, optional
        Alpha value for observation plots, by default 1.0.
    sigma : float, optional
        Sigma value for uncertainty, by default 2.
    percentiles : list[float], optional
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
    smb_flux_varname = config["Flux Variables"]["smb_flux"]
    smb_flux_uncertainty_varname = config["Flux Uncertainty Variables"][
        "smb_flux_uncertainty"
    ]

    if filter_var == "grounding_line_flux":
        m = -1
    elif filter_var == "surface_mass_balance":
        m = -2
    else:
        m = 1

    p_vars = [mass_cumulative_varname]
    if level >= 2:
        p_vars.append(grounding_line_flux_varname)
    if level >= 3:
        p_vars.append(mass_flux_varname)
    if level >= 4:
        p_vars.append(smb_flux_varname)

    with mpl.rc_context({"font.size": fontsize, "font.family": "DejaVu Sans"}):

        fig, axs = plt.subplots(
            level,
            1,
            sharex=True,
            figsize=figsize,
            height_ratios=[2] + [1] * (level - 1),
        )
        fig.subplots_adjust(hspace=0.05, wspace=0.05)
        if level == 1:
            ax_0 = axs
            ax_last = axs
        else:
            ax_0 = axs[0]
            ax_last = axs[-1]

        obs_ci = ax_0.fill_between(
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

        if mass_flux_varname in obs.data_vars and level >= 3:
            axs[1].fill_between(
                obs["time"],
                obs[mass_flux_varname] - sigma * obs[mass_flux_uncertainty_varname],
                obs[mass_flux_varname] + sigma * obs[mass_flux_uncertainty_varname],
                color=obs_cmap[0],
                alpha=obs_alpha,
                lw=0,
            )

        if smb_flux_varname in obs.data_vars and level >= 4:
            axs[-2].fill_between(
                obs["time"],
                obs[smb_flux_varname] - sigma * obs[smb_flux_uncertainty_varname],
                obs[smb_flux_varname] + sigma * obs[smb_flux_uncertainty_varname],
                color=obs_cmap[0],
                alpha=obs_alpha,
                lw=0,
            )

        if grounding_line_flux_varname in obs.data_vars and level >= 2:
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

        if (sim_posterior is not None) and (filter_var in obs.data_vars):
            obs_filtered = obs.sel(
                time=slice(f"{filter_range[0]}", f"{filter_range[-1]}")
            )
            if filter_var in p_vars:
                obs_filtered_ci = axs[m].fill_between(
                    obs_filtered["time"],
                    obs_filtered[filter_var]
                    - fudge_factor * obs_filtered[filter_var + "_uncertainty"],
                    obs_filtered[filter_var]
                    + fudge_factor * obs_filtered[filter_var + "_uncertainty"],
                    alpha=1.0,
                    edgecolor="k",
                    facecolor="none",
                    hatch="///",
                    lw=0.25,
                    label="Filtering Range",
                )
                l_f = axs[m].legend(handles=[obs_filtered_ci], loc="lower left")
                l_f.get_frame().set_linewidth(0.0)
                l_f.get_frame().set_alpha(0.0)

        sim_cis = []
        if sim_prior is not None:
            sim_prior = sim_prior[p_vars + ["ensemble"]]
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", r"All-NaN (slice|axis) encountered")
                quantiles = {}
                for q in [percentiles[0], 0.5, percentiles[1]]:
                    quantiles[q] = sim_prior.utils.drop_nonnumeric_vars().quantile(
                        q, dim="exp_id", skipna=True
                    )

            if level >= 2:
                for k, m_var in enumerate(p_vars):
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
            else:
                m_var = "cumulative_mass_balance"
                sim_ci = ax_0.fill_between(
                    quantiles[0.5].time,
                    quantiles[percentiles[0]][m_var],
                    quantiles[percentiles[1]][m_var],
                    alpha=sim_alpha,
                    color=sim_cmap[0],
                    label=f"""{sim_prior["ensemble"].values} ({percentile_range:.0f}% credibility interval)""",
                    lw=0,
                )
                sim_cis.append(sim_ci)
        if sim_posterior is not None:
            sim_posterior = sim_posterior[p_vars + ["ensemble"]]
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", r"All-NaN (slice|axis) encountered")
                quantiles = {}
                for q in [percentiles[0], 0.5, percentiles[1]]:
                    quantiles[q] = sim_posterior.utils.drop_nonnumeric_vars().quantile(
                        q, dim="exp_id", skipna=True
                    )

            if level >= 2:
                for k, m_var in enumerate(p_vars):
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
            else:
                m_var = "cumulative_mass_balance"
                sim_ci = ax_0.fill_between(
                    quantiles[0.5].time,
                    quantiles[percentiles[0]][m_var],
                    quantiles[percentiles[1]][m_var],
                    alpha=sim_alpha,
                    color=sim_cmap[1],
                    label=f"""{sim_posterior["ensemble"].values} ({percentile_range:.0f}% credibility interval)""",
                    lw=0,
                )
                sim_cis.append(sim_ci)
                ax_0.plot(
                    quantiles[0.5].time,
                    quantiles[0.5][m_var],
                    lw=0.75,
                    color=sim_cmap[1],
                )
        legend = ax_0.legend(
            handles=[obs_ci, *sim_cis],
        )
        legend.get_frame().set_linewidth(0.0)
        legend.get_frame().set_alpha(0.0)

        if not reduced:
            ax_0.add_artist(legend)

        ax_0.xaxis.set_tick_params(labelbottom=False)
        ax_0.set_ylabel(f"""Mass change\nsince {reference_date.split("-")[0]} (Gt)""")
        ax_last.xaxis.set_tick_params(labelbottom=True)

        if sim_posterior is not None:
            ax_0.set_title(f"{basin} filtered by {filter_var}")
        else:
            ax_0.set_title(f"{basin}")

        if level >= 2:
            axs[-1].set_title("")
            axs[-1].set_ylabel("Grounding Line\nFlux (Gt/yr)")
        if level >= 3:
            axs[1].set_title("")
            axs[1].set_ylabel("Mass balance\n (Gt/yr)")
        if level >= 3:
            axs[-2].set_title("")
            axs[-2].set_ylabel("SMB\n (Gt/yr)")

        ax_last.set_xlim(np.datetime64("1980-01-01"), np.datetime64("2020-01-01"))
        # Set major ticks format to every 5 years
        ax_last.xaxis.set_major_locator(mdates.YearLocator(5))
        ax_last.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

        if reduced:
            if level >= 2:
                for l in range(level):
                    axs[l].set_title("")
                    axs[l].set_ylabel("")
                    try:
                        axs[l].get_legend().remove()
                    except:
                        pass
            else:
                ax_0.set_title("")
                ax_0.set_ylabel("")
                try:
                    ax_0.get_legend().remove()
                except:
                    pass

        if sim_prior is not None:
            prior_str = "prior"
        else:
            prior_str = ""
        if sim_posterior is not None:
            posterior_str = "_posterior"
        else:
            posterior_str = ""
        prior_posterior_str = prior_str + posterior_str

        fig.tight_layout()
        fig.set_dpi(600)
        fig.savefig(
            pdf_dir
            / Path(
                f"{basin}_mass_accounting_{prior_posterior_str}_filtered_by_{filter_var}_{level}.pdf"
            )
        )
        fig.savefig(
            png_dir
            / Path(
                f"{basin}_mass_accounting_{prior_posterior_str}_filtered_by_{filter_var}_{level}.png",
            )
        )

        # # Create a new figure for the legend
        # fig_legend = plt.figure()
        # # Add the legend from the original plot to the new figure
        # fig_legend.legend(*ax_0.get_legend_handles_labels(), loc='center')

        # # Remove the axes from the legend figure
        # fig_legend.gca().axis('off')
        # # Save the legend figure as a separate PDF
        # fig_legend.savefig('legend.pdf', format='pdf', bbox_inches='tight')
        plt.close()
        del fig


def plot_outliers(
    filtered_da: xr.DataArray,
    outliers_da: xr.DataArray,
    filename: Path | str,
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
    filename : Path or str
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
