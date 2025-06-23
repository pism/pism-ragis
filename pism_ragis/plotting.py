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

# pylint: disable=unused-argument,unused-import,too-many-positional-arguments,eval-used

"""
Module for data plotting.
"""
import json
import warnings
from importlib.resources import files
from pathlib import Path

import cartopy.crs as ccrs
import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import seaborn as sns
import toml
import xarray as xr
from dask.distributed import Client, progress
from matplotlib import colors
from tqdm.auto import tqdm

from pism_ragis.decorators import profileit, timeit

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


def qgis2cmap(
    filename: Path | str,
    num_levels: int = 256,  # Renamed from N to num_levels
    name: str = "my colormap",
) -> colors.LinearSegmentedColormap:
    """
    Read a colormap exported from QGIS raster layers and return a matplotlib.colors.LinearSegmentedColormap.

    Parameters
    ----------
    filename : Path or str
        The path to the QGIS colormap file.
    num_levels : int, optional
        The number of RGB quantization levels, by default 256.
    name : str, optional
        The name of the colormap, by default "my colormap".

    Returns
    -------
    matplotlib.colors.LinearSegmentedColormap
        The matplotlib colormap.
    """
    m_data = np.loadtxt(filename, skiprows=2, delimiter=",")[:, :-1]
    values_scaled = (m_data[:, 0] - np.min(m_data[:, 0])) / (
        np.max(m_data[:, 0]) - np.min(m_data[:, 0])
    )
    colors_scaled = m_data[:, 1::] / 255.0
    m_colors = [(values_scaled[k], colors_scaled[k]) for k in range(len(values_scaled))]
    cmap = colors.LinearSegmentedColormap.from_list(name, m_colors, N=num_levels)

    return cmap


def register_colormaps(path: str | Path | None = None) -> None:
    """
    Register colormaps from text files.

    Parameters
    ----------
    path : str, Path, optional
        The directory where the colormap text files are located. If not provided, the
        'pism_ragis.data' directory is used.

    Examples
    --------
    >>> register_colormaps()
    >>> register_colormaps('/path/to/colormap/files')
    """
    if path is not None:
        cmap_files = Path(path).glob("*.txt")
    else:
        cmap_files = Path(str(files("pism_ragis.data").joinpath("*.txt"))).parent.glob(
            "*.txt"
        )
    for cmap_file in cmap_files:
        name = cmap_file.name.removesuffix(".txt")
        cmap = qgis2cmap(cmap_file, name=name)
        plt.colormaps.register(cmap)


register_colormaps()


def plot_mapplane(
    da: xr.DataArray,
    fname: str | Path,
    figwidth: float = 6.4,
    fontsize: float = 6,
    **kwargs,
):
    """
    Plot a 2D map of a DataArray on a polar stereographic projection.

    Parameters
    ----------
    da : xr.DataArray
        The data array to be plotted.
    fname : str or Path
        The filename or path where the plot will be saved.
    figwidth : float, optional
        The width of the figure in inches, by default 3.2.
    fontsize : float, optional
        The font size for the plot, by default 6.
    **kwargs : dict
        Additional keyword arguments passed to the `plot` method of the DataArray.

    Notes
    -----
    - The function uses a North Polar Stereographic projection with specific settings.
    - Colormaps are registered before plotting.
    - The aspect ratio of the figure is adjusted dynamically based on the data bounds.

    Examples
    --------
    >>> import xarray as xr
    >>> da = xr.DataArray([[1, 2], [3, 4]])
    >>> plot_mapplane(da, "output.png")
    """

    rc_params = {
        "font.size": fontsize,
        "font.family": "DejaVu Sans",
        # Add other rcParams settings if needed
    }
    # ar = 1.0  # initial aspect ratio for first trial
    wi = figwidth  # width in inches
    # hi = wi * ar  # height in inches
    crs = ccrs.NorthPolarStereo(
        central_longitude=-45, true_scale_latitude=70, globe=None
    )
    with mpl.rc_context(rc=rc_params):

        p = da.plot(
            transform=crs,
            cbar_kwargs={"shrink": 0.7},
            subplot_kws={"projection": crs},
            **kwargs,
        )

        for ax in p.axs.flat:
            ax.gridlines(
                draw_labels={"top": "x", "left": "y"},
                dms=True,
                x_inline=False,
                y_inline=False,
                rotate_labels=20,
                ls="dotted",
                color="k",
                xlabel_style={"size": fontsize},
                ylabel_style={"size": fontsize},
            )

            # Get proper ratio here
            xmin, xmax = ax.get_xbound()
            ymin, ymax = ax.get_ybound()
            y2x_ratio = (ymax - ymin) / (xmax - xmin)
            ax.set_aspect(
                y2x_ratio,
                adjustable="box",
            )
        fig = p.fig
        fig.set_figwidth(wi)
        fig.savefig(fname, dpi=300)
        plt.close()
    del fig


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
            total=len(df.groupby(by=group_columns, observed=True)),
        ) as progress_bar:
            for (basin, filter_var), m_df in df.groupby(
                by=group_columns, observed=True
            ):
                fig, axs = plt.subplots(
                    4,
                    4,
                    sharey=True,
                    figsize=figsize,
                )
                fig.subplots_adjust(
                    hspace=0.6, wspace=0.1, left=0.05, right=0.95, bottom=0.1, top=0.95
                )
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
    x_lim: list[int] = [1980, 2020],
    **kwargs,
):
    """
    Plot basins using observed, prior, and posterior datasets.

    Parameters
    ----------
    observed : xr.Dataset
        The observed dataset.
    prior : xr.Dataset
        The prior dataset.
    posterior : xr.Dataset
        The posterior dataset.
    x_lim : list[int], optional
        A list containing the start and end years for plotting, by default [1980, 2020].
    **kwargs : dict
        Additional keyword arguments for the plotting function.
    """

    client = Client()
    observed_scattered = client.scatter(
        [
            observed.sel(basin=basin).sel({"time": slice(str(x_lim[0]), str(x_lim[1]))})
            for basin in observed.basin
        ]
    )
    prior_scattered = client.scatter(
        [
            prior.sel(basin=basin).sel({"time": slice(str(x_lim[0]), str(x_lim[1]))})
            for basin in prior.basin
        ]
    )
    posterior_scattered = client.scatter(
        [
            posterior.sel(basin=basin).sel(
                {"time": slice(str(x_lim[0]), str(x_lim[1]))}
            )
            for basin in posterior.basin
        ]
    )

    futures = client.map(
        plot_timeseries,
        observed_scattered,
        prior_scattered,
        posterior_scattered,
        x_lim=x_lim,
        **kwargs,
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

    with mpl.rc_context({"font.size": fontsize, "font.family": "Arial"}):

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


def plot_timeseries(
    obs: xr.Dataset,
    sim_prior: xr.Dataset | None = None,
    sim_posterior: xr.Dataset | None = None,
    config: dict = {},
    filter_var: str | None = None,
    filter_range: list[int] = [1990, 2019],
    figsize: tuple[float, float] | None = None,
    fig_dir: str | Path = "figures",
    fudge_factor: float = 3.0,
    plot_vars: str | list = ["cumulative_mass_flux", "grounding_line_flux"],
    x_lim: list[int] = [1980, 2020],
    y_lim: list | list[list] | None = None,
    reference_date: str = "2020-01-01",
    sim_alpha: float = 0.4,
    obs_alpha: float = 1.0,
    sigma: float = 2,
    percentiles: list[float] = [0.025, 0.975],
    fontsize: float = 6,
    add_lineplot: bool = False,
    add_median: bool = False,
) -> None:
    """
    Plot cumulative mass balance and grounding line flux.

    Parameters
    ----------
    obs : xr.Dataset
        Observational dataset.
    sim_prior : xr.Dataset or None, optional
        Prior simulation dataset, by default None.
    sim_posterior : xr.Dataset or None, optional
        Posterior simulation dataset, by default None.
    config : dict, optional
        Configuration dictionary containing variable names, by default {}.
    filter_var : str or None, optional
        Variable used for filtering, by default None.
    filter_range : list[int], optional
        Range of years for filtering, by default [1990, 2019].
    figsize : tuple[float, float] or None, optional
        Size of the figure, by default None.
    fig_dir : str or Path, optional
        Directory to save the figures, by default "figures".
    fudge_factor : float, optional
        A multiplicative factor applied to the observed standard deviation to widen the likelihood function,
        allowing for greater tolerance in the matching process, by default 3.0.
    plot_vars : str or list, optional
        Variables to plot, by default ["cumulative_mass_flux", "grounding_line_flux"].
    x_lim : list[int], optional
        A list containing the start and end years for plotting, by default [1980, 2020].
    y_lim : list or list[list] or None, optional
        Y-axis limits for the plots, by default None.
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
    add_lineplot : bool, optional
        Whether to add line plots for individual simulations, by default False.
    add_median : bool, optional
        Whether to add simulated median, by default False.
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

    v_dict = config["Flux Variables"] | config["Cumulative Variables"]
    vu_dict = (
        config["Flux Uncertainty Variables"]
        | config["Cumulative Uncertainty Variables"]
    )

    if isinstance(plot_vars, str):
        plot_vars = [plot_vars]
    p_vars = [v_dict[k] for k in plot_vars]
    level = len(plot_vars)

    if sim_prior is not None:
        if "ensemble" in sim_prior.data_vars:
            sim_prior = sim_prior[p_vars + ["ensemble"]]
        else:
            sim_prior = sim_prior[p_vars]

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", r"All-NaN (slice|axis) encountered")
            sim_prior_quantiles = {}
            for q in [percentiles[0], 0.5, percentiles[1]]:
                sim_prior_quantiles[
                    q
                ] = sim_prior.utils.drop_nonnumeric_vars().quantile(
                    q, dim="exp_id", skipna=True
                )
    if sim_posterior is not None:
        if "ensemble" in sim_posterior.data_vars:
            sim_posterior = sim_posterior[p_vars + ["ensemble"]]
        else:
            sim_posterior = sim_posterior[p_vars]
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", r"All-NaN (slice|axis) encountered")
            sim_posterior_quantiles = {}
            for q in [percentiles[0], 0.5, percentiles[1]]:
                sim_posterior_quantiles[
                    q
                ] = sim_posterior.utils.drop_nonnumeric_vars().quantile(
                    q, dim="exp_id", skipna=True
                )

    with mpl.rc_context({"font.size": fontsize, "font.family": "Arial"}):

        fig, axs = plt.subplots(
            level,
            1,
            sharex=True,
            figsize=figsize,
            height_ratios=[(1 + np.sqrt(5)) / 2] + [1] * (level - 1),
        )
        fig.subplots_adjust(hspace=0.05, wspace=0.05)
        if level == 1:
            ax_0 = axs
            ax_last = axs
        else:
            ax_0 = axs[0]
            ax_last = axs[-1]

        for k, plot_var in enumerate(plot_vars):
            p_var = v_dict[plot_var]
            pu_var = vu_dict[plot_var + "_uncertainty"]

            if level == 1:
                ax = axs
            else:
                ax = axs[k]

            if p_var and pu_var in obs.data_vars:
                obs_cis = []
                sim_cis = []

                obs_ci = ax.fill_between(
                    obs["time"],
                    obs[p_var] - sigma * obs[pu_var],
                    obs[p_var] + sigma * obs[pu_var],
                    color=obs_cmap[0],
                    alpha=obs_alpha,
                    lw=0,
                    label=rf"Observed ({sigma}-$\sigma$ uncertainty)",
                )
                obs_cis.append(obs_ci)
                if sim_prior is not None:
                    sim_ci = ax.fill_between(
                        sim_prior_quantiles[0.5].time,
                        sim_prior_quantiles[percentiles[0]][p_var],
                        sim_prior_quantiles[percentiles[1]][p_var],
                        alpha=sim_alpha,
                        color=sim_cmap[0],
                        lw=0,
                        label=f"""{sim_prior["ensemble"].values} ({percentile_range:.0f}% credibility interval)""",
                    )
                    sim_cis.append(sim_ci)
                    if add_lineplot:
                        sim_prior[p_var].plot(
                            hue="exp_id",
                            color=sim_cmap[0],
                            add_legend=False,
                            ax=ax,
                            lw=0.20,
                            ls="dotted",
                            dashes=(1, 5),
                        )
                    if add_median:
                        sim_prior_quantiles[0.5][p_var].plot(
                            color=sim_cmap[0],
                            add_legend=False,
                            ax=ax,
                            lw=1,
                            ls="solid",
                        )

                if sim_posterior is not None:
                    sim_ci = ax.fill_between(
                        sim_posterior_quantiles[0.5].time,
                        sim_posterior_quantiles[percentiles[0]][p_var],
                        sim_posterior_quantiles[percentiles[1]][p_var],
                        alpha=sim_alpha,
                        color=sim_cmap[1],
                        lw=0,
                        label=f"""{sim_posterior["ensemble"].values} ({percentile_range:.0f}% credibility interval)""",
                    )
                    sim_cis.append(sim_ci)
                    if add_lineplot:
                        sim_posterior[p_var].plot(
                            hue="exp_id",
                            color=sim_cmap[1],
                            add_legend=False,
                            ax=ax,
                            lw=0.20,
                            ls="dotted",
                            dashes=(1, 5),
                        )
                    if add_median:
                        sim_posterior_quantiles[0.5][p_var].plot(
                            color=sim_cmap[1],
                            add_legend=False,
                            ax=ax,
                            lw=1,
                            ls="solid",
                        )

                if (
                    (filter_var is not None)
                    and (filter_var == p_var)
                    and (filter_var in obs.data_vars)
                ):
                    obs_filtered = obs.sel(
                        time=slice(f"{filter_range[0]}", f"{filter_range[-1]}")
                    )

                    obs_filtered_ci = ax.fill_between(
                        obs_filtered["time"],
                        obs_filtered[p_var] - fudge_factor * obs_filtered[pu_var],
                        obs_filtered[p_var] + fudge_factor * obs_filtered[pu_var],
                        alpha=1.0,
                        edgecolor="k",
                        facecolor="none",
                        hatch="///",
                        lw=0.25,
                        label="Filtering Range",
                    )
                    l_f = ax.legend(handles=[obs_filtered_ci], loc="lower left")
                    l_f.get_frame().set_linewidth(0.0)
                    l_f.get_frame().set_alpha(0.0)

            ylabel_str = config["Plotting"][plot_var]
            if "Cumulative" in ylabel_str:
                ylabel = eval(f"f'''{ylabel_str}'''")
            else:
                ylabel = ylabel_str
            ax.set_ylabel(ylabel)
            ax.set_title(None)
            ax.set_xlabel(None)

        handles = [*obs_cis, *sim_cis]
        legend = ax_0.legend(
            handles=handles,
        )
        legend.get_frame().set_linewidth(0.0)
        legend.get_frame().set_alpha(0.0)

        ax_0.xaxis.set_tick_params(labelbottom=False)
        ax_last.xaxis.set_tick_params(labelbottom=True)

        if sim_posterior is not None:
            ax_0.set_title(f"{basin} filtered by {filter_var}")
        else:
            ax_0.set_title(f"{basin}")

        ax_last.set_xlabel("Year")
        ax_last.set_xlim(
            np.datetime64(f"{x_lim[0]}-01-01"),
            np.datetime64(f"{x_lim[1]}-01-01"),
        )
        if y_lim is not None:
            if level == 1:
                ax_last.set_yim(**y_lim)
            else:
                for k, ax_ in enumerate(axs):
                    ax_.set_ylim(y_lim[k])
        # Set major ticks format to every 5 years
        ax_last.xaxis.set_major_locator(mdates.YearLocator(5))
        ax_last.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

        if sim_prior is not None:
            prior_str = "prior"
        else:
            prior_str = ""
        if sim_posterior is not None:
            posterior_str = "_posterior"
        else:
            posterior_str = ""
        prior_posterior_str = prior_str + posterior_str

        fig_name = f"{basin}_mass_accounting_{prior_posterior_str}"
        if filter_var is not None:
            fig_name += f"_filtered_by_{filter_var}"
        fig.tight_layout()
        fig.set_dpi(600)
        fig.savefig(pdf_dir / Path(f"{fig_name}.pdf"))
        fig.savefig(
            png_dir
            / Path(
                f"{fig_name}.png",
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
