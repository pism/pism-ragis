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

# pylint: disable=too-many-positional-arguments

"""
Plot mass balance observations.
"""

import json
import warnings
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from importlib.resources import files
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import toml
import xarray as xr

from pism_ragis.logger import get_logger
from pism_ragis.observations import load_mouginot
from pism_ragis.processing import normalize_cumulative_variables

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


def plot_basin(
    ds: xr.Dataset | list[xr.Dataset],
    basin: str = "GIS",
    sigma: float = 1,
    figsize=(6.4, 6.2),
    fig_dir: str | Path = "figures",
    fontsize: float = 6,
    cmap=["#117733", "#CC6677", "#882255", "#feba80"],
    alpha=0.5,
    reference_date: str = "2003-01-01",
    plot_range: list[str] = ["1986-01-01", "2020-01-01"],
):
    """
    Plot basin data from one or more xarray Datasets.

    This function plots cumulative mass balance, mass balance, surface mass balance,
    and grounding line flux for a specified basin from one or more xarray Datasets.
    The plots are saved as PDF and PNG files in the specified directory.

    Parameters
    ----------
    ds : xr.Dataset or list[xr.Dataset]
        The dataset(s) containing the data to plot.
    basin : str, optional
        The basin to plot, by default "GIS".
    sigma : float, optional
        The number of standard deviations for the uncertainty interval, by default 1.
    figsize : tuple, optional
        The size of the figure, by default (6.4, 6.2).
    fig_dir : str or Path, optional
        The directory to save the figures, by default "figures".
    fontsize : float, optional
        The font size for the plot, by default 6.
    cmap : list, optional
        The color map for the plots, by default ["#CC6677", "#882255", "#feba80"].
    alpha : float, optional
        The alpha value for the uncertainty interval, by default 0.5.
    reference_date : str, optional
        The reference date for cumulative mass change, by default "2003-01-01".
    plot_range : list[str], optional
        The time range for the plot, by default ["1986-01-01", "2020-01-01"].

    Examples
    --------
    >>> ds = xr.Dataset(...)
    >>> plot_basin(ds)
    """

    fig_dir = Path(fig_dir)
    fig_dir.mkdir(exist_ok=True)
    pdf_dir = fig_dir / Path("pdfs")
    pdf_dir.mkdir(parents=True, exist_ok=True)
    png_dir = fig_dir / Path("pngs")
    png_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(ds, xr.Dataset):
        observations = [ds]
    else:
        observations = ds

    vs = [
        "cumulative_mass_balance",
        "mass_balance",
        "surface_mass_balance",
        "grounding_line_flux",
    ]
    vus = [v + "_uncertainty" for v in vs]
    rc_params = {
        "font.size": fontsize,
        # Add other rcParams settings if needed
    }

    with mpl.rc_context(rc=rc_params):

        fig, axs = plt.subplots(
            4,
            1,
            sharex=True,
            figsize=figsize,
            height_ratios=[2, 1, 1, 1],
        )
        fig.subplots_adjust(hspace=0.05, wspace=0.05)

        for k, obs in enumerate(observations):
            obs = obs.sel({"basin": basin}).sel({"time": slice(*plot_range)})
            for l, (v, vu) in enumerate(zip(vs, vus)):
                if (v and vu) in obs.data_vars:
                    _ = axs[l].plot(obs["time"], obs[v], color=cmap[k], lw=0.75)
                    _ = axs[l].fill_between(
                        obs["time"],
                        obs[v] - sigma * obs[vu],
                        obs[v] + sigma * obs[vu],
                        color=cmap[k],
                        alpha=alpha,
                        lw=0,
                        label=obs.name.values,
                    )
        axs[0].legend()

        axs[0].set_ylabel(f"Cumulative mass change\nsince {reference_date} (Gt)")
        axs[0].set_xlabel("")
        axs[1].set_title("")
        axs[1].set_ylabel("Mass balance\n (Gt/yr)")
        axs[-2].set_title("")
        axs[-2].set_ylabel("SMB\n (Gt/yr)")
        axs[-1].set_title("")
        axs[-1].set_ylabel("Grounding Line\nFlux (Gt/yr)")
        axs[-1].set_xlim(np.datetime64(plot_range[0]), np.datetime64(plot_range[1]))
        axs[0].set_title(f"{basin}")
        fig.tight_layout()
        fig.savefig(pdf_dir / Path(f"{basin}.pdf"))
        fig.savefig(png_dir / Path(f"{basin}.png"))
        plt.close(fig)
        del fig


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
        default=None,
    )
    parser.add_argument(
        "--reference_date",
        help="""Reference date.""",
        type=str,
        default="2018-01-1",
    )
    parser.add_argument(
        "--log",
        default="WARNING",
        help="Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    )

    print("================================================================")
    print("Plot mass balance observations")
    print("================================================================\n\n")

    options, unknown = parser.parse_known_args()
    engine = options.engine
    notebook = options.notebook
    parallel = options.parallel
    resampling_frequency = options.resampling_frequency
    reference_date = options.reference_date
    ragis_config_file = Path(
        str(files("pism_ragis.data").joinpath("ragis_config.toml"))
    )
    ragis_config = toml.load(ragis_config_file)
    config = json.loads(json.dumps(ragis_config))

    result_dir = Path(options.result_dir)
    result_dir.mkdir(exist_ok=True)

    fig_dir = result_dir / Path("observations")

    rcparams = {
        "axes.linewidth": 0.25,
        "xtick.direction": "in",
        "xtick.major.size": 2.5,
        "xtick.major.width": 0.25,
        "ytick.direction": "in",
        "ytick.major.size": 2.5,
        "ytick.major.width": 0.25,
        "hatch.linewidth": 0.25,
        "font.size": 6,
    }

    mpl.rcParams.update(rcparams)

    cumulative_vars = list(config["Cumulative Variables"].values())
    cumulative_uncertainty_vars = list(
        config["Cumulative Uncertainty Variables"].values()
    )
    normalize_vars = cumulative_vars + cumulative_uncertainty_vars
    mou_ds = load_mouginot(
        "/Users/andy/base/pism-ragis/data/mass_balance/pnas.1904242116.sd02.xlsx"
    )
    mou_ds = normalize_cumulative_variables(
        mou_ds, normalize_vars, reference_date=reference_date
    )
    mou_ds["name"] = "MOU19"

    man_ds = xr.open_dataset("data/mass_balance/mankoff_greenland_mass_balance.nc")
    man_ds = normalize_cumulative_variables(
        man_ds, normalize_vars, reference_date=reference_date
    )
    man_pub_ds = xr.open_dataset(
        "data/mass_balance/mankoff_greenland_mass_balance_no_smb_err.nc"
    )
    man_pub_ds = normalize_cumulative_variables(
        man_pub_ds, normalize_vars, reference_date=reference_date
    )

    grace_ds = xr.open_dataset("data/mass_balance/grace_greenland_mass_balance.nc")
    grace_ds = normalize_cumulative_variables(
        grace_ds,
        ["cumulative_mass_balance", "cumulative_mass_balance_uncertainty"],
        reference_date=reference_date,
    )

    if resampling_frequency is not None:
        man_ds = man_ds.resample({"time": resampling_frequency}).mean()
        grace_ds = grace_ds.resample({"time": resampling_frequency}).mean()

    man_ds["name"] = "MAN21 (with SMB error)"
    man_pub_ds["name"] = "MAN21 (no SMB error)"
    grace_ds["name"] = "GRACE"
    grace_ds["basin"] = ["GIS"]

    for basin in ["GIS"]:
        plot_basin(
            [man_pub_ds, man_ds, mou_ds, grace_ds],
            basin=basin,
            fig_dir=fig_dir,
            plot_range=["1980", "2020"],
        )

    for basin in ["CW", "CE", "SW", "SE", "NW", "NE", "NO"]:
        plot_basin([man_ds, mou_ds], basin=basin, fig_dir=fig_dir)
