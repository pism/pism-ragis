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
"""
Analyze RAGIS ensemble
"""

from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from importlib.resources import files
from pathlib import Path
from typing import Any, Hashable, Mapping

import dask
import numpy as np
import pandas as pd
import pylab as plt
import seaborn as sns
import toml
import xarray as xr
from dask.distributed import Client, LocalCluster, progress
from SALib.analyze import delta

from pism_ragis.analysis import resample_ensemble_by_data
from pism_ragis.observations import load_imbie, load_imbie_2021, load_mouginot


def normalize_cumulative_variables(
    ds: xr.Dataset, cumulative_variables, reference_year: int = 1992
) -> xr.Dataset:
    """
    Normalize cumulative variables in an xarray Dataset by subtracting their values at a reference year.

    Parameters
    ----------
    ds : xr.Dataset
        The xarray Dataset containing the cumulative variables to be normalized.
    cumulative_variables : str or list of str
        The name(s) of the cumulative variables to be normalized.
    reference_year : int, optional
        The reference year to use for normalization. Default is 1992.

    Returns
    -------
    xr.Dataset
        The xarray Dataset with normalized cumulative variables.

    Examples
    --------
    >>> import xarray as xr
    >>> import pandas as pd
    >>> time = pd.date_range("1990-01-01", "1995-01-01", freq="A")
    >>> data = xr.Dataset({
    ...     "cumulative_var": ("time", [10, 20, 30, 40, 50, 60]),
    ... }, coords={"time": time})
    >>> normalize_cumulative_variables(data, "cumulative_var", reference_year=1992)
    <xarray.Dataset>
    Dimensions:         (time: 6)
    Coordinates:
      * time            (time) datetime64[ns] 1990-12-31 1991-12-31 ... 1995-12-31
    Data variables:
        cumulative_var  (time) int64 0 10 20 30 40 50
    """
    ds[cumulative_variables] -= ds[cumulative_variables].sel(
        time=f"{reference_year}-01-01", method="nearest"
    )
    return ds


def standarize_variable_names(
    ds: xr.Dataset, name_dict: Mapping[Any, Hashable] | None
) -> xr.Dataset:
    """
    Standardize variable names in an xarray Dataset.

    Parameters
    ----------
    ds : xr.Dataset
        The xarray Dataset whose variable names need to be standardized.
    name_dict : Mapping[Any, Hashable] or None
        A dictionary mapping the current variable names to the new standardized names.
        If None, no renaming is performed.

    Returns
    -------
    xr.Dataset
        The xarray Dataset with standardized variable names.

    Examples
    --------
    >>> import xarray as xr
    >>> ds = xr.Dataset({'temp': ('x', [1, 2, 3]), 'precip': ('x', [4, 5, 6])})
    >>> name_dict = {'temp': 'temperature', 'precip': 'precipitation'}
    >>> standarize_variable_names(ds, name_dict)
    <xarray.Dataset>
    Dimensions:      (x: 3)
    Dimensions without coordinates: x
    Data variables:
        temperature   (x) int64 1 2 3
        precipitation (x) int64 4 5 6
    """
    return ds.rename_vars(name_dict)


def load_ensemble(
    result_dir: Path,
    glob_str: str = "basin*.nc",
) -> xr.Dataset:
    """
    Load an ensemble of NetCDF files into an xarray Dataset.

    Parameters
    ----------
    result_dir : Path
        The directory containing the NetCDF files.
    glob_str : str, optional
        The glob pattern to match the NetCDF files. Default is "basin*.nc".
    ensemble_id : str, optional
        The identifier for the ensemble. Default is "RAGIS".

    Returns
    -------
    xr.Dataset
        The loaded xarray Dataset containing the ensemble data.

    Notes
    -----
    This function uses Dask to load the dataset in parallel and handle large chunks efficiently.
    """
    m_files = result_dir.glob(glob_str)
    with dask.config.set(**{"array.slicing.split_large_chunks": True}):
        print("Loading ensemble files")
        ds = xr.open_mfdataset(m_files, parallel=True, chunks="auto")
        if "time" in ds["config"].coords:
            ds["config"] = ds["config"].isel(time=0).drop_vars("time")
        print("Done")
        return ds


def select_experiment(ds, exp_id, n):
    """
    Reset the experiment id.
    """
    exp = ds.sel(exp_id=exp_id)
    exp["exp_id"] = n
    return exp


def simplify(my_str: str) -> str:
    """
    Simplify string
    """
    return Path(my_str).name


def convert_column_to_float(column):
    """
    Convert column to numeric if possible.
    """
    try:
        return pd.to_numeric(column, errors="raise")
    except ValueError:
        return column


def simplify_climate(my_str: str):
    """
    Simplify climate
    """
    if "MAR" in my_str:
        return "MAR"
    else:
        return "HIRHAM"


def simplify_ocean(my_str: str):
    """
    Simplify ocean
    """
    return "-".join(my_str.split("_")[1:2])


def transpose_dataframe(df, exp_id):
    """
    Transpose dataframe.
    """
    param_names = df["config_axis"]
    df = df[["config"]].T
    df.columns = param_names
    df["exp_id"] = exp_id
    return df


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
        "--imbie_url",
        help="""Path to IMBIE excel file.""",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--mouginot_url",
        help="""Path to Mouginot 2019 excel file.""",
        type=str,
        default="/mnt/storstrommen/data/mouginot/pnas.1904242116.sd02.xlsx",
    )
    parser.add_argument(
        "--temporal_range",
        help="""Time slice to extract.""",
        type=str,
        nargs=2,
        default=None,
    )
    parser.add_argument(
        "--crs",
        help="""Coordinate reference system. Default is EPSG:3413.""",
        type=str,
        default="EPSG:3413",
    )

    options = parser.parse_args()
    crs = options.crs

    colorblind_colors = [
        "#882255",
        "#DDCC77",
        "#CC6677",
        "#AA4499",
        "#88CCEE",
        "#44AA99",
        "#117733",
    ]
    ragis_config_file = Path(
        str(files("pism_ragis.data").joinpath("ragis_config.toml"))
    )
    ragis_config = toml.load(ragis_config_file)
    all_params_dict = ragis_config["Parameters"]

    params = [
        "calving.vonmises_calving.sigma_max",
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

    reference_year = 1980

    ds = load_ensemble(result_dir)
    ds = standarize_variable_names(ds, ragis_config["PISM"])
    ds = normalize_cumulative_variables(
        ds,
        ragis_config["Cumulative Variables"]["mass_cumulative"],
        reference_year=reference_year,
    )
    config = ds.sel(basin="GIS").sel(config_axis=params).config
    uq_df = config.to_dataframe()

    ragis = pd.concat(
        [
            transpose_dataframe(df, exp_id)
            for exp_id, df in uq_df.reset_index().groupby(by="exp_id")
        ]
    ).reset_index(drop=True)

    # Load observations
    if options.imbie_url is not None:
        imbie = load_imbie(url=Path(options.imbie_url))
    else:
        imbie = load_imbie()
    imbie_2021 = load_imbie_2021()

    mou = load_mouginot(url=Path(options.mouginot_url), norm_year=reference_year)

    flux_vars = ragis_config["Flux Variables"]
    flux_uncertainty_vars = {
        k + "_uncertainty": v + "_uncertainty" for k, v in flux_vars.items()
    }
    cumulative_vars = ragis_config["Cumulative Variables"]
    cumulative_uncertainty_vars = {
        k + "_uncertainty": v + "_uncertainty" for k, v in cumulative_vars.items()
    }
    imbie_mean = imbie.sel(time=slice("1992-1-1", "2012-1-1"))[
        list(cumulative_uncertainty_vars.values())
        + list(flux_uncertainty_vars.values())
    ].mean()
    mou_mean = mou.sel(time=slice("1992-1-1", "2012-1-1"))[
        list(cumulative_uncertainty_vars.values())
        + list(flux_uncertainty_vars.values())
    ].mean()
    u_ratio = imbie_mean / mou_mean / 2
    sigma_adjusted = np.maximum(u_ratio, 1)

    mou_adjusted = mou
    mou_adjusted[
        list(cumulative_uncertainty_vars.values())
        + list(flux_uncertainty_vars.values())
    ] *= sigma_adjusted[
        list(cumulative_uncertainty_vars.values())
        + list(flux_uncertainty_vars.values())
    ]
    mou_gis = mou_adjusted.sel(basin="GIS")

    observed = mou_adjusted.sel(basin="GIS")
    observed_days_in_month = observed["time"].dt.days_in_month
    observed_wgts = (
        observed_days_in_month.groupby("time.year")
        / observed_days_in_month.groupby("time.year").sum()
    )
    observed_yearly = (observed * observed_wgts).resample(time="YS").sum(dim="time")

    simulated = ds.sel(basin="GIS", ensemble_id="RAGIS").drop_vars(
        "config", errors="ignore"
    )
    simulated["ensemble_id"] = "Prior"

    simulated_days_in_month = simulated["time"].dt.days_in_month
    simulated_wgts = (
        simulated_days_in_month.groupby("time.year")
        / simulated_days_in_month.groupby("time.year").sum()
    )
    simulated_yearly = (simulated * simulated_wgts).resample(time="YS").sum(dim="time")

    # Apply the conversion function to each column
    ragis = ragis.apply(convert_column_to_float)
    for col in ["surface.given.file", "ocean.th.file"]:
        ragis[col] = ragis[col].apply(simplify)
    ragis["surface.given.file"] = ragis["surface.given.file"].apply(simplify_climate)
    ragis["ocean.th.file"] = ragis["ocean.th.file"].apply(simplify_ocean)

    ragis["Ensemble"] = "Prior"

    simulated_resampled_all = {}
    resampled_all = {}
    for obs_mean_var, obs_std_var, sim_var in zip(
        flux_vars.values(),
        flux_uncertainty_vars.values(),
        flux_vars.values(),
    ):
        print(f"Particle filtering using {obs_mean_var}")
        resampled_ids = resample_ensemble_by_data(
            observed_yearly,
            simulated_yearly,
            start_date="1992-01-01",
            end_date="2018-01-01",
            fudge_factor=3,
            n_samples=len(simulated.exp_id),
            obs_mean_var=obs_mean_var,
            obs_std_var=obs_std_var,
            sim_var=sim_var,
        )
        print(resampled_ids)
        simulated_resampled = xr.concat(
            [
                select_experiment(simulated, exp_id, k)
                for k, exp_id in enumerate(resampled_ids)
            ],
            dim="exp_id",
        )
        simulated_resampled["ensemble_id"] = "Posterior"
        simulated_resampled_all[obs_mean_var] = simulated_resampled
        resampled_df = [ragis[ragis["exp_id"] == k] for k in resampled_ids]

        ragis_resampled = pd.concat(resampled_df)
        ragis_resampled["Ensemble"] = "Posterior"

        resampled_all[obs_mean_var] = pd.concat([ragis, ragis_resampled]).rename(
            columns=params_short_dict
        )

    posterior_df = resampled_all["mass_balance"]
    plt.rcParams["font.size"] = 6
    obs_cmap = sns.color_palette("crest", n_colors=4)
    obs_cmap = ["0.4", "0.0", "0.6", "0.0"]
    sim_cmap = sns.color_palette("flare", n_colors=4)
    hist_cmap = ["#a6cee3", "#1f78b4", "#b2df8a", "#33a02c"]
    hist_cmap = ["#a6cee3", "#1f78b4"]

    n_params = len(params_short_dict)
    fig, axs = plt.subplots(
        7,
        2,
        figsize=[6.2, 9.2],
    )
    fig.subplots_adjust(hspace=0.75, wspace=0.25)

    for k, v in enumerate(params_short_dict.values()):
        try:
            sns.histplot(
                data=posterior_df,
                x=v,
                hue="Ensemble",
                palette=hist_cmap,
                common_norm=False,
                stat="density",
                multiple="dodge",
                alpha=0.8,
                linewidth=0.2,
                ax=axs.ravel()[k],
                legend=False,
            )
        except:
            pass
    for ax in axs.flatten():
        ticklabels = ax.get_xticklabels()
        for tick in ticklabels:
            tick.set_rotation(45)
    fig.savefig("hist.pdf")

    sim_alpha = 0.5

    for resampling_var, simulated_resampled in simulated_resampled_all.items():
        mass_cumulative_varname = ragis_config["Cumulative Variables"][
            "mass_cumulative"
        ]
        mass_cumulative_uncertainty_varname = mass_cumulative_varname + "_uncertainty"
        discharge_flux_varname = ragis_config["Flux Variables"]["discharge_flux"]
        discharge_flux_uncertainty_varname = discharge_flux_varname + "_uncertainty"
        smb_flux_varname = ragis_config["Flux Variables"]["smb_flux"]
        smb_flux_uncertainty_varname = discharge_flux_varname + "_uncertainty"

        fig, axs = plt.subplots(
            3, 1, sharex=True, figsize=(6.2, 4.2), height_ratios=[2, 1, 1]
        )
        mou_ci = axs[0].fill_between(
            mou_gis["time"],
            mou_gis[mass_cumulative_varname]
            - mou_gis[mass_cumulative_uncertainty_varname],
            mou_gis[mass_cumulative_varname]
            + mou_gis[mass_cumulative_uncertainty_varname],
            color=obs_cmap[3],
            alpha=0.4,
            lw=0,
            label="Mouginot adj/IMBIE",
        )

        axs[1].fill_between(
            mou_gis["time"],
            mou_gis[discharge_flux_varname]
            - mou_gis[discharge_flux_uncertainty_varname],
            mou_gis[discharge_flux_varname]
            + mou_gis[discharge_flux_uncertainty_varname],
            color=obs_cmap[3],
            alpha=0.4,
            lw=0,
        )
        axs[2].fill_between(
            mou_gis["time"],
            mou_gis[smb_flux_varname] - mou_gis[smb_flux_uncertainty_varname],
            mou_gis[smb_flux_varname] + mou_gis[smb_flux_uncertainty_varname],
            color=obs_cmap[3],
            alpha=0.4,
            lw=0,
        )

        sim_cis = []
        quantiles = {}
        for q in [0.16, 0.5, 0.84]:
            quantiles[q] = (
                simulated.load()
                .drop_vars("config", errors="ignore")
                .quantile(q, dim="exp_id", skipna=True)
            )

        for k, m_var in enumerate(
            [mass_cumulative_varname, discharge_flux_varname, smb_flux_varname]
        ):
            simulated[m_var].plot(
                hue="exp_id", color=sim_cmap[1], ax=axs[k], lw=0.1, add_legend=False
            )
            sim_ci = axs[k].fill_between(
                quantiles[0.5].time,
                quantiles[0.16][m_var],
                quantiles[0.84][m_var],
                alpha=0.4,
                color=sim_cmap[1],
                label=simulated["ensemble_id"].values,
                lw=0,
            )
            if k == 0:
                sim_cis.append(sim_ci)

        quantiles = {}
        for q in [0.16, 0.5, 0.84]:
            quantiles[q] = (
                simulated_resampled.load()
                .drop_vars("config", errors="ignore")
                .quantile(q, dim="exp_id", skipna=True)
            )

        for k, m_var in enumerate(
            [mass_cumulative_varname, discharge_flux_varname, smb_flux_varname]
        ):
            sim_ci = axs[k].fill_between(
                quantiles[0.5].time,
                quantiles[0.16][m_var],
                quantiles[0.84][m_var],
                alpha=0.4,
                color=sim_cmap[3],
                label=simulated_resampled["ensemble_id"].values,
                lw=0,
            )
            if k == 0:
                sim_cis.append(sim_ci)
            axs[k].plot(
                quantiles[0.5].time, quantiles[0.5][m_var], lw=1, color=sim_cmap[3]
            )
        legend_obs = axs[0].legend(handles=[mou_ci], loc="lower left", title="Observed")
        legend_obs.get_frame().set_linewidth(0.0)
        legend_obs.get_frame().set_alpha(0.0)

        legend_sim = axs[0].legend(
            handles=sim_cis,
            loc="center left",
            title="Simulated (13-month rolling mean)",
        )
        legend_sim.get_frame().set_linewidth(0.0)
        legend_sim.get_frame().set_alpha(0.0)

        axs[0].add_artist(legend_obs)
        axs[0].add_artist(legend_sim)

        axs[0].set_ylim(-10_000, 500)
        axs[1].set_ylim(-750, 0)
        axs[2].set_ylim(0, 750)
        axs[0].xaxis.set_tick_params(labelbottom=False)

        axs[0].set_ylabel(f"Cumulative mass\nloss since {reference_year} (Gt)")
        axs[0].set_xlabel("")
        axs[0].set_title(f"GIS resampled by {resampling_var}")
        axs[1].set_title("")
        axs[2].set_title("")
        axs[1].set_ylabel("Grounding Line\nFlux (Gt/yr)")
        axs[2].set_ylabel("Climatic Mass\nBalance (Gt/yr)")
        axs[-1].set_xlim(np.datetime64("1980-01-01"), np.datetime64("2021-01-01"))
        fig.tight_layout()
        fig.savefig(f"GIS_mass_accounting_filtered_by_{resampling_var}.pdf")

    fig, axs = plt.subplots(
        4,
        2,
        figsize=[6.2, 7.2],
    )
    fig.subplots_adjust(hspace=0.75, wspace=0.25)

    for k, basin in enumerate(mou_adjusted.basin):
        obs = mou_adjusted.sel(basin=basin)
        sim = ds.sel(ensemble_id="RAGIS", basin=basin)
        ax = axs.ravel()[k]
        quantiles = {}
        for q in [0.16, 0.5, 0.84]:
            quantiles[q] = (
                sim.load().drop_vars("config").quantile(q, dim="exp_id", skipna=True)
            )
        m_var = "cumulative_mass_balance"
        sim_ci = ax.fill_between(
            quantiles[0.5].time,
            quantiles[0.16][m_var],
            quantiles[0.84][m_var],
            alpha=0.2,
            color=sim_cmap[1],
            label=simulated["ensemble_id"].values,
            lw=0,
        )
        mou_ci = ax.fill_between(
            obs["time"],
            obs[mass_cumulative_varname] - obs[mass_cumulative_uncertainty_varname],
            obs[mass_cumulative_varname] + obs[mass_cumulative_uncertainty_varname],
            color=obs_cmap[0],
            alpha=0.5,
            lw=0,
            label="Mouginot adj/IMBIE",
        )
        ax.set_title(basin.values)
        ax.set_xlim(np.datetime64("1980-01-01"), np.datetime64("2021-01-01"))
        fig.tight_layout()
    ensemble_df = ragis.drop(columns=["Ensemble", "exp_id"], errors="ignore")
    climate_dict = {
        v: k for k, v in enumerate(ensemble_df["surface.given.file"].unique())
    }
    ocean_dict = {v: k for k, v in enumerate(ensemble_df["ocean.th.file"].unique())}
    ensemble_df["surface.given.file"] = ensemble_df["surface.given.file"].map(
        climate_dict
    )
    ensemble_df["ocean.th.file"] = ensemble_df["ocean.th.file"].map(ocean_dict)
    problem = {
        "num_vars": len(ensemble_df.columns),
        "names": ensemble_df.columns,  # Parameter names
        "bounds": zip(
            ensemble_df.min().values,
            ensemble_df.max().values,
        ),  # Parameter bounds
    }

    def get_delta(response, problem, ensemble_df):
        """
        Perform SALib delta analysis.
        """
        try:
            delta_analysis = delta.analyze(
                problem,
                ensemble_df.values,
                response,
                num_resamples=10,
                seed=0,
                print_to_console=False,
            )
            df = delta_analysis.to_df()
        except:
            delta_analysis = {
                key: np.empty(problem["num_vars"]) + np.nan
                for key in ["delta", "delta_conf", "S1", "S1_conf"]
            }
            df = pd.DataFrame.from_dict(delta_analysis)
            df["config_axis"] = problem["names"]
            df.set_index("config_axis", inplace=True)
        return xr.Dataset.from_dataframe(df)

    to_analyze = ds.sel(time=slice("1980-01-01", "2020-01-01")).rolling(time=13).mean()
    print("Calculating Sensitivity Indices")
    cluster = LocalCluster(n_workers=8, threads_per_worker=1)
    with Client(cluster, asynchronous=True) as client:
        print(f"Open client in browser: {client.dashboard_link}")
        dim = "time"
        all_delta_indices = []
        for response_var in [
            "mass_balance",
            "ice_discharge",
        ]:
            responses = to_analyze.sel(basin="GIS")[response_var]
            responses_scattered = client.scatter(
                [responses.isel(time=k).to_numpy() for k in range(len(responses[dim]))]
            )

            futures = client.map(
                get_delta,
                responses_scattered,
                problem=problem,
                ensemble_df=ensemble_df,
            )
            progress(futures)
            result = client.gather(futures)

            delta_indices = xr.concat([r.expand_dims(dim) for r in result], dim=dim)
            delta_indices[dim] = responses[dim]
            delta_indices.expand_dims("name", axis=1)
            delta_indices["name"] = [response_var]
            all_delta_indices.append(delta_indices)
        all_delta_indices = xr.concat(all_delta_indices, dim="name")
