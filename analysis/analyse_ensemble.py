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
from typing import Any, Hashable, List, Mapping, Union
import warnings

import dask
import numpy as np
import pandas as pd
import pylab as plt
import seaborn as sns
import toml
import xarray as xr
from dask.distributed import Client, LocalCluster, progress
from joblib import Parallel, delayed
from SALib.analyze import delta
from tqdm.auto import tqdm

from pism_ragis.filter import particle_filter
from pism_ragis.processing import tqdm_joblib


def plot_obs_sims(
    obs: xr.Dataset,
    sim_prior: xr.Dataset,
    sim_posterior: xr.Dataset,
    config,
    filtering_var: str,
    filter_range: List = [1990, 2019],
    fig_dir: Union[str, Path] = "figures",
):
    """
    Plot figure with cumulative mass balance and ice discharge and climatic
    mass balance fluxes.
    """

    Path(fig_dir).mkdir(exist_ok=True)
    obs_filtered = obs.sel(time=slice(str(filter_range[0]), str(filter_range[-1])))

    basin = obs.basin.values
    mass_cumulative_varname = config["Cumulative Variables"]["mass_cumulative"]
    mass_cumulative_uncertainty_varname = mass_cumulative_varname + "_uncertainty"
    discharge_flux_varname = config["Flux Variables"]["discharge_flux"]
    discharge_flux_uncertainty_varname = discharge_flux_varname + "_uncertainty"
    smb_flux_varname = config["Flux Variables"]["smb_flux"]
    smb_flux_uncertainty_varname = discharge_flux_varname + "_uncertainty"

    plt.rcParams["font.size"] = 6

    fig, axs = plt.subplots(
        3, 1, sharex=True, figsize=(6.2, 4.2), height_ratios=[2, 1, 1]
    )
    fig.subplots_adjust(hspace=0.05, wspace=0.05)

    obs_ci = axs[0].fill_between(
        obs["time"],
        obs[mass_cumulative_varname] - obs[mass_cumulative_uncertainty_varname],
        obs[mass_cumulative_varname] + obs[mass_cumulative_uncertainty_varname],
        color=obs_cmap[0],
        alpha=1.0,
        lw=0,
        label="1-$\sigma$",
    )

    obs_filtered_ci = axs[0].fill_between(
        obs_filtered["time"],
        obs_filtered[mass_cumulative_varname]
        - obs_filtered[mass_cumulative_uncertainty_varname],
        obs_filtered[mass_cumulative_varname]
        + obs_filtered[mass_cumulative_uncertainty_varname],
        color=obs_cmap[1],
        alpha=1.0,
        lw=0,
        label="Filter Interval",
    )

    axs[1].fill_between(
        obs["time"],
        obs[discharge_flux_varname] - obs[discharge_flux_uncertainty_varname],
        obs[discharge_flux_varname] + obs[discharge_flux_uncertainty_varname],
        color=obs_cmap[0],
        alpha=1.0,
        lw=0,
    )

    axs[1].fill_between(
        obs_filtered["time"],
        obs_filtered[discharge_flux_varname]
        - obs_filtered[discharge_flux_uncertainty_varname],
        obs_filtered[discharge_flux_varname]
        + obs_filtered[discharge_flux_uncertainty_varname],
        color=obs_cmap[1],
        alpha=1.0,
        lw=0,
    )

    axs[2].fill_between(
        obs["time"],
        obs[smb_flux_varname] - obs[smb_flux_uncertainty_varname],
        obs[smb_flux_varname] + obs[smb_flux_uncertainty_varname],
        color=obs_cmap[0],
        alpha=1.0,
        lw=0,
    )

    sim_cis = []
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', r'All-NaN (slice|axis) encountered')
        quantiles = {}
        for q in [0.16, 0.5, 0.84]:
            quantiles[q] = sim_prior.drop_vars(
                ["pism_config", "run_stats", "Ensemble"], errors="ignore"
            ).quantile(q, dim="exp_id", skipna=True)

    for k, m_var in enumerate(
        [mass_cumulative_varname, discharge_flux_varname, smb_flux_varname]
    ):
        sim_prior[m_var].plot(
            hue="exp_id",
            color=sim_cmap[1],
            ax=axs[k],
            lw=0.1,
            alpha=0.4,
            add_legend=False,
        )
        sim_ci = axs[k].fill_between(
            quantiles[0.5].time,
            quantiles[0.16][m_var],
            quantiles[0.84][m_var],
            alpha=0.4,
            color=sim_cmap[1],
            label=sim_prior["Ensemble"].values,
            lw=0,
        )
        if k == 0:
            sim_cis.append(sim_ci)

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', r'All-NaN (slice|axis) encountered')
        quantiles = {}
        for q in [0.16, 0.5, 0.84]:
            quantiles[q] = sim_posterior.drop_vars(
                ["pism_config", "run_stats", "Ensemble"], errors="ignore"
            ).quantile(q, dim="exp_id", skipna=True)

    for k, m_var in enumerate(
        [mass_cumulative_varname, discharge_flux_varname, smb_flux_varname]
    ):
        sim_ci = axs[k].fill_between(
            quantiles[0.5].time,
            quantiles[0.16][m_var],
            quantiles[0.84][m_var],
            alpha=0.4,
            color=sim_cmap[3],
            label=sim_posterior["Ensemble"].values,
            lw=0,
        )
        if k == 0:
            sim_cis.append(sim_ci)
        axs[k].plot(
            quantiles[0.5].time, quantiles[0.5][m_var], lw=0.75, color=sim_cmap[3]
        )
    legend_obs = axs[0].legend(
        handles=[obs_ci, obs_filtered_ci], loc="lower left", title="Observed"
    )
    legend_obs.get_frame().set_linewidth(0.0)
    legend_obs.get_frame().set_alpha(0.0)

    legend_sim = axs[0].legend(
        handles=sim_cis,
        loc="center left",
        title="Simulated (66% c.i.)",
    )
    legend_sim.get_frame().set_linewidth(0.0)
    legend_sim.get_frame().set_alpha(0.0)

    axs[0].add_artist(legend_obs)
    axs[0].add_artist(legend_sim)

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



def normalize_cumulative_variables(
    ds: xr.Dataset, variables, reference_year: int = 1992
) -> xr.Dataset:
    """
    Normalize cumulative variables in an xarray Dataset by subtracting their values at a reference year.

    Parameters
    ----------
    ds : xr.Dataset
        The xarray Dataset containing the cumulative variables to be normalized.
    variables : str or list of str
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
    ds[variables] -= ds[variables].sel(time=f"{reference_year}-01-01", method="nearest")
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


def select_experiments(df: pd.DataFrame, ids_to_select: List[int]) -> pd.DataFrame:
    """
    Select rows from a DataFrame based on a list of experiment IDs, including duplicates.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing experiment data.
    ids_to_select : List[int]
        A list of experiment IDs to select from the DataFrame. Duplicates in this list
        will result in duplicate rows in the output DataFrame.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the selected rows, including duplicates as specified
        in `ids_to_select`.
    """
    # Create a DataFrame with the rows to select
    selected_rows = df[df["exp_id"].isin(ids_to_select)]

    # Repeat the indices according to the number of times they appear in ids_to_select
    repeated_indices = selected_rows.index.repeat(
        [ids_to_select.count(id) for id in selected_rows["exp_id"]]
    )

    # Select the rows based on the repeated indices
    return df.loc[repeated_indices]


def load_ensemble(
    result_dir: Union[Path, str],
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
    result_dir = Path(result_dir)
    m_files = result_dir.glob(glob_str)
    with dask.config.set(**{"array.slicing.split_large_chunks": True}):
        print("Loading ensemble files")
        ds = xr.open_mfdataset(m_files, parallel=True, chunks="auto")
        if "time" in ds["pism_config"].coords:
            ds["pism_config"] = ds["pism_config"].isel(time=0).drop_vars("time")
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


def simplify_calving(my_str: str):
    """
    Simplify ocean
    """
    return int(my_str.split("_")[3])


def transpose_dataframe(df, exp_id):
    """
    Transpose dataframe.
    """
    param_names = df["pism_config_axis"]
    df = df[["pism_config"]].T
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
        "--mankoff_url",
        help="""Path to Mankoff basins mass balance.""",
        type=str,
        default="data/mankoff/mankoff_mass_balance.nc",
    )
    parser.add_argument(
        "--temporal_range",
        help="""Time slice to extract.""",
        type=str,
        nargs=2,
        default=None,
    )
    parser.add_argument(
        "--filter_range",
        help="""Time slice used for the Particle Filter. Default="1990 2019". """,
        type=str,
        nargs=2,
        default="1990 2019",
    )
    parser.add_argument(
        "--crs",
        help="""Coordinate reference system. Default is EPSG:3413.""",
        type=str,
        default="EPSG:3413",
    )
    parser.add_argument(
        "--reference_year",
        help="""Reference year.""",
        type=int,
        default=1986,
    )
    parser.add_argument(
        "--n_jobs", help="""Number of parallel jobs.""", type=int, default=4
    )
    parser.add_argument(
        "--fudge_factor", help="""Observational uncertainty multiplier. Default=3""", type=int, default=3
    )

    options = parser.parse_args()
    crs = options.crs
    fudge_factor = options.fudge_factor
    reference_year = options.reference_year
    filter_start_year, filter_end_year = options.filter_range.split(" ")

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
    plt.rcParams["font.size"] = 6

    flux_vars = ragis_config["Flux Variables"]
    flux_uncertainty_vars = {
        k + "_uncertainty": v + "_uncertainty" for k, v in flux_vars.items()
    }
    cumulative_vars = ragis_config["Cumulative Variables"]
    cumulative_uncertainty_vars = {
        k + "_uncertainty": v + "_uncertainty" for k, v in cumulative_vars.items()
    }

    ds = load_ensemble(result_dir).sortby("basin").dropna(dim="exp_id").load()
    ds = standarize_variable_names(ds, ragis_config["PISM"])
    ds[ragis_config["Cumulative Variables"]["discharge_cumulative"]] = ds[
        ragis_config["Flux Variables"]["discharge_flux"]
    ].cumsum() / len(ds.time)
    ds[ragis_config["Cumulative Variables"]["smb_cumulative"]] = ds[
        ragis_config["Flux Variables"]["smb_flux"]
    ].cumsum() / len(ds.time)
    ds = normalize_cumulative_variables(
        ds,
        list(ragis_config["Cumulative Variables"].values()),
        reference_year=reference_year,
    )
    fig, ax = plt.subplots(1, 1)
    ds.sel(time=slice(str(filter_start_year), str(filter_end_year))).sel(
        basin="GIS", ensemble_id="RAGIS"
    ).ice_discharge.plot(hue="exp_id", add_legend=False, ax=ax, lw=0.5)
    fig.savefig("ice_discharge_unfiltered.pdf")

    lower_bound = -750
    upper_bound = -100

    filter_ds = (
        ds.sel(basin="GIS", ensemble_id="RAGIS")
        .drop_vars(["pism_config", "run_stats"], errors="ignore")["ice_discharge"]
        .sel(time=slice(str(filter_start_year), str(filter_end_year)))
    )
    filter_days_in_month = filter_ds.time.dt.days_in_month
    filter_wgts = filter_days_in_month.groupby(
        "time.year"
    ) / filter_days_in_month.groupby("time.year").sum(dim="time")
    outlier_filter = (filter_ds * filter_wgts).resample({"time": "YS"}).sum()
    outlier_filter = filter_ds
    # Identify exp_id values that fall within the 99% credibility interval
    mask = (outlier_filter >= lower_bound) & (outlier_filter <= upper_bound)
    mask = ~(~mask).any(dim="time")
    filtered_ds = outlier_filter.sel(exp_id=mask)
    filtered_exp_ids = filtered_ds.exp_id.values
    print(len(filtered_exp_ids))
    ds = ds.sel(exp_id=filtered_exp_ids)

    fig, ax = plt.subplots(1, 1)
    ds.sel(time=slice(str(filter_start_year), str(filter_end_year))).sel(
        basin="GIS", ensemble_id="RAGIS"
    ).ice_discharge.plot(hue="exp_id", add_legend=False, ax=ax, lw=0.5)
    fig.savefig("ice_discharge_filtered.pdf")

    pism_config = ds.sel(basin="GIS").sel(pism_config_axis=params).pism_config

    uq_df = pism_config.to_dataframe()

    ragis = pd.concat(
        [
            transpose_dataframe(df, exp_id)
            for exp_id, df in uq_df.reset_index().groupby(by="exp_id")
        ]
    ).reset_index(drop=True)

    observed = (
        xr.open_dataset(options.mankoff_url).sel(time=slice("1986", "2021")).load()
    )
    observed = observed.sortby("basin")
    observed = normalize_cumulative_variables(
        observed,
        list(cumulative_vars.values()) + list(cumulative_uncertainty_vars.values()),
        reference_year,
    )

    sim_alpha = 0.5
    obs_cmap = sns.color_palette("crest", n_colors=4)
    # obs_cmap = ["#DC267F", "#DC267F"]
    sim_cmap = ["#FFB000", "#FE6100"]
    sim_cmap = sns.color_palette("flare", n_colors=4)
    obs_cmap = ["0.8", "0.7"]
    hist_cmap = ["#a6cee3", "#1f78b4", "#b2df8a", "#33a02c"]
    hist_cmap = ["#a6cee3", "#1f78b4"]

    resampling_freq = "MS"
    freq_dict = {"MS": "month", "YS": "year"}

    observed_days_in_month = observed["time"].dt.days_in_month
    observed_days_in_month = observed.time.dt.days_in_month

    observed_wgts = 1 / (observed_days_in_month)
    observed_resampled = (
        (observed * observed_wgts).resample(time=resampling_freq)
        .sum(dim="time")
        .rolling(time=13)
        .mean()
        )

    # observed_resampled = observed_resampled.rolling(time=13).mean()
    # simulated = (
    #     ds.drop_vars(["pism_config", "run_stats"], errors="ignore")
    #     .rolling(time=13)
    #     .mean()
    # )
    # simulated = xr.merge([simulated, ds[["pism_config", "run_stats"]]])

    # simulated_days_in_month = simulated["time"].dt.days_in_month
    # simulated_wgts = (
    #     simulated_days_in_month.groupby(f"""time.{freq_dict[resampling_freq]}""")
    #     / simulated_days_in_month.groupby(
    #         f"""time.{freq_dict[resampling_freq]}"""
    #     ).sum()
    # )
    # simulated_resampled = (
    #     (
    #         simulated.drop_vars(["pism_config", "run_stats"], errors="ignore")
    #         * simulated_wgts
    #     )
    #     .resample(time=resampling_freq)
    #     .sum(dim="time")
    # )
    simulated = ds
    simulated_resampled = (
        simulated.drop_vars(["pism_config", "run_stats"], errors="ignore")
        .resample(time="MS")
        .sum(dim="time")
        .rolling(time=13)
        .mean()
    )
    # simulated_resampled = (
    #     simulated.drop_vars(["pism_config", "run_stats"], errors="ignore")
    #     .rolling(time=13)
    #     .mean()
    # )
    simulated_resampled["pism_config"] = simulated["pism_config"]

    # Apply the conversion function to each column
    ragis = ragis.apply(convert_column_to_float)
    for col in ["surface.given.file", "ocean.th.file", "calving.rate_scaling.file"]:
        ragis[col] = ragis[col].apply(simplify)
    ragis["surface.given.file"] = ragis["surface.given.file"].apply(simplify_climate)
    ragis["ocean.th.file"] = ragis["ocean.th.file"].apply(simplify_ocean)
    ragis["calving.rate_scaling.file"] = ragis["calving.rate_scaling.file"].apply(
        simplify_calving
    )

    ragis["Ensemble"] = "Prior"


    simulated_filtered_all = {}
    filtered_all = {}
    for obs_mean_var, obs_std_var, sim_var in zip(
        flux_vars.values(),
        flux_uncertainty_vars.values(),
        flux_vars.values(),
    ):
        print(f"Particle filtering using {obs_mean_var}")
        filtered_ids = particle_filter(
            simulated=simulated_resampled.sel(time=slice(str(filter_start_year), str(filter_end_year))),
            observed=observed_resampled.sel(time=slice(str(filter_start_year), str(filter_end_year))),
            fudge_factor=fudge_factor,
            n_samples=len(simulated.exp_id),
            obs_mean_var=obs_mean_var,
            obs_std_var=obs_std_var,
            sim_var=sim_var,
        )
        filtered_ids["basin"] = filtered_ids["basin"].astype("<U3")
        simulated_filtered = (
            simulated_resampled.sel(exp_id=filtered_ids)
            .drop_vars("exp_id")
            .rename_dims({"exp_id_sampled": "exp_id"})
        )
        simulated_filtered["Ensemble"] = "Posterior"
        simulated_filtered_all[obs_mean_var] = simulated_filtered

        sim_prior = simulated_resampled
        sim_prior["Ensemble"] = "Prior"
        sim_posterior = simulated_filtered
        sim_posterior["Ensemble"] = "Posterior"

        with tqdm_joblib(
            tqdm(desc="Plotting basins", total=len(observed_resampled.basin))
        ) as progress_bar:
            result = Parallel(n_jobs=options.n_jobs)(
                delayed(plot_obs_sims)(
                    observed_resampled.sel(basin=basin).rolling(time=13).mean(),
                    sim_prior.sel(basin=basin, ensemble_id="RAGIS"),
                    sim_posterior.sel(basin=basin, ensemble_id="RAGIS"),
                    config=ragis_config,
                    filtering_var=obs_mean_var,
                    filter_range=[filter_start_year, filter_end_year],
                    fig_dir=result_dir / Path("figures"),
                )
                for basin in observed_resampled.basin
            )

        ids_to_select = list(filtered_ids.sel(basin="GIS", ensemble_id="RAGIS").values)

        ragis_filtered = select_experiments(ragis, ids_to_select)
        ragis_filtered["Ensemble"] = "Posterior"

        filtered_all[obs_mean_var] = pd.concat([ragis, ragis_filtered]).rename(
            columns=params_short_dict
        )

    for filtering_var, simulated_filtered in simulated_filtered_all.items():
        for basin in simulated_filtered.basin.values:
            obs = observed.sel(basin=basin).rolling(time=390).mean()
            sim_prior = (
                simulated.sel(basin=basin, ensemble_id="RAGIS")
                .drop_vars(["pism_config", "run_stats"], errors="ignore")
                .rolling(time=13)
                .mean()
            )
            sim_prior["Ensemble"] = "Prior"
            sim_posterior = (
                simulated_filtered.sel(basin=basin, ensemble_id="RAGIS")
                .drop_vars(["pism_config", "run_stats"], errors="ignore")
                .rolling(time=13)
                .mean()
            )
            sim_posterior["Ensemble"] = "Posterior"

            posterior_df = filtered_all[filtering_var]

            n_params = len(params_short_dict)
            fig, axs = plt.subplots(
                5,
                3,
                figsize=[6.2, 7.2],
            )
            fig.subplots_adjust(hspace=1.5, wspace=0.25)

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
            fn = (
                result_dir
                / Path("figures")
                / Path(f"{basin}_hist_prior_posterior_filtered_by_{filtering_var}.pdf")
            )
            fig.savefig(fn)
            plt.close()

    ensemble_df = ragis.drop(columns=["Ensemble", "exp_id"], errors="ignore")
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
            df["pism_config_axis"] = problem["names"]
            df.set_index("pism_config_axis", inplace=True)
        return xr.Dataset.from_dataframe(df)

    to_analyze = ds.sel(time=slice("1980-01-01", "2020-01-01"))
    print("Calculating Sensitivity Indices")
    cluster = LocalCluster(n_workers=options.n_jobs, threads_per_worker=1)
    client = Client(cluster, asynchronous=True)
    print(f"Open client in browser: {client.dashboard_link}")
    dim = "time"
    all_delta_indices_list = []
    for basin in to_analyze.basin.values:
        for filter_var in [
            "mass_balance",
            "ice_discharge",
        ]:
            responses = to_analyze.sel(basin=basin, ensemble_id="RAGIS")[filter_var]
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
            delta_indices = delta_indices.expand_dims("basin", axis=1)
            delta_indices["basin"] = [basin]
            delta_indices = delta_indices.expand_dims("filtered_by", axis=2)
            delta_indices["filtered_by"] = [filter_var]
            all_delta_indices_list.append(delta_indices)
    all_delta_indices: xr.Dataset = xr.merge(all_delta_indices_list)
    
    # Extract the prefix from each coordinate value
    prefixes = [
        name.split(".")[0] for name in all_delta_indices.pism_config_axis.values
    ]

    # Add the prefixes as a new coordinate
    all_delta_indices = all_delta_indices.assign_coords(
        prefix=("pism_config_axis", prefixes)
    )

    sensitivity_indices_groups = {"surface": "Climate", "atmosphere": "Climate", "ocean": "Ocean",
                                  "calving": "Calving", "frontal_melt": "Frontal Melt",
                                  "basal_resistance": "Flow", "basal_yield_stress": "Flow", "stress_balance": "Flow"}
    
  

    si_prefixes = [
        sensitivity_indices_groups[name] for name in all_delta_indices.prefix.values
    ]
    all_delta_indices = all_delta_indices.assign_coords(
        sensitivity_indices_group=("pism_config_axis", si_prefixes)
    )
    # Group by the new coordinate and compute the sum for each group
    aggregated_data = all_delta_indices.groupby("sensitivity_indices_group").sum().rolling(time=13).mean()

    for basin in aggregated_data.basin.values:
        for filter_var in aggregated_data.filtered_by.values:
            fig, ax = plt.subplots(1, 1)
            aggregated_data.sel(filtered_by=filter_var, basin=basin).S1.plot(
                hue="sensitivity_indices_group", ax=ax
            )
            ax.set_title(f"S1 for {basin} filtered by {filter_var}")
            fn = (
                result_dir
                / Path("figures")
                / Path(f"S1_{basin}_filtered_by_{filter_var}.pdf")
            )
            fig.savefig(fn)
            plt.close()
