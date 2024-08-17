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
from typing import Any, Hashable, List, Mapping, Union

import dask
import numpy as np
import pandas as pd
import pylab as plt
import seaborn as sns
import toml
import xarray as xr
from dask.distributed import Client, LocalCluster, progress
from joblib import Parallel, delayed
from tqdm.auto import tqdm

import pism_ragis.processing as prp
from pism_ragis.analysis import delta_analysis
from pism_ragis.filtering import particle_filter

sim_alpha = 0.5
sim_cmap = sns.color_palette("crest", n_colors=4).as_hex()[0:3:2]
obs_alpha = 1.0
obs_cmap = ["0.8", "0.7"]
hist_cmap = ["#a6cee3", "#1f78b4"]


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
) -> None:
    """
    Plot figure with cumulative mass balance and ice discharge and climatic
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
        alpha=obs_alpha,
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
        alpha=obs_alpha,
        lw=0,
        label="Filter Interval",
    )

    axs[1].fill_between(
        obs["time"],
        obs[discharge_flux_varname] - obs[discharge_flux_uncertainty_varname],
        obs[discharge_flux_varname] + obs[discharge_flux_uncertainty_varname],
        color=obs_cmap[0],
        alpha=obs_alpha,
        lw=0,
    )

    axs[1].fill_between(
        obs_filtered["time"],
        obs_filtered[discharge_flux_varname]
        - obs_filtered[discharge_flux_uncertainty_varname],
        obs_filtered[discharge_flux_varname]
        + obs_filtered[discharge_flux_uncertainty_varname],
        color=obs_cmap[1],
        alpha=obs_alpha,
        lw=0,
    )

    axs[2].fill_between(
        obs["time"],
        obs[smb_flux_varname] - obs[smb_flux_uncertainty_varname],
        obs[smb_flux_varname] + obs[smb_flux_uncertainty_varname],
        color=obs_cmap[0],
        alpha=obs_alpha,
        lw=0,
    )

    sim_cis = []
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", r"All-NaN (slice|axis) encountered")
        quantiles = {}
        for q in [0.16, 0.5, 0.84]:
            quantiles[q] = sim_prior.utils.drop_nonnumeric_vars().quantile(
                q, dim="exp_id", skipna=True
            )

    for k, m_var in enumerate(
        [mass_cumulative_varname, discharge_flux_varname, smb_flux_varname]
    ):
        sim_prior[m_var].plot(
            hue="exp_id",
            color=sim_cmap[0],
            ax=axs[k],
            lw=0.1,
            alpha=sim_alpha,
            add_legend=False,
        )
        sim_ci = axs[k].fill_between(
            quantiles[0.5].time,
            quantiles[0.16][m_var],
            quantiles[0.84][m_var],
            alpha=sim_alpha,
            color=sim_cmap[0],
            label=sim_prior["Ensemble"].values,
            lw=0,
        )
        if k == 0:
            sim_cis.append(sim_ci)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", r"All-NaN (slice|axis) encountered")
        quantiles = {}
        for q in [0.16, 0.5, 0.84]:
            quantiles[q] = sim_posterior.utils.drop_nonnumeric_vars().quantile(
                q, dim="exp_id", skipna=True
            )

    for k, m_var in enumerate(
        [mass_cumulative_varname, discharge_flux_varname, smb_flux_varname]
    ):
        sim_ci = axs[k].fill_between(
            quantiles[0.5].time,
            quantiles[0.16][m_var],
            quantiles[0.84][m_var],
            alpha=sim_alpha,
            color=sim_cmap[1],
            label=sim_posterior["Ensemble"].values,
            lw=0,
        )
        if k == 0:
            sim_cis.append(sim_ci)
        axs[k].plot(
            quantiles[0.5].time, quantiles[0.5][m_var], lw=0.75, color=sim_cmap[1]
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
        "--crs",
        help="""Coordinate reference system. Default is EPSG:3413.""",
        type=str,
        default="EPSG:3413",
    )
    parser.add_argument(
        "--filter_range",
        help="""Time slice used for the Particle Filter. Default="1990 2019". """,
        type=str,
        nargs=2,
        default="1990 2019",
    )
    parser.add_argument(
        "--fudge_factor",
        help="""Observational uncertainty multiplier. Default=3""",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--n_jobs", help="""Number of parallel jobs.""", type=int, default=4
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
        help="""Resampling data to resampling_frequency for particle filtering. Default is "MS".""",
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
    crs = options.crs
    filter_start_year, filter_end_year = options.filter_range.split(" ")
    fudge_factor = options.fudge_factor
    notebook = options.notebook
    parallel = options.parallel
    reference_year = options.reference_year
    resampling_frequency = options.resampling_frequency

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

    ds = (
        prp.load_ensemble(basin_files, parallel=parallel)
        .sortby("basin")
        .dropna(dim="exp_id")
    )
    ds = prp.standardize_variable_names(ds, ragis_config["PISM"])
    ds[ragis_config["Cumulative Variables"]["discharge_cumulative"]] = ds[
        ragis_config["Flux Variables"]["discharge_flux"]
    ].cumsum() / len(ds.time)
    ds[ragis_config["Cumulative Variables"]["smb_cumulative"]] = ds[
        ragis_config["Flux Variables"]["smb_flux"]
    ].cumsum() / len(ds.time)
    ds = prp.normalize_cumulative_variables(
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
    upper_bound = -150

    outlier_filter = (
        ds.sel(basin="GIS", ensemble_id="RAGIS")
        .utils.drop_nonnumeric_vars()["ice_discharge"]
        .sel(time=slice(str(filter_start_year), str(filter_end_year)))
    )
    # filter_days_in_month = filter_ds.time.dt.days_in_month
    # filter_wgts = filter_days_in_month.groupby(
    #     "time.year"
    # ) / filter_days_in_month.groupby("time.year").sum(dim="time")
    # outlier_filter = (filter_ds * filter_wgts).resample({"time": "YS"}).sum()
    # Identify exp_id values that fall within the 99% credibility interval
    mask = (outlier_filter >= lower_bound) & (outlier_filter <= upper_bound)
    mask = ~(~mask).any(dim="time")
    filtered_ds = outlier_filter.sel(exp_id=mask)
    filtered_exp_ids = filtered_ds.exp_id.values
    n_members = len(ds.exp_id)
    n_members_filtered = len(filtered_exp_ids)
    print(f"Ensemble size: {n_members}, outlier-filtered size: {n_members_filtered}")
    ds = ds.sel(exp_id=filtered_exp_ids)

    fig, ax = plt.subplots(1, 1)
    ds.sel(time=slice(str(filter_start_year), str(filter_end_year))).sel(
        basin="GIS", ensemble_id="RAGIS"
    ).ice_discharge.plot(hue="exp_id", add_legend=False, ax=ax, lw=0.5)
    fig.savefig("ice_discharge_filtered.pdf")

    pism_config = ds.sel(basin="GIS").sel(pism_config_axis=params).pism_config

    prior_df = pism_config.to_dataframe()

    prior = pd.concat(
        [
            prp.transpose_dataframe(df, exp_id)
            for exp_id, df in prior_df.reset_index().groupby(by="exp_id")
        ]
    ).reset_index(drop=True)

    # Apply the conversion function to each column
    prior = prior.apply(prp.convert_column_to_float)
    for col in ["surface.given.file", "ocean.th.file", "calving.rate_scaling.file"]:
        prior[col] = prior[col].apply(prp.simplify)
    prior["surface.given.file"] = prior["surface.given.file"].apply(
        prp.simplify_climate
    )
    prior["ocean.th.file"] = prior["ocean.th.file"].apply(prp.simplify_ocean)
    prior["calving.rate_scaling.file"] = prior["calving.rate_scaling.file"].apply(
        prp.simplify_calving
    )

    prior["Ensemble"] = "Prior"

    observed = xr.open_dataset(options.obs_url).sel(time=slice("1986", "2021"))
    observed = observed.sortby("basin")
    observed = prp.normalize_cumulative_variables(
        observed,
        list(cumulative_vars.values()) + list(cumulative_uncertainty_vars.values()),
        reference_year,
    )

    observed_days_in_month = observed["time"].dt.days_in_month
    observed_days_in_month = observed.time.dt.days_in_month

    observed_wgts = 1 / (observed_days_in_month)
    observed_resampled = (
        (observed * observed_wgts)
        .resample(time=resampling_frequency)
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
    #     simulated_days_in_month.groupby(f"""time.{freq_dict[resampling_frequency]}""")
    #     / simulated_days_in_month.groupby(
    #         f"""time.{freq_dict[resampling_frequency]}"""
    #     ).sum()
    # )
    # simulated_resampled = (
    #     (
    #         simulated.drop_vars(["pism_config", "run_stats"], errors="ignore")
    #         * simulated_wgts
    #     )
    #     .resample(time=resampling_frequency)
    #     .sum(dim="time")
    # )
    simulated = ds
    simulated_resampled = (
        simulated.drop_vars(["pism_config", "run_stats"], errors="ignore")
        .resample(time=resampling_frequency)
        .sum(dim="time")
        .rolling(time=13)
        .mean()
    )
    simulated_resampled["pism_config"] = simulated["pism_config"]

    simulated_filtered_all = {}
    filtered_all = {}
    prior_posterior_list = []
    for obs_mean_var, obs_std_var, sim_var in zip(
        list(flux_vars.values())[:2],
        list(flux_uncertainty_vars.values())[:2],
        list(flux_vars.values())[:2],
    ):
        print(f"Particle filtering using {obs_mean_var}")
        filtered_ids = particle_filter(
            simulated=simulated_resampled.sel(
                time=slice(str(filter_start_year), str(filter_end_year))
            ).load(),
            observed=observed_resampled.sel(
                time=slice(str(filter_start_year), str(filter_end_year))
            ).load(),
            fudge_factor=fudge_factor,
            n_samples=len(simulated.exp_id),
            obs_mean_var=obs_mean_var,
            obs_std_var=obs_std_var,
            sim_var=sim_var,
        )
        filtered_ids["basin"] = filtered_ids["basin"].astype("<U3")

        prior_config = ds.sel(pism_config_axis=params).pism_config
        dims = [dim for dim in prior_config.dims if not dim in ["pism_config_axis"]]
        prior_df = prior_config.to_dataframe().reset_index()
        prior = prior_df.pivot(
            index=dims, columns="pism_config_axis", values="pism_config"
        )
        prior.reset_index(inplace=True)
        prior["Ensemble"] = "Prior"

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
        simulated_filtered_all[obs_mean_var] = simulated_filtered

        sim_prior = simulated_resampled.load()
        sim_prior["Ensemble"] = "Prior"
        sim_posterior = simulated_filtered.load()
        sim_posterior["Ensemble"] = "Posterior"

        with prp.tqdm_joblib(
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
                    obs_alpha=obs_alpha,
                    sim_alpha=sim_alpha,
                )
                for basin in observed_resampled.basin
            )

    prior_posterior = pd.concat(prior_posterior_list).reset_index()
    prior_posterior = prior_posterior.apply(prp.convert_column_to_float)
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

    # for filtering_var, simulated_filtered in simulated_filtered_all.items():
    #     for basin in simulated_filtered.basin.values:
    #         obs = observed.sel(basin=basin).rolling(time=390).mean()
    #         sim_prior = (
    #             simulated.sel(basin=basin, ensemble_id="RAGIS")
    #             .drop_vars(["pism_config", "run_stats"], errors="ignore")
    #             .rolling(time=13)
    #             .mean()
    #         )
    #         sim_prior["Ensemble"] = "Prior"
    #         sim_posterior = (
    #             simulated_filtered.sel(basin=basin, ensemble_id="RAGIS")
    #             .drop_vars(["pism_config", "run_stats"], errors="ignore")
    #             .rolling(time=13)
    #             .mean()
    #         )
    #         sim_posterior["Ensemble"] = "Posterior"

    #         posterior_df = filtered_all[filtering_var]

    for (basin, filtering_var), df in prior_posterior.rename(
        columns=params_short_dict
    ).groupby(by=["basin", "filtered_by"]):
        n_params = len(params_short_dict)
        plt.rcParams["font.size"] = 4
        fig, axs = plt.subplots(
            5,
            3,
            sharey=True,
            figsize=[6.2, 5.2],
        )
        fig.subplots_adjust(hspace=1.0, wspace=0.1)
        for k, v in enumerate(params_short_dict.values()):
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
                    legend=False,
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

    ensemble_df = prior_posterior.drop(columns=["Ensemble", "exp_id"], errors="ignore")
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

    to_analyze = ds.sel(time=slice("1980-01-01", "2020-01-01"))
    print("Calculating Sensitivity Indices")
    print("===============================")
    cluster = LocalCluster(n_workers=options.n_jobs, threads_per_worker=1)
    client = Client(cluster, asynchronous=True)
    # client = Client()
    print(f"Open client in browser: {client.dashboard_link}")
    dim = "time"
    all_delta_indices_list = []
    for basin in to_analyze.basin.values:
        for filter_var in [
            "mass_balance",
            "ice_discharge",
        ]:
            print(
                f"  ...sensitivity indices for basin {basin} filtered by {filter_var} ",
            )
            start = time.time()

            responses = to_analyze.sel(basin=basin, ensemble_id="RAGIS")[filter_var]
            responses_scattered = client.scatter(
                [responses.isel(time=k).to_numpy() for k in range(len(responses[dim]))]
            )

            futures = client.map(
                delta_analysis,
                responses_scattered,
                problem=problem,
                ensemble_df=ensemble_df,
            )
            progress(futures, notebook=notebook)
            result = client.gather(futures)

            end = time.time()
            time_elapsed = end - start
            print(f"  ...took {time_elapsed:.0f}s")

            delta_indices = xr.concat([r.expand_dims(dim) for r in result], dim=dim)
            delta_indices[dim] = responses[dim]
            delta_indices = delta_indices.expand_dims("basin", axis=1)
            delta_indices["basin"] = [basin]
            delta_indices = delta_indices.expand_dims("filtered_by", axis=2)
            delta_indices["filtered_by"] = [filter_var]
            all_delta_indices_list.append(delta_indices)

    all_delta_indices: xr.Dataset = xr.merge(all_delta_indices_list)
    client.close()

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
