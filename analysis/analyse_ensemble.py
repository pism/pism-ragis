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
from pathlib import Path
from typing import List
from importlib.resources import files
import toml
import geopandas as gp
import numpy as np
import pandas as pd
import pylab as plt
import seaborn as sns
import xarray as xr

from pism_ragis.analysis import resample_ensemble_by_data
from pism_ragis.observations import load_imbie, load_imbie_2021, load_mouginot

kg2cmsle = 1 / 1e12 * 1.0 / 362.5 / 10.0
gt2cmsle = 1 / 362.5 / 10.0


if __name__ == "__main__":
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
        "--basin_url",
        help="""Basin shapefile.""",
        type=str,
        default="~/base/pism-ragis/data/mouginot/Greenland_Basins_PS_v1.4.2_w_shelves.gpkg",
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
    ragis_config_file = Path(str(files("pism_ragis.data").joinpath("ragis_config.toml")))
    all_params_dict = toml.load(ragis_config_file)["Parameters"]
    
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
    result_dir.mkdir(parents=True, exist_ok=True)

    fig_dir = result_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    reference_year = 1992

    mass_flux_varname = "mass_balance"
    mass_flux_uncertainty_varname = "mass_balance_uncertainty"
    mass_cumulative_varname = "cumulative_mass_balance"
    mass_cumulative_uncertainty_varname = "cumulative_mass_balance_uncertainty"

    discharge_flux_varname = "ice_discharge"
    discharge_flux_uncertainty_varname = "ice_discharge_uncertainty"
    smb_flux_varname = "surface_mass_balance"
    smb_flux_uncertainty_varname = "surface_mass_balance_uncertainty"

    basal_flux_varname = "tendency_of_ice_mass_due_to_basal_mass_flux"
    basal_grounded_flux_varname = "tendency_of_ice_mass_due_to_basal_mass_flux_grounded"
    basal_floating_flux_varname = "tendency_of_ice_mass_due_to_basal_mass_flux_floating"

    sim_mass_cumulative_varname = "ice_mass"
    sim_mass_flux_varname = "tendency_of_ice_mass"
    sim_smb_flux_varname = "tendency_of_ice_mass_due_to_surface_mass_flux"
    sim_discharge_flux_varname = "grounding_line_flux"

    # Load basins, merge all ICE_CAP geometries
    basin_url = Path(options.basin_url)
    basins = gp.read_file(basin_url).to_crs(crs)

    # Load observations
    if options.imbie_url is not None:
        imbie = load_imbie(url=Path(options.imbie_url))
    else:
        imbie = load_imbie()
    imbie_2021 = load_imbie_2021()

    mou = load_mouginot(url=Path(options.mouginot_url), norm_year=reference_year)
    mou_gis = mou.sel(basin="GIS")

    comp = {"zlib": True, "complevel": 2}

    basins_files = result_dir.glob("basin*.nc")
    basins_sums = xr.open_mfdataset(basins_files, parallel=True, chunks="auto")
    basins_sums = basins_sums.sel(ensemble_id="RAGIS").sel(
        time=slice("1980-01-01", "2020-01-01")
    )
    basins_sums[sim_mass_cumulative_varname] -= basins_sums.sel(
        time=f"{reference_year}-01-01", method="nearest"
    )[sim_mass_cumulative_varname]
    basins_sums[sim_discharge_flux_varname] = (
        basins_sums["ice_mass_transport_across_grounding_line"]
        + basins_sums["tendency_of_ice_mass_due_to_basal_mass_flux_grounded"]
    )
    basins_sums.load()

    observed = imbie_2021
    simulated = basins_sums.sel(basin="GIS")
    resampled_ensemble = resample_ensemble_by_data(observed, simulated, fudge_factor=20)

    basins_sums_resampled = []
    for k, exp_id in enumerate(resampled_ensemble):
        b = basins_sums.sel(exp_id=exp_id)
        b["exp_id"] = k
        basins_sums_resampled.append(b)
    basins_sums_resampled = xr.concat(basins_sums_resampled, dim="exp_id")
    
    config = basins_sums.sel(basin="CE").sel(config_axis=params).config
    uq_df = config.to_dataframe()

    def transpose_dataframe(df, exp_id):
        """
        Transpose dataframe.
        """
        param_names = df["config_axis"]
        df = df[["config"]].T
        df.columns = param_names
        df["exp_id"] = exp_id
        return df

    ragis = pd.concat(
        [transpose_dataframe(df, exp_id) for exp_id, df in uq_df.reset_index().groupby(by="exp_id")]
    ).reset_index(drop=True)
    
    def simplify(my_str: str) -> str:
        return Path(my_str).name
    
    # Function to convert column to float if possible
    def convert_column_to_float(column):
        try:
            return pd.to_numeric(column, errors='raise')
        except ValueError:
            return column

    # Apply the conversion function to each column
    ragis = ragis.apply(convert_column_to_float)
    for col in ["surface.given.file", "ocean.th.file"]:
        ragis[col] = ragis[col].apply(simplify)
    ragis["Ensemble"] = "Prior"
    
    resampled_df = []
    for k in resampled_ensemble:
        resampled_df.append(ragis[ragis["exp_id"] == k])
    ragis_resampled = pd.concat(resampled_df)
    ragis_resampled["Ensemble"] = "Posterior"
    
    posterior_df = pd.concat([ragis, ragis_resampled]).rename(columns=params_short_dict)

    n_params = len(params_short_dict)
    fig, axs = plt.subplots(
        5,
        3,
        figsize=[6.2, 11],
    )
    fig.subplots_adjust(hspace=1.0, wspace=1.0)

    for k, v in enumerate(params_short_dict.values()):
        try:
            sns.histplot(
                data=posterior_df,
                x=v,
                hue="Ensemble",
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
    fig.savefig("hist.pdf")
        
    obs_cmap = sns.color_palette("crest", n_colors=4)
    obs_cmap = ["0.4", "0.0", "0.6", "0.0"]
    sim_cmap = sns.color_palette("flare", n_colors=4)
    plt.rcParams["font.size"] = 6

    fig, axs = plt.subplots(
        3, 1, sharex=True, figsize=(6.2, 4.2), height_ratios=[2, 1, 1]
    )

    sim_labels: List = []
    exp_labels = ["Forced Retreat", "Control"]
    sim_lines: List = []
    sim_alpha = 0.5

    gris = basins_sums.sel(basin="GIS").rolling(time=13).mean()
    gris_resampled = basins_sums_resampled.sel(basin="GIS").rolling(time=13).mean()

    mou_ci = axs[0].fill_between(
        mou_gis["time"],
        mou_gis[mass_cumulative_varname] - mou_gis[mass_cumulative_uncertainty_varname],
        mou_gis[mass_cumulative_varname] + mou_gis[mass_cumulative_uncertainty_varname],
        color=obs_cmap[0],
        alpha=0.5,
        lw=0,
        label="Mouginot et al (2019)",
    )
    imbie_ci = axs[0].fill_between(
        imbie_2021["time"],
        imbie_2021[mass_cumulative_varname]
        - imbie_2021[mass_cumulative_uncertainty_varname],
        imbie_2021[mass_cumulative_varname]
        + imbie_2021[mass_cumulative_uncertainty_varname],
        color=obs_cmap[2],
        alpha=0.5,
        lw=0,
        label="IMBIE 2021",
    )

    axs[1].fill_between(
        mou_gis["time"],
        mou_gis[discharge_flux_varname] - mou_gis[discharge_flux_uncertainty_varname],
        mou_gis[discharge_flux_varname] + mou_gis[discharge_flux_uncertainty_varname],
        color=obs_cmap[2],
        alpha=0.5,
        lw=0,
    )
    axs[1].fill_between(
        imbie["time"],
        imbie[discharge_flux_varname] - imbie[discharge_flux_uncertainty_varname],
        imbie[discharge_flux_varname] + imbie[discharge_flux_uncertainty_varname],
        color=obs_cmap[0],
        alpha=0.5,
        lw=0,
    )
    axs[1].fill_between(
        mou_gis["time"],
        mou_gis[discharge_flux_varname] - mou_gis[discharge_flux_uncertainty_varname],
        mou_gis[discharge_flux_varname] + mou_gis[discharge_flux_uncertainty_varname],
        color=obs_cmap[0],
        alpha=0.5,
        lw=0,
    )
    axs[2].fill_between(
        imbie["time"],
        imbie[smb_flux_varname] - imbie[smb_flux_uncertainty_varname],
        imbie[smb_flux_varname] + imbie[smb_flux_uncertainty_varname],
        color=obs_cmap[0],
        alpha=0.5,
        lw=0,
    )
    axs[2].fill_between(
        mou_gis["time"],
        mou_gis[smb_flux_varname] - mou_gis[smb_flux_uncertainty_varname],
        mou_gis[smb_flux_varname] + mou_gis[smb_flux_uncertainty_varname],
        color=obs_cmap[0],
        alpha=0.5,
        lw=0,
    )

    quantiles = {}
    for q in [0.16, 0.5, 0.84]:
        quantiles[q] = gris.drop_vars("config").quantile(q, dim="exp_id", skipna=False)

    for k, m_var in enumerate(
        [sim_mass_cumulative_varname, sim_discharge_flux_varname, sim_smb_flux_varname]
    ):
        sim_ci = axs[k].fill_between(
            quantiles[0.5].time,
            quantiles[0.16][m_var],
            quantiles[0.84][m_var],
            alpha=0.3,
            color=sim_cmap[1],
            lw=0,
        )
    quantiles = {}
    for q in [0.16, 0.5, 0.84]:
        quantiles[q] = gris_resampled.drop_vars("config").quantile(q, dim="exp_id", skipna=False)

    for k, m_var in enumerate(
        [sim_mass_cumulative_varname, sim_discharge_flux_varname, sim_smb_flux_varname]
    ):
        sim_ci = axs[k].fill_between(
            quantiles[0.5].time,
            quantiles[0.16][m_var],
            quantiles[0.84][m_var],
            alpha=0.3,
            color=sim_cmap[3],
            lw=0,
        )

    legend_obs = axs[0].legend(
        handles=[mou_ci, imbie_ci], loc="lower left", title="Observed"
    )
    legend_obs.get_frame().set_linewidth(0.0)
    legend_obs.get_frame().set_alpha(0.0)

    legend_sim = axs[0].legend(
        handles=sim_lines, loc="upper left", title="Simulated (13-month rolling mean)"
    )
    legend_sim.get_frame().set_linewidth(0.0)
    legend_sim.get_frame().set_alpha(0.0)

    axs[0].add_artist(legend_obs)
    axs[0].add_artist(legend_sim)

    # axs[0].set_ylim(-6000, 1500)
    axs[1].set_ylim(-750, 0)
    axs[2].set_ylim(0, 750)
    axs[0].xaxis.set_tick_params(labelbottom=False)

    axs[0].set_ylabel(f"Cumulative mass\nloss since {reference_year} (Gt)")
    axs[0].set_xlabel("")
    axs[0].set_title("basin = GIS")
    axs[1].set_title("")
    axs[1].set_ylabel("Grounding Line\nFlux (Gt/yr)")
    axs[2].set_ylabel("Climatic Mass\nBalance (Gt/yr)")
    axs[-1].set_xlim(np.datetime64("1980-01-01"), np.datetime64("2021-01-01"))
    fig.tight_layout()
    fig.savefig("GIS_mass_accounting.pdf")

    # basins_sums = basins_sums.rolling(time=13).mean()
    # basins_sums = basins_sums.rename(
    #     {
    #         sim_mass_cumulative_varname: mass_cumulative_varname,
    #         sim_mass_flux_varname: mass_flux_varname,
    #         sim_discharge_flux_varname: discharge_flux_varname,
    #         sim_smb_flux_varname: smb_flux_varname,
    #     }
    # )
    # basins_sums[mass_cumulative_varname] -= basins_sums.sel(
    #     time="1980-01-01", method="nearest"
    # )[mass_cumulative_varname]
    # basins_sums[discharge_cumulative_varname] = basins_sums[
    #     discharge_flux_varname
    # ].cumsum(dim="time")
    # basins_sums[discharge_cumulative_varname] -= basins_sums.sel(
    #     time="1980-01-01", method="nearest"
    # )[discharge_cumulative_varname]
    # basins_sums[smb_cumulative_varname] = basins_sums[smb_flux_varname].cumsum(
    #     dim="time"
    # )
    # basins_sums[smb_cumulative_varname] -= basins_sums.sel(
    #     time="1980-01-01", method="nearest"
    # )[smb_cumulative_varname]

    sim_colors = colorblind_colors
    obs = mou_gis
    obs_color = "#216778"
    obs_alpha = 1.0
    sim_alpha = 0.1

    plt.rc("font", size=6)
