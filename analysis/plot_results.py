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

import geopandas as gp
import numpy as np
import pandas as pd
import pylab as plt
import xarray as xr

from pism_ragis.observations import load_imbie, load_mouginot

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
        default="data/basins/Greenland_Basins_PS_v1.4.2.shp",
    )
    parser.add_argument(
        "--imbie_url",
        help="""Path to IMBIE excel file.""",
        type=str,
        default="/mnt/storstrommen/data/imbie/imbie_dataset_greenland_dynamics-2020_02_28.xlsx",
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

    result_dir = Path(options.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    fig_dir = result_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    mass_cumulative_varname = "Cumulative ice sheet mass change (Gt)"
    mass_cumulative_uncertainty_varname = (
        "Cumulative ice sheet mass change uncertainty (Gt)"
    )
    mass_flux_varname = "Rate of ice sheet mass change (Gt/yr)"
    mass_flux_uncertainty_varname = "Rate of ice sheet mass change uncertainty (Gt/yr)"

    discharge_cumulative_varname = "Cumulative ice discharge anomaly (Gt)"
    discharge_cumulative_uncertainty_varname = (
        "Cumulative ice discharge anomaly uncertainty (Gt)"
    )
    discharge_flux_varname = "Rate of ice discharge (Gt/yr)"
    discharge_flux_uncertainty_varname = "Rate of ice discharge uncertainty (Gt/yr)"

    smb_cumulative_varname = "Cumulative surface mass balance anomaly (Gt)"
    smb_cumulative_uncertainty_varname = (
        "Cumulative surface mass balance anomaly uncertainty (Gt)"
    )
    smb_flux_varname = "Rate of surface mass balance (Gt/yr)"
    smb_flux_uncertainty_varname = "Rate of surface mass balance uncertainty (Gt/yr)"

    basal_flux_varname = "tendency_of_ice_mass_due_to_basal_mass_flux"

    sim_mass_cumulative_varname = "ice_mass"
    sim_mass_flux_varname = "tendency_of_ice_mass"
    sim_smb_flux_varname = "tendency_of_ice_mass_due_to_surface_mass_flux"
    sim_discharge_flux_varname = "ice_mass_transport_across_grounding_line"

    # Load basins, merge all ICE_CAP geometries
    basin_url = Path(options.basin_url)
    basins = gp.read_file(basin_url).to_crs(crs)
    # if "SUBREGION1" in basins:
    #     ice_sheet = basins[basins["SUBREGION1"] != "ICE_CAP"]
    #     ice_caps = basins[basins["SUBREGION1"] == "ICE_CAP"].unary_union
    #     ice_caps = gp.GeoDataFrame(pd.DataFrame(data=["ICE_CAP"], columns=["SUBREGION1"]), geometry=[ice_caps], crs=basins.crs)
    #     basins = pd.concat([ice_sheet, ice_caps]).reset_index(drop=True)

    # Load observations
    imbie = load_imbie(url=Path(options.imbie_url))
    mou = load_mouginot(url=Path(options.mouginot_url), norm_year=1980)
    mou[discharge_flux_varname] = -mou[discharge_flux_varname]
    mou_gis = mou[mou["Basin"] == "GIS"]

    imbie_mean = imbie[imbie.Date.between("1992-1-1", "2012-1-1")][
        [
            mass_flux_uncertainty_varname,
            smb_flux_uncertainty_varname,
            discharge_flux_uncertainty_varname,
        ]
    ].mean()
    mou_mean = mou[mou.Date.between("1992-1-1", "2012-1-1")][
        [
            mass_flux_uncertainty_varname,
            smb_flux_uncertainty_varname,
            discharge_flux_uncertainty_varname,
        ]
    ].mean()
    u_ratio = imbie_mean / mou_mean / 2
    sigma_adjusted = np.maximum(u_ratio, 1)

    sigma_mass = sigma_adjusted[mass_flux_uncertainty_varname]
    sigma_smb = sigma_adjusted[smb_flux_uncertainty_varname]
    sigma_discharge = sigma_adjusted[discharge_flux_uncertainty_varname]

    comp = {"zlib": True, "complevel": 2}

    basins_files = result_dir.glob("basin*.nc")

    basins_sums = xr.open_mfdataset(basins_files, chunks="auto", parallel=True)
    basins_sums = basins_sums.rename(
        {
            sim_mass_cumulative_varname: mass_cumulative_varname,
            sim_mass_flux_varname: mass_flux_varname,
            sim_discharge_flux_varname: discharge_flux_varname,
            sim_smb_flux_varname: smb_flux_varname,
        }
    )
    basins_sums[mass_cumulative_varname] -= basins_sums.sel(
        time="1980-01-01", method="nearest"
    )[mass_cumulative_varname]
    basins_sums[discharge_cumulative_varname] = basins_sums[
        discharge_flux_varname
    ].cumsum(dim="time")
    basins_sums[discharge_cumulative_varname] -= basins_sums.sel(
        time="1980-01-01", method="nearest"
    )[discharge_cumulative_varname]
    basins_sums[smb_cumulative_varname] = basins_sums[smb_flux_varname].cumsum(
        dim="time"
    )
    basins_sums[smb_cumulative_varname] -= basins_sums.sel(
        time="1980-01-01", method="nearest"
    )[smb_cumulative_varname]
    basins_sums.load()

    sim_colors = colorblind_colors
    obs = mou_gis
    obs_color = "#216778"
    obs_alpha = 1.0
    sim_alpha = 0.1

    plt.rc("font", size=6)

    fig, axs = plt.subplots(nrows=4, ncols=2, sharex="col", figsize=(6.2, 4.2))
    obs_color = "0.5"
    for k, basin_name in enumerate(["GIS", "CE", "CW", "NE", "NO", "NW", "SE", "SW"]):
        obs = mou[mou["Basin"] == basin_name]
        ax = axs.ravel()[k]
        sim_cis = []
        for m, (ens_id, da) in enumerate(
            basins_sums.sel(basin=basin_name).groupby("ensemble_id", squeeze=False)
        ):
            da = da.squeeze()
            g_low = da.quantile(0.16, dim="exp_id")
            g_med = da.quantile(0.50, dim="exp_id")
            g_high = da.quantile(0.84, dim="exp_id")
            sim_ci = ax.fill_between(
                g_med.time,
                g_low[mass_cumulative_varname],
                g_high[mass_cumulative_varname],
                alpha=sim_alpha,
                color=sim_colors[m],
                lw=0,
                label=ens_id,
            )
            g_sim = ax.plot(
                g_med.time,
                g_med[mass_cumulative_varname],
                lw=0.75,
                color=sim_colors[m],
                label="Median",
            )
            # flux_ax.fill_between(g_med.time, g_low.rolling(time=13).mean()[discharge_flux_varname], g_high.rolling(time=13).mean()[discharge_flux_varname],
            #                     alpha=sim_alpha, color=sim_colors[m], lw=0, label=ens_id)
            # flux_ax.plot(g_med.time, g_med.rolling(time=13).mean()[discharge_flux_varname], lw=0.5, ls="dotted", color=sim_colors[m], label="Median")
            sim_cis.append(g_sim[0])

        for m_var, u_var in zip(
            [mass_cumulative_varname], [mass_cumulative_uncertainty_varname]
        ):
            obs_ci = ax.fill_between(
                obs["Date"],
                (obs[m_var] + obs[u_var]),
                (obs[m_var] - obs[u_var]),
                ls="solid",
                color=obs_color,
                lw=0,
                alpha=obs_alpha,
                label="1-$\sigma$",
            )
            obs_ci_2 = ax.fill_between(
                obs["Date"],
                (obs[m_var] + 2 * obs[u_var]),
                (obs[m_var] - 2 * obs[u_var]),
                ls="solid",
                color=obs_color,
                lw=0,
                alpha=obs_alpha / 2,
                label="2-$\sigma$",
            )
        # for m_var, u_var in zip([discharge_flux_varname], [discharge_flux_uncertainty_varname]):
        #     flux_ax.fill_between(obs["Date"],
        #                         (obs[m_var] + obs[u_var]),
        #                         (obs[m_var] - obs[u_var]),
        #                         ls="solid", color=obs_color, lw=0, alpha=obs_alpha, label="1-$\sigma$")
        #     flux_ax.fill_between(obs["Date"],
        #                         (obs[m_var] + 2  * obs[u_var]),
        #                         (obs[m_var] - 2  * obs[u_var]),
        #                         ls="solid", color=obs_color, lw=0, alpha=obs_alpha / 2, label="2-$\sigma$")

        # legend_obs = ax.legend(handles=[obs_ci, obs_ci_2], loc="lower left",
        #                                title="Observed\n(Mouginot 2019)")
        # legend_obs.get_frame().set_linewidth(0.0)
        # legend_obs.get_frame().set_alpha(0.0)

        # legend_sim = ax.legend(handles=sim_cis, loc="upper left",
        #                                title="Simulated")
        # legend_sim.get_frame().set_linewidth(0.0)
        # legend_sim.get_frame().set_alpha(0.0)

        # ax.add_artist(legend_obs)
        # ax.add_artist(legend_sim)

        ax.axhline(0, ls="dotted", color="k", lw=0.5)
        ax.set_xlim(pd.to_datetime("1980-1-1"), pd.to_datetime("2000-1-1"))
        # ax.set_ylim(-2500, 2000)
        ax.set_ylabel("Cumulative\nMass Change (Gt)")
        ax.set_title(basin_name)
    fig.tight_layout()
    fig.savefig(fig_dir / "basin_ice_mass_scalar_1980-2000.pdf")
    fig.savefig(fig_dir / "basin_ice_mass_scalar_1980-2000.png", dpi=600)

    plt.style.use("tableau-colorblind10")
    lw = 1.25
    fig, axs = plt.subplots(nrows=4, ncols=2, sharex="col", figsize=(10.2, 10.2))
    obs_color = "0.5"
    for k, basin_name in enumerate(["GIS", "CE", "CW", "NE", "NO", "NW", "SE", "SW"]):
        obs = mou[mou["Basin"] == basin_name]
        ax = axs.ravel()[k]
        sim_cis = []
        for m, (ens_id, da) in enumerate(
            basins_sums.sel(basin=basin_name).groupby("ensemble_id", squeeze=False)
        ):
            da = da.squeeze()
            g_low = da.quantile(0.16, dim="exp_id").rolling({"time": 13}).mean()
            g_med = da.quantile(0.50, dim="exp_id").rolling({"time": 13}).mean()
            g_high = da.quantile(0.84, dim="exp_id").rolling({"time": 13}).mean()
            d_sim = ax.plot(
                g_med.time,
                g_med[discharge_flux_varname],
                lw=lw,
                color=sim_colors[m],
                label="Grounding Line Flux",
            )
            sim_cis.append(d_sim[0])
            # df_sim = ax.plot(g_med.time, g_med["tendency_of_ice_mass_due_to_discharge"], lw=lw, label="Discharge Flux")
            # sim_cis.append(df_sim[0])
            # dff_sim = ax.plot(g_med.time, g_med[smb_flux_varname]+g_med[discharge_flux_varname], lw=lw, label="M=SMB+D")
            # sim_cis.append(dff_sim[0])
            bg_sim = ax.plot(
                g_med.time,
                g_med[f"{basal_flux_varname}_grounded"],
                lw=lw,
                label="Basal Flux Grounded",
            )
            sim_cis.append(bg_sim[0])
            bf_sim = ax.plot(
                g_med.time,
                g_med[f"{basal_flux_varname}_floating"],
                lw=lw,
                label="Basal Flux Floating",
            )
            sim_cis.append(bf_sim[0])
            # m_sim = ax.plot(g_med.time, g_med[mass_flux_varname]-g_med[smb_flux_varname]-g_med[discharge_flux_varname]-g_med[basal_flux_varname], lw=lw,  label="M-SMB-D-BMB")
            # sim_cis.append(m_sim[0])

        for m_var, u_var in zip(
            [discharge_flux_varname], [discharge_flux_uncertainty_varname]
        ):
            obs_ci = ax.fill_between(
                obs["Date"],
                (obs[m_var] + obs[u_var]),
                (obs[m_var] - obs[u_var]),
                ls="solid",
                color=obs_color,
                lw=0,
                alpha=obs_alpha,
                label="1-$\sigma$",
            )
            obs_ci_2 = ax.fill_between(
                obs["Date"],
                (obs[m_var] + 2 * obs[u_var]),
                (obs[m_var] - 2 * obs[u_var]),
                ls="solid",
                color=obs_color,
                lw=0,
                alpha=obs_alpha / 2,
                label="2-$\sigma$",
            )

        ax.axhline(0, ls="dotted", color="k", lw=0.5)
        ax.set_xlim(pd.to_datetime("1980-1-1"), pd.to_datetime("2000-1-1"))
        # ax.set_ylim(-2500, 2000)
        ax.set_ylabel("Mass Flux (Gt/yr)")
        ax.set_title(basin_name)
    legend_obs = ax.legend(
        handles=[obs_ci, obs_ci_2], loc="lower left", title="Observed\n(Mouginot 2019)"
    )
    legend_obs.get_frame().set_linewidth(0.0)
    legend_obs.get_frame().set_alpha(0.0)

    legend_sim = ax.legend(handles=sim_cis, loc="upper right", title="Simulated")
    legend_sim.get_frame().set_linewidth(0.0)
    legend_sim.get_frame().set_alpha(0.0)

    ax.add_artist(legend_obs)
    ax.add_artist(legend_sim)
    fig.tight_layout()
    fig.savefig(fig_dir / "basin_ice_fluxes_scalar_1980-2000.pdf")
    fig.savefig(fig_dir / "basin_ice_fluxes_scalar_1980-2000.png", dpi=600)

    obs = mou_gis
    fig, ax = plt.subplots(nrows=1, ncols=1, sharex="col", figsize=(7, 4.2))
    sim_cis = []
    obs_color = "0.5"
    gris = basins_sums.sel(basin="GIS")
    for k, (ens_id, da) in enumerate(gris.groupby("ensemble_id")):
        da = da.load()
        g_low = da.quantile(0.16, dim="exp_id")
        g_med = da.quantile(0.50, dim="exp_id")
        g_high = da.quantile(0.84, dim="exp_id")

        sim_ci = ax.fill_between(
            g_med.time,
            g_low[mass_cumulative_varname],
            g_high[mass_cumulative_varname],
            alpha=sim_alpha,
            color=sim_colors[k],
            lw=0,
            label=f"{ens_id} 16-84% c.i.",
        )
        sim_cis.append(sim_ci)
        g_sim = ax.plot(
            g_med.time,
            g_med[mass_cumulative_varname],
            lw=1.0,
            color=sim_colors[k],
            label=f"{ens_id} Median",
        )
        sim_cis.append(g_sim[0])

    obs_ci = ax.fill_between(
        obs["Date"],
        (obs[mass_cumulative_varname] + obs[mass_cumulative_uncertainty_varname]),
        (obs[mass_cumulative_varname] - obs[mass_cumulative_uncertainty_varname]),
        ls="solid",
        color=obs_color,
        lw=0,
        alpha=obs_alpha,
        label="1-$\sigma$",
    )

    legend_obs = ax.legend(
        handles=[obs_ci], loc="lower left", title="Observed\n(Mouginot 2019)"
    )
    legend_obs.get_frame().set_linewidth(0.0)
    legend_obs.get_frame().set_alpha(0.0)

    legend_sim = ax.legend(handles=sim_cis, loc="upper left", title="Simulated")
    legend_sim.get_frame().set_linewidth(0.0)
    legend_sim.get_frame().set_alpha(0.0)

    ax.add_artist(legend_obs)
    ax.add_artist(legend_sim)
    # scalar_mass.sel(exp_id="BAYES-MEDIAN").plot.line(x="time", ax=ax)
    ax.axhline(0, ls="dotted", color="k", lw=0.5)
    ax.set_xlim(pd.to_datetime("1980-1-1"), pd.to_datetime("2000-1-1"))
    ax.set_ylim(-2500, 2000)
    ax.set_ylabel("Cumulative Mass Change\(Gt)")
    fig.tight_layout()
    fig.savefig(fig_dir / "ice_mass_scalar_1980-2000.pdf")
    fig.savefig(fig_dir / "ice_mass_scalar_1980-2000.png", dpi=600)

    fig, axs = plt.subplots(
        nrows=3, ncols=1, sharex="col", sharey="row", figsize=(7, 5.2)
    )
    sim_cis = []
    obs_color = "0.5"
    for k, (ens_id, da) in enumerate(gris.groupby("ensemble_id")):
        da = da.load()
        g_low = da.quantile(0.16, dim="exp_id").rolling({"time": 13}).mean()
        g_med = da.quantile(0.50, dim="exp_id").rolling({"time": 13}).mean()
        g_high = da.quantile(0.84, dim="exp_id").rolling({"time": 13}).mean()

        sim_ci = axs[0].fill_between(
            g_med.time,
            g_low[mass_cumulative_varname],
            g_high[mass_cumulative_varname],
            alpha=sim_alpha,
            color=sim_colors[k],
            lw=0,
            label=ens_id,
        )
        g_sim = axs[0].plot(
            g_med.time,
            g_med[mass_cumulative_varname],
            lw=1.0,
            color=sim_colors[k],
            label="Median",
        )
        sim_ci = axs[1].fill_between(
            g_med.time,
            g_low[discharge_flux_varname],
            g_high[discharge_flux_varname],
            alpha=sim_alpha,
            color=sim_colors[k],
            lw=0,
            label=ens_id,
        )
        g_sim = axs[1].plot(
            g_med.time, g_med[discharge_flux_varname], lw=1.0, color=sim_colors[k]
        )
        g_sim = axs[1].plot(
            g_med.time,
            g_med[mass_flux_varname] - g_med[smb_flux_varname],
            lw=2.0,
            color=sim_colors[k],
        )
        g_sim = axs[1].plot(
            g_med.time,
            g_med["tendency_of_ice_mass_due_to_basal_mass_flux"],
            lw=2.0,
            color="g",
        )
        sim_ci = axs[2].fill_between(
            g_med.time,
            g_low[smb_flux_varname],
            g_high[smb_flux_varname],
            alpha=sim_alpha,
            color=sim_colors[k],
            lw=0,
            label=ens_id,
        )
        g_sim = axs[2].plot(
            g_med.time, g_med[smb_flux_varname], lw=1.0, color=sim_colors[k]
        )
        sim_cis.append(sim_ci)
        sim_cis.append(g_sim[0])

    obs_ci = axs[0].fill_between(
        obs["Date"],
        (obs[mass_cumulative_varname] + obs[mass_cumulative_uncertainty_varname]),
        (obs[mass_cumulative_varname] - obs[mass_cumulative_uncertainty_varname]),
        ls="solid",
        color=obs_color,
        lw=0,
        alpha=obs_alpha,
        label="1-$\sigma$",
    )

    axs[1].fill_between(
        obs["Date"],
        (obs[discharge_flux_varname] + obs[discharge_flux_uncertainty_varname]),
        (obs[discharge_flux_varname] - obs[discharge_flux_uncertainty_varname]),
        ls="solid",
        color=obs_color,
        lw=0,
        alpha=obs_alpha,
        label="1-$\sigma$",
    )
    axs[1].fill_between(
        obs["Date"],
        (
            obs[discharge_flux_varname]
            + u_ratio[discharge_flux_uncertainty_varname]
            * obs[discharge_flux_uncertainty_varname]
        ),
        (
            obs[discharge_flux_varname]
            - u_ratio[discharge_flux_uncertainty_varname]
            * obs[discharge_flux_uncertainty_varname]
        ),
        ls="solid",
        color=obs_color,
        lw=0,
        alpha=obs_alpha / 2,
        label="adjusted $\sigma$",
    )
    axs[2].fill_between(
        obs["Date"],
        (obs[smb_flux_varname] + obs[smb_flux_uncertainty_varname]),
        (obs[smb_flux_varname] - obs[smb_flux_uncertainty_varname]),
        ls="solid",
        color=obs_color,
        lw=0,
        alpha=obs_alpha,
        label="1-$\sigma$",
    )
    axs[2].fill_between(
        obs["Date"],
        (
            obs[smb_flux_varname]
            + u_ratio[smb_flux_uncertainty_varname] * obs[smb_flux_uncertainty_varname]
        ),
        (
            obs[smb_flux_varname]
            - u_ratio[smb_flux_uncertainty_varname] * obs[smb_flux_uncertainty_varname]
        ),
        ls="solid",
        color=obs_color,
        lw=0,
        alpha=obs_alpha / 2,
        label="adjusted $\sigma$",
    )

    ax = axs[1]
    legend_obs = ax.legend(
        handles=[obs_ci], loc="upper left", title="Observed\n(Mouginot 2019)"
    )
    legend_obs.get_frame().set_linewidth(0.0)
    legend_obs.get_frame().set_alpha(0.0)

    legend_sim = ax.legend(handles=sim_cis, loc="upper right", title="Simulated")
    legend_sim.get_frame().set_linewidth(0.0)
    legend_sim.get_frame().set_alpha(0.0)

    ax.add_artist(legend_obs)
    ax.add_artist(legend_sim)
    ax.axhline(0, ls="dotted", color="k", lw=0.5)
    ax.set_xlim(pd.to_datetime("1980-1-1"), pd.to_datetime("2000-1-1"))
    axs[0].set_ylim(-2000, 2000)
    axs[0].set_ylabel("Cumulative Mass Change\n(Gt)")
    axs[1].set_ylabel("Discharge Flux (Gt/yr)")
    axs[2].set_ylabel("SMB (Gt/yr)")
    fig.tight_layout()
    fig.savefig(fig_dir / "ice_fluxes_scalar_1980-2000.pdf")
    fig.savefig(fig_dir / "ice_fluxes_scalar_1980-2000.png", dpi=600)
