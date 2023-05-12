# Copyright (C) 2023 Andy Aschwanden
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

import pandas as pd
import pylab as plt

kg2cmsle = 1 / 1e12 * 1.0 / 362.5 / 10.0
gt2cmsle = 1 / 362.5 / 10.0
sigma = 2


def load_imbie_csv(url: str = "imbie_greenland_2021_Gt.csv", proj_start=1992):
    df = pd.read_csv(url)

    df = df.rename(
        columns={
            "Mass balance (Gt/yr)": "Mass change (Gt/yr)",
            "Mass balance uncertainty (Gt/yr)": "Mass change uncertainty (Gt/yr)",
            "Cumulative mass balance (Gt)": "Mass (Gt)",
            "Cumulative mass balance uncertainty (Gt)": "Mass uncertainty (Gt)",
        }
    )
    for v in [
        "Mass (Gt)",
    ]:
        df[v] -= df[df["Year"] == proj_start][v].values

    cmSLE = 1.0 / 362.5 / 10.0
    df["SLE (cm)"] = -df["Mass (Gt)"] * cmSLE
    df["SLE uncertainty (cm)"] = -df["Mass uncertainty (Gt)"] * cmSLE
    df["SLE change uncertainty (cm/yr)"] = (
        df["Mass change uncertainty (Gt/yr)"] * gt2cmsle
    )
    return df


def load_imbie(
    url: str = "http://imbie.org/wp-content/uploads/2012/11/imbie_dataset_greenland_dynamics-2020_02_28.xlsx",
):
    """
    Loading the IMBIE Greenland data set downloaded from
    http://imbie.org/wp-content/uploads/2012/11/imbie_dataset_greenland_dynamics-2020_02_28.xlsx.

    """
    df_df = pd.read_excel(
        url,
        sheet_name="Greenland Ice Mass",
        engine="openpyxl",
    )
    df = df_df[
        [
            "Year",
            "Cumulative ice sheet mass change (Gt)",
            "Cumulative ice sheet mass change uncertainty (Gt)",
            "Cumulative surface mass balance anomaly (Gt)",
            "Cumulative surface mass balance anomaly uncertainty (Gt)",
            "Cumulative ice dynamics anomaly (Gt)",
            "Cumulative ice dynamics anomaly uncertainty (Gt)",
            "Rate of mass balance anomaly (Gt/yr)",
            "Rate of ice dynamics anomaly (Gt/yr)",
            "Rate of mass balance anomaly uncertainty (Gt/yr)",
            "Rate of ice dyanamics anomaly uncertainty (Gt/yr)",
        ]
    ].rename(
        columns={
            "Cumulative ice sheet mass change (Gt)": "Mass (Gt)",
            "Cumulative ice sheet mass change uncertainty (Gt)": "Mass uncertainty (Gt)",
            "Cumulative surface mass balance anomaly (Gt)": "SMB (Gt)",
            "Cumulative surface mass balance anomaly uncertainty (Gt)": "SMB uncertainty (Gt)",
            "Cumulative ice dynamics anomaly (Gt)": "D (Gt)",
            "Cumulative ice dynamics anomaly uncertainty (Gt)": "D uncertainty (Gt)",
            "Rate of mass balance anomaly (Gt/yr)": "SMB (Gt/yr)",
            "Rate of ice dynamics anomaly (Gt/yr)": "D (Gt/yr)",
            "Rate of mass balance anomaly uncertainty (Gt/yr)": "SMB uncertainty (Gt/yr)",
            "Rate of ice dyanamics anomaly uncertainty (Gt/yr)": "D uncertainty (Gt/yr)",
        }
    )

    df = df[df["Year"] >= 1992.0]
    df["SMB (Gt/yr)"] += 2 * 1964 / 10
    df["D (Gt/yr)"] -= 2 * 1964 / 10
    cmSLE = 1.0 / 362.5 / 10.0
    df["SLE (cm)"] = -df["Mass (Gt)"] * cmSLE
    df["SLE uncertainty (cm)"] = df["Mass uncertainty (Gt)"] * cmSLE

    y = df["Year"].astype("int")
    df["Date"] = pd.to_datetime({"year": y, "month": 1, "day": 1}) + pd.to_timedelta(
        (df["Year"] - df["Year"].astype("int")) * 3.15569259747e7, "seconds"
    )

    return df


def plot_imbie(
    df: pd.DataFrame,
    sigma: float = 2,
    mass_color: str = "k",
    d_color: str = "#648fff",
    smb_color: str = "#dc267f",
):
    fig, axs = plt.subplots(nrows=2, ncols=1, sharex="col", figsize=(6.2, 4))
    fig.subplots_adjust(wspace=0.0, hspace=0.0)

    axs[0].fill_between(
        df["Date"],
        (df["Mass (Gt)"] + sigma * df["Mass uncertainty (Gt)"]) * gt2cmsle,
        (df["Mass (Gt)"] - sigma * df["Mass uncertainty (Gt)"]) * gt2cmsle,
        ls="solid",
        lw=0,
        alpha=0.5,
        color=mass_color,
        label=f"{sigma}-$\sigma$ DF",
    )
    axs[0].plot(df["Date"], df["Mass (Gt)"] * gt2cmsle, color=mass_color)

    axs[1].fill_between(
        df["Date"],
        (df["D (Gt/yr)"] + sigma * df["D uncertainty (Gt/yr)"]),
        (df["D (Gt/yr)"] - sigma * df["D uncertainty (Gt/yr)"]),
        ls="solid",
        lw=0,
        alpha=0.5,
        color=d_color,
    )
    axs[1].fill_between(
        df["Date"],
        (df["SMB (Gt/yr)"] + sigma * df["SMB uncertainty (Gt/yr)"]),
        (df["SMB (Gt/yr)"] - sigma * df["SMB uncertainty (Gt/yr)"]),
        ls="solid",
        lw=0,
        alpha=0.5,
        color=smb_color,
    )
    axs[1].plot(df["Date"], df["D (Gt/yr)"], color=d_color)
    axs[1].plot(df["Date"], df["SMB (Gt/yr)"], color=smb_color)

    axs[0].set_xlabel("")
    axs[0].set_ylabel("Contribution to sea-level\n since 1992 (cm SLE)")
    axs[1].set_xlabel("Year")
    axs[1].set_ylabel("Flux (Gt/yr)")
    legend = axs[0].legend()
    legend.get_frame().set_linewidth(0.0)
    legend.get_frame().set_alpha(0.0)
    fig.tight_layout()

    return fig
