# Copyright (C) 2023-24 Andy Aschwanden
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
Module provides functions to deal with observations
"""

from functools import reduce
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import pylab as plt

from pism_ragis.processing import to_decimal_year

kg2cmsle = 1 / 1e12 * 1.0 / 362.5 / 10.0
gt2cmsle = 1 / 362.5 / 10.0


def load_imbie_csv(
    url: Union[str, Path] = "imbie_greenland_2021_Gt.csv", proj_start=1992
):
    """Loading the IMBIE Greenland data set from a CSV file and return pd.DataFrame"""
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
    df["SLE (cm)"] = df["Mass (Gt)"] * cmSLE
    df["SLE uncertainty (cm)"] = -df["Mass uncertainty (Gt)"] * cmSLE
    df["SLE change uncertainty (cm/yr)"] = (
        df["Mass change uncertainty (Gt/yr)"] * gt2cmsle
    )
    return df


def load_imbie(
    url: Union[
        str, Path
    ] = "http://imbie.org/wp-content/uploads/2012/11/imbie_dataset_greenland_dynamics-2020_02_28.xlsx",
):
    """
    Loading the IMBIE Greenland data set downloaded from
    http://imbie.org/wp-content/uploads/2012/11/imbie_dataset_greenland_dynamics-2020_02_28.xlsx
    and return pd.DataFrame.

    """
    df_df = pd.read_excel(
        url,
        sheet_name="Greenland Ice Mass",
        engine="openpyxl",
    )
    df = df_df.rename(
        columns={
            "Cumulative ice dynamics anomaly (Gt)": "Cumulative ice discharge anomaly (Gt)",
            "Cumulative ice dynamics anomaly uncertainty (Gt)": "Cumulative ice discharge anomaly uncertainty (Gt)",
            "Rate of mass balance anomaly (Gt/yr)": "Rate of surface mass balance anomaly (Gt/yr)",
            "Rate of ice dynamics anomaly (Gt/yr)": "Rate of ice discharge anomaly (Gt/yr)",
            "Rate of mass balance anomaly uncertainty (Gt/yr)": "Rate of surface mass balance anomaly uncertainty (Gt/yr)",
            "Rate of ice dyanamics anomaly uncertainty (Gt/yr)": "Rate of ice discharge anomaly uncertainty (Gt/yr)",
        }
    )

    # df = df[df["Year"] >= 1992.0]
    df["Rate of surface mass balance (Gt/yr)"] = (
        df["Rate of surface mass balance anomaly (Gt/yr)"] + 2 * 1964 / 10
    )
    df["Rate of ice discharge (Gt/yr)"] = (
        df["Rate of ice discharge anomaly (Gt/yr)"] - 2 * 1964 / 10
    )
    df["Rate of surface mass balance uncertainty (Gt/yr)"] = df[
        "Rate of surface mass balance anomaly uncertainty (Gt/yr)"
    ]
    df["Rate of ice discharge uncertainty (Gt/yr)"] = df[
        "Rate of ice discharge anomaly uncertainty (Gt/yr)"
    ]
    cmSLE = 1.0 / 362.5 / 10.0
    df["SLE (cm)"] = df["Cumulative ice sheet mass change (Gt)"] * cmSLE
    df["SLE uncertainty (cm)"] = (
        df["Cumulative ice sheet mass change uncertainty (Gt)"] * cmSLE
    )

    y = df["Year"].astype("int")
    df["Date"] = pd.to_datetime({"year": y, "month": 1, "day": 1}) + pd.to_timedelta(
        (df["Year"] - df["Year"].astype("int")) * 3.15569259747e7, "seconds"
    )

    return df


def load_mouginot(
    url: Union[
        str, Path
    ] = "https://www.pnas.org/doi/suppl/10.1073/pnas.1904242116/suppl_file/pnas.1904242116.sd02.xlsx",
    norm_year: Union[None, float] = None,
) -> pd.DataFrame:
    """
    Load the Mouginot et al (2019) data set.

    This function loads the Mouginot et al (2019) data set from the provided URL. The data set is an Excel file with two sheets, "(2) MB_GIS" for the main data and "(2) MB_GIS" for the uncertainty data. The function processes the data and returns a pandas DataFrame.

    Parameters
    ----------
    url : str or Path, optional
        The URL or local path of the Excel file to load, by default "https://www.pnas.org/doi/suppl/10.1073/pnas.1904242116/suppl_file/pnas.1904242116.sd02.xlsx".
    norm_year : None or float, optional
        The year to normalize the data to, by default None.

    Returns
    -------
    pd.DataFrame
        The processed data as a pandas DataFrame.

    Examples
    --------
    >>> df = load_mouginot()
    """
    # Load the main data and the uncertainty data
    df_m = pd.read_excel(
        url, sheet_name="(2) MB_GIS", header=8, usecols="B,P:BJ", engine="openpyxl"
    )
    df_u = pd.read_excel(
        url, sheet_name="(2) MB_GIS", header=8, usecols="B,CE:DY", engine="openpyxl"
    )

    # Process the main data
    dfs = []
    for k, m_var in zip(
        [0, 12, 22, 34, 45, 57],
        [
            "Rate of ice discharge (Gt/yr)",
            "Rate of surface mass balance (Gt/yr)",
            "Rate of ice sheet mass change (Gt/yr)",
            "Cumulative ice sheet mass change (Gt)",
            "Cumulative ice discharge anomaly (Gt)",
            "Cumulative surface mass balance anomaly (Gt)",
        ],
    ):
        p_dfs = [
            process_row(row, m_var, norm_year)
            for _, row in df_m.loc[k : k + 7].iterrows()
        ]
        dfs.append(pd.concat(p_dfs).reset_index(drop=True))

    df = reduce(
        lambda left, right: pd.merge(
            left, right, on=["Basin", "Year", "Date"], how="outer"
        ),
        dfs,
    )

    # Process the uncertainty data
    dfs_u = []
    for k, m_var in zip(
        [0, 12, 22, 34, 45, 57],
        [
            "Rate of ice discharge uncertainty (Gt/yr)",
            "Rate of surface mass balance uncertainty (Gt/yr)",
            "Rate of ice sheet mass change uncertainty (Gt/yr)",
            "Cumulative ice sheet mass change uncertainty (Gt)",
            "Cumulative ice discharge anomaly uncertainty (Gt)",
            "Cumulative surface mass balance anomaly uncertainty (Gt)",
        ],
    ):
        p_dfs = [
            process_row(row, m_var, norm_year)
            for _, row in df_u.loc[k : k + 7].iterrows()
        ]
        dfs_u.append(pd.concat(p_dfs).reset_index(drop=True))

    df_u = reduce(
        lambda left, right: pd.merge(
            left, right, on=["Basin", "Year", "Date"], how="outer"
        ),
        dfs_u,
    )

    return pd.merge(df, df_u)


def process_row(
    row,
    m_var,
    norm_year,
    norm_vars=[
        "Cumulative ice sheet mass change (Gt)",
        "Cumulative surface mass balance anomaly (Gt)",
        "Cumulative ice discharge anomaly (Gt)",
        "Cumulative ice sheet mass change uncertainty (Gt)",
        "Cumulative surface mass balance anomaly uncertainty (Gt)",
        "Cumulative ice discharge anomaly uncertainty (Gt)",
    ],
):
    """
    Helper function to process a row of the data.

    This function takes a row of the data and a variable name, and returns a DataFrame with the processed data for that row.

    Parameters
    ----------
    row : pd.Series
        The row of the data to process.
    m_var : str
        The variable name for the data in the row.

    Returns
    -------
    pd.DataFrame
        The processed data for the row as a DataFrame.
    """
    df = pd.DataFrame(row.values[1::], columns=[m_var], dtype=float)
    df["Year"] = range(1972, 2019)
    df["Basin"] = row.values[0]
    df["Date"] = pd.date_range(start="1972-1-1", end="2018-1-1", freq="YS")
    if (norm_year is not None) & (m_var in norm_vars):
        df[m_var] -= df[df["Year"] == norm_year][m_var].values

    return df


# def load_mouginot(
#     url: Union[
#         str, Path
#     ] = "https://www.pnas.org/doi/suppl/10.1073/pnas.1904242116/suppl_file/pnas.1904242116.sd02.xlsx",
#     norm_year: Union[None, float] = None,
# ):
#     """
#     Load the Mouginot et al (2019) data set
#     """

#     # req = Request(url)
#     # req.add_header('User-Agent', 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:77.0) Gecko/20100101 Firefox/124.0')
#     # content = urlopen(req)

#     df_m = pd.read_excel(
#         url,
#         sheet_name="(2) MB_GIS",
#         header=8,
#         usecols="B,P:BJ",
#         engine="openpyxl",
#     )

#     df_u = pd.read_excel(
#         url,
#         sheet_name="(2) MB_GIS",
#         header=8,
#         usecols="B,CE:DY",
#         engine="openpyxl",
#     )

#     dfs = []
#     for k, m_var in zip([0, 12, 22, 34, 45, 57], ["Rate of ice discharge (Gt/yr)",
#                                                   "Rate of surface mass balance (Gt/yr)",
#                                                   "Rate of ice sheet mass change (Gt/yr)",
#                                                   "Cumulative ice sheet mass change (Gt)",
#                                                   "Cumulative ice discharge anomaly (Gt)",
#                                                   "Cumulative surface mass balance anomaly (Gt)"]):
#         p_dfs = []
#         for _, row in df_m.loc[k:k+7].iterrows():
#             p_df = pd.DataFrame(row.values[1::], columns=[m_var], dtype=float)
#             p_df["Year"] = range(1972, 2019)
#             p_df["Basin"] = row.values[0]
#             p_df["Date"] = pd.date_range(start="1972-1-1", end="2018-1-1", freq="YS")
#             p_dfs.append(p_df)
#         dfs.append(pd.concat(p_dfs).reset_index(drop=True))
#     df = reduce(lambda  left,right: pd.merge(left, right,on=["Basin", "Year", "Date"],
#                                             how='outer'), dfs)
#     dfs_u = []
#     for k, m_var in zip([0, 12, 22, 34, 45, 57], ["Rate of ice discharge uncertainty (Gt/yr)",
#                                                   "Rate of surface mass balance uncertainty (Gt/yr)",
#                                                   "Rate of ice sheet mass change uncertainty (Gt/yr)",
#                                                   "Cumulative ice sheet mass change uncertainty (Gt)",
#                                                   "Cumulative ice discharge anomaly uncertainty (Gt)",
#                                                   "Cumulative surface mass balance anomaly uncertainty (Gt)"]):
#         p_dfs = []
#         for _, row in df_u.loc[k:k+7].iterrows():
#             p_df = pd.DataFrame(row.values[1::], columns=[m_var], dtype=float)
#             p_df["Year"] = range(1972, 2019)
#             p_df["Basin"] = row.values[0]
#             p_df["Date"] = pd.date_range(start="1972-1-1", end="2018-1-1", freq="YS")
#             p_dfs.append(p_df)
#         dfs_u.append(pd.concat(p_dfs).reset_index(drop=True))
#     df_u = reduce(lambda  left,right: pd.merge(left, right,on=["Basin", "Year", "Date"],
#                                             how='outer'), dfs_u)
#     return pd.merge(df, df_u)


def load_mankoff(
    url: Union[str, Path] = Path(
        "/Users/andy/Google Drive/My Drive/Projects/RAGIS/data/MB_SMB_D_BMB.csv"
    ),
    norm_year: Union[None, float] = None,
) -> pd.DataFrame:
    """
    Load Mass Balance from Mankoff
    """
    df = pd.read_csv(url, parse_dates=["time"])

    df = df.rename(
        columns={
            "time": "Date",
            "MB": "Rate of ice sheet mass change (Gt/day)",
            "SMB": "Rate of surface mass balance (Gt/day)",
            "D": "Rate of ice discharge (Gt/day)",
            "MB_err": "Rate of ice sheet mass change uncertainty (Gt/day)",
            "SMB_err": "Rate of surface mass balance uncertainty (Gt/day)",
            "D_err": "Rate of ice discharge uncertainty (Gt/day)",
        }
    )

    days_per_year = np.where(df["Date"].dt.is_leap_year, 366, 365)
    time = df[["Date"]]
    time["delta"] = 1
    time["delta"][df["Date"] < "1986-01-01"] = days_per_year[df["Date"] < "1986-01-01"]
    df = df[
        [
            "Rate of ice sheet mass change (Gt/day)",
            "Rate of surface mass balance (Gt/day)",
            "Rate of ice discharge (Gt/day)",
            "Rate of ice sheet mass change uncertainty (Gt/day)",
            "Rate of surface mass balance uncertainty (Gt/day)",
            "Rate of ice discharge uncertainty (Gt/day)",
        ]
    ]
    # df = df.set_index(time["Date"]).resample("1D").mean().ffill().reset_index(drop=True)
    df["Cumulative ice sheet mass change (Gt)"] = (
        df["Rate of ice sheet mass change (Gt/day)"].multiply(time["delta"]).cumsum()
    )
    df["Cumulative ice discharge anomaly (Gt)"] = (
        df["Rate of ice discharge (Gt/day)"].multiply(time["delta"]).cumsum()
    )
    df["Cumulative surface mass balance anomaly (Gt)"] = (
        df["Rate of surface mass balance (Gt/day)"].multiply(time["delta"]).cumsum()
    )
    df["Cumulative ice sheet mass change uncertainty (Gt)"] = (
        df["Rate of ice sheet mass change uncertainty (Gt/day)"]
        .apply(np.square)
        .cumsum()
        .apply(np.sqrt)
    )
    df["Cumulative surface mass balance anomaly uncertainty (Gt)"] = (
        df["Rate of surface mass balance uncertainty (Gt/day)"]
        .apply(np.square)
        .cumsum()
        .apply(np.sqrt)
    )
    df["Cumulative ice discharge anomaly uncertainty (Gt)"] = (
        df["Rate of ice discharge uncertainty (Gt/day)"]
        .apply(np.square)
        .cumsum()
        .apply(np.sqrt)
    )
    df["Rate of ice sheet mass change (Gt/yr)"] = df[
        "Rate of ice sheet mass change (Gt/day)"
    ].multiply(days_per_year)
    df["Rate of ice discharge (Gt/yr)"] = df["Rate of ice discharge (Gt/day)"].multiply(
        days_per_year
    )
    df["Rate of surface mass balance (Gt/yr)"] = df[
        "Rate of surface mass balance (Gt/day)"
    ].multiply(days_per_year)
    df = pd.merge(df, time, left_index=True, right_index=True)
    df["Year"] = [to_decimal_year(d) for d in df["Date"]]

    cmSLE = 1.0 / 362.5 / 10.0
    df["SLE (cm)"] = df["Cumulative ice sheet mass change (Gt)"] * cmSLE
    df["SLE uncertainty (cm)"] = (
        df["Cumulative ice sheet mass change uncertainty (Gt)"] * cmSLE
    )
    if norm_year:
        # Normalize
        for v in [
            "Cumulative ice sheet mass change (Gt)",
            "Cumulative surface mass balance anomaly (Gt)",
            "Cumulative ice discharge anomaly (Gt)",
            "Cumulative ice sheet mass change uncertainty (Gt)",
            "Cumulative surface mass balance anomaly uncertainty (Gt)",
            "Cumulative ice discharge anomaly uncertainty (Gt)",
        ]:
            df[v] -= df[df["Year"] == norm_year][v].values

    return df


def plot_observations(
    df: pd.DataFrame,
    sigma: float = 1,
    title: Union[None, str] = None,
    norm_year: Union[None, float] = None,
    mass_varname: str = "Cumulative ice sheet mass change (Gt)",
    mass_uncertainty_varname: str = "Cumulative ice sheet mass change uncertainty (Gt)",
    smb_varname: str = "Cumulative surface mass balance anomaly (Gt)",
    smb_uncertainty_varname: str = "Cumulative surface mass balance anomaly uncertainty (Gt)",
    discharge_varname: str = "Cumulative ice discharge anomaly (Gt)",
    discharge_uncertainty_varname: str = "Cumulative ice discharge anomaly uncertainty (Gt)",
    mass_rate_varname: str = "Rate of ice sheet mass change (Gt/yr)",
    mass_rate_uncertainty_varname: str = "Rate of ice sheet mass change uncertainty (Gt/yr)",
    smb_rate_varname: str = "Rate of surface mass balance (Gt/yr)",
    smb_rate_uncertainty_varname: str = "Rate of surface mass balance uncertainty (Gt/yr)",
    discharge_rate_varname: str = "Rate of ice discharge (Gt/yr)",
    discharge_rate_uncertainty_varname: str = "Rate of ice discharge uncertainty (Gt/yr)",
    smb_color: str = "#bae4bc",
    discharge_color: str = "#7bccc4",
    mass_color: str = "#2b8cbe",
) -> plt.Figure:
    """Plot observation time series"""
    fig, axs = plt.subplots(nrows=2, ncols=1, sharex="col", figsize=(6.2, 4))
    fig.subplots_adjust(wspace=0.0, hspace=0.0)

    for m_var, m_u_var, m_color, m_label in zip(
        [mass_varname, smb_varname, discharge_varname],
        [
            mass_uncertainty_varname,
            smb_uncertainty_varname,
            discharge_uncertainty_varname,
        ],
        [mass_color, smb_color, discharge_color],
        ["Total", "Surface", "Ice Discharge"],
    ):
        if norm_year:
            d_m_var_normed = df[m_var] - df[df["Year"] == norm_year][m_var].values
            df[m_var] = d_m_var_normed

        axs[0].fill_between(
            df["Date"],
            df[m_var] + sigma * df[m_u_var],
            df[m_var] - sigma * df[m_u_var],
            ls="solid",
            lw=0,
            alpha=0.6,
            color=m_color,
        )
        axs[0].plot(df["Date"], df[m_var], lw=1, color=m_color, label=m_label)

    for m_var, m_u_var, m_color, m_label in zip(
        [mass_rate_varname, smb_rate_varname, discharge_rate_varname],
        [
            mass_rate_uncertainty_varname,
            smb_rate_uncertainty_varname,
            discharge_rate_uncertainty_varname,
        ],
        [mass_color, smb_color, discharge_color],
        ["Total", "Surface", "Ice Discharge"],
    ):
        axs[1].fill_between(
            df["Date"],
            df[m_var] + sigma * df[m_u_var],
            df[m_var] - sigma * df[m_u_var],
            ls="solid",
            lw=0,
            alpha=0.6,
            color=m_color,
        )
        axs[1].plot(df["Date"], df[m_var], lw=1, color=m_color, label=m_label)

    axs[0].axhline(0, ls="dashed", color="k", lw=0.5)
    axs[1].axhline(0, ls="dashed", color="k", lw=0.5)
    axs[0].set_xlabel("")
    axs[0].set_ylabel("Mass change (Gt)")
    axs[1].set_xlabel("Year")
    axs[1].set_ylabel("Rate (Gt/yr)")
    legend = axs[0].legend()
    legend.get_frame().set_linewidth(0.0)
    legend.get_frame().set_alpha(0.0)
    if title:
        axs[0].set_title(title)
    fig.tight_layout()

    return fig


def plot_multiple_observations(
    dfs: list[pd.DataFrame],
    sigma: float = 1,
    title: Union[None, str] = None,
    norm_year: Union[None, float] = None,
    mass_varname: str = "Cumulative ice sheet mass change (Gt)",
    mass_uncertainty_varname: str = "Cumulative ice sheet mass change uncertainty (Gt)",
    smb_varname: str = "Cumulative surface mass balance anomaly (Gt)",
    smb_uncertainty_varname: str = "Cumulative surface mass balance anomaly uncertainty (Gt)",
    discharge_varname: str = "Cumulative ice discharge anomaly (Gt)",
    discharge_uncertainty_varname: str = "Cumulative ice discharge anomaly uncertainty (Gt)",
    mass_rate_varname: str = "Rate of ice sheet mass change (Gt/yr)",
    mass_rate_uncertainty_varname: str = "Rate of ice sheet mass change uncertainty (Gt/yr)",
    smb_rate_varname: str = "Rate of surface mass balance (Gt/yr)",
    smb_rate_uncertainty_varname: str = "Rate of surface mass balance uncertainty (Gt/yr)",
    discharge_rate_varname: str = "Rate of ice discharge (Gt/yr)",
    discharge_rate_uncertainty_varname: str = "Rate of ice discharge uncertainty (Gt/yr)",
    smb_colors: list[str] = ["#bae4bc", "#b3cde3", "#fdcc8a"],
    discharge_colors: list[str] = ["#7bccc4", "#8c96c6", "#fc8d59,"],
    mass_colors: list[str] = ["#2b8cbe", "#88419d", "#d7301f"],
) -> plt.Figure:
    """Plot multiple observation time series"""
    fig, axs = plt.subplots(
        nrows=2, ncols=1, sharex="col", figsize=(6.2, 4.2), height_ratios=[16, 9]
    )
    fig.subplots_adjust(wspace=0.0, hspace=0.0)

    for k, df in enumerate(dfs):
        for m_var, m_u_var, m_color, m_label in zip(
            [mass_varname, smb_varname, discharge_varname],
            [
                mass_uncertainty_varname,
                smb_uncertainty_varname,
                discharge_uncertainty_varname,
            ],
            [mass_colors[k], smb_colors[k], discharge_colors[k]],
            ["Total", "Surface", "Ice Discharge"],
        ):
            if norm_year:
                d_m_var_normed = df[m_var] - df[df["Year"] == norm_year][m_var].values
                df[m_var] = d_m_var_normed

            axs[0].fill_between(
                df["Date"],
                df[m_var] + sigma * df[m_u_var],
                df[m_var] - sigma * df[m_u_var],
                ls="solid",
                lw=0,
                alpha=0.7,
                color=m_color,
            )
            axs[0].plot(df["Date"], df[m_var], lw=1, color=m_color, label=m_label)

        for m_var, m_u_var, m_color, m_label in zip(
            [mass_rate_varname, smb_rate_varname, discharge_rate_varname],
            [
                mass_rate_uncertainty_varname,
                smb_rate_uncertainty_varname,
                discharge_rate_uncertainty_varname,
            ],
            [mass_colors[k], smb_colors[k], discharge_colors[k]],
            ["Total", "Surface", "Ice Discharge"],
        ):
            axs[1].fill_between(
                df["Date"],
                df[m_var] + sigma * df[m_u_var],
                df[m_var] - sigma * df[m_u_var],
                ls="solid",
                lw=0,
                alpha=0.7,
                color=m_color,
            )
            axs[1].plot(df["Date"], df[m_var], lw=1, color=m_color, label=m_label)
    legend = axs[0].legend()
    legend.get_frame().set_linewidth(0.0)
    legend.get_frame().set_alpha(0.0)

    axs[0].axhline(0, ls="dashed", color="k", lw=0.5)
    axs[1].axhline(0, ls="dashed", color="k", lw=0.5)
    axs[0].set_xlabel("")
    axs[0].set_ylabel("Mass change (Gt)")
    axs[1].set_xlabel("Year")
    axs[1].set_ylabel("Rate (Gt/yr)")
    if title:
        axs[0].set_title(title)
    fig.tight_layout()

    return fig
