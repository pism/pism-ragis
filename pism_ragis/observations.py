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

import pandas as pd
import pylab as plt
import xarray as xr


def load_grace(
    url: Union[Path, str] = Path("../data/grace/gsfc_grace_mouginot_basins.nc"),
) -> xr.Dataset:
    """
    Load GRACE.
    """

    ds = xr.open_dataset(url)

    return ds


def load_imbie_2021(
    url: Union[
        str, Path
    ] = "https://ramadda.data.bas.ac.uk/repository/entry/get/imbie_greenland_2021_Gt.csv?entryid=synth:77b64c55-7166-4a06-9def-2e400398e452:L2ltYmllX2dyZWVubGFuZF8yMDIxX0d0LmNzdg=="
) -> xr.Dataset:
    """Loading the IMBIE Greenland data set from a CSV file and return xr.DataSet"""

    b_vars = ["mass_balance"]
    bu_vars = [f"{v}_uncertainty" for v in b_vars]
    cb_vars = [
        "cumulative_mass_balance",
    ]
    cbu_vars = [f"{v}_uncertainty" for v in cb_vars]

    df = pd.read_csv(url)
    df = df.rename(
        columns={
            "Year": "year",
            "Mass balance (Gt/yr)": "mass_balance",
            "Mass balance uncertainty (Gt/yr)": "mass_balance_uncertainty",
            "Cumulative mass balance (Gt)": "cumulative_mass_balance",
            "Cumulative mass balance uncertainty (Gt)": "cumulative_mass_balance_uncertainty",
        }
    )

    date = pd.date_range(start="1992-01-01", periods=len(df), freq="1MS")
    df.set_index(date, inplace=True)
    ds = xr.Dataset.from_dataframe(df)
    ds = ds.rename_dims({"index": "time"})
    ds = ds.rename_vars({"index": "time"})

    for v in b_vars + bu_vars:
        ds[v].attrs["units"] = "Gt year-1"
    for v in cb_vars + cbu_vars:
        ds[v].attrs["units"] = "Gt"

    return ds


def load_imbie(
    url: Union[
        str, Path
    ] = "http://imbie.org/wp-content/uploads/2012/11/imbie_dataset_greenland_dynamics-2020_02_28.xlsx",
) -> xr.Dataset:
    """
    Loading the IMBIE Greenland data set downloaded from
    http://imbie.org/wp-content/uploads/2012/11/imbie_dataset_greenland_dynamics-2020_02_28.xlsx
    and return xr.Dataset.

    """
    b_vars = ["mass_balance", "surface_mass_balance", "ice_discharge"]
    bu_vars = [f"{v}_uncertainty" for v in b_vars]
    cb_vars = [
        "cumulative_mass_balance",
        "cumulative_surface_mass_balance",
        "cumulative_ice_discharge",
    ]
    cbu_vars = [f"{v}_uncertainty" for v in cb_vars]

    df = pd.read_excel(
        url,
        sheet_name="Greenland Ice Mass",
        engine="openpyxl",
    )
    df = df.rename(
        columns={
            "Rate of ice sheet mass change (Gt/yr)": "mass_balance",
            "Rate of ice sheet mass change uncertainty (Gt/yr)": "mass_balance_uncertainty",
            "Rate of mass balance anomaly (Gt/yr)": "surface_mass_balance",
            "Rate of ice dynamics anomaly (Gt/yr)": "ice_discharge",
            "Rate of mass balance anomaly uncertainty (Gt/yr)": "surface_mass_balance_uncertainty",
            "Rate of ice dyanamics anomaly uncertainty (Gt/yr)": "ice_discharge_uncertainty",
            "Cumulative ice sheet mass change (Gt)": "cumulative_mass_balance",
            "Cumulative ice sheet mass change uncertainty (Gt)": "cumulative_mass_balance_uncertainty",
            "Cumulative ice dynamics anomaly (Gt)": "cumulative_ice_discharge",
            "Cumulative ice dynamics anomaly uncertainty (Gt)": "cumulative_ice_discharge_uncertainty",
            "Cumulative surface mass balance anomaly (Gt)": "cumulative_surface_mass_balance",
            "Cumulative surface mass balance anomaly uncertainty (Gt)": "cumulative_surface_mass_balance_uncertainty",
        }
    )
    date = pd.date_range(start="1980-01-01", periods=len(df), freq="1MS")
    df.set_index(date, inplace=True)
    df["surface_mass_balance"] += 2 * 1964 / 10
    df["ice_discharge"] -= 2 * 1964 / 10

    ds = xr.Dataset.from_dataframe(df)
    ds = ds.rename_dims({"index": "time"})
    ds = ds.rename_vars({"index": "time"})

    for v in b_vars + bu_vars:
        ds[v].attrs["units"] = "Gt year-1"
    for v in cb_vars + cbu_vars:
        ds[v].attrs["units"] = "Gt"

    return ds


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

    b_vars = ["mass_balance", "surface_mass_balance", "ice_discharge"]
    bu_vars = [f"{v}_uncertainty" for v in b_vars]
    cb_vars = [
        "cumulative_mass_balance",
        "cumulative_surface_mass_balance",
        "cumulative_ice_discharge",
    ]
    cbu_vars = [f"{v}_uncertainty" for v in cb_vars]

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
            "ice_discharge",
            "surface_mass_balance",
            "mass_balance",
            "cumulative_mass_balance",
            "cumulative_ice_discharge",
            "cumulative_surface_mass_balance",
        ],
    ):
        p_dfs = [
            process_row(row, m_var, norm_year)
            for _, row in df_m.loc[k : k + 7].iterrows()
        ]
        dfs.append(pd.concat(p_dfs).reset_index(drop=True))

    df = reduce(
        lambda left, right: pd.merge(
            left, right, on=["basin", "year", "time"], how="outer"
        ),
        dfs,
    )

    # Process the uncertainty data
    dfs_u = []
    for k, m_var in zip(
        [0, 12, 22, 34, 45, 57],
        [
            "ice_discharge_uncertainty",
            "surface_mass_balance_uncertainty",
            "mass_balance_uncertainty",
            "cumulative_mass_balance_uncertainty",
            "cumulative_ice_discharge_uncertainty",
            "cumulative_surface_mass_balance_uncertainty",
        ],
    ):
        p_dfs = [
            process_row(row, m_var, norm_year)
            for _, row in df_u.loc[k : k + 7].iterrows()
        ]
        dfs_u.append(pd.concat(p_dfs).reset_index(drop=True))

    df_u = reduce(
        lambda left, right: pd.merge(
            left, right, on=["basin", "year", "time"], how="outer"
        ),
        dfs_u,
    )

    m_df = pd.merge(df, df_u)
    m_df.set_index(["time", "basin"], inplace=True)

    ds = xr.Dataset.from_dataframe(m_df)

    for v in b_vars + bu_vars:
        ds[v].attrs["units"] = "Gt year-1"
    for v in cb_vars + cbu_vars:
        ds[v].attrs["units"] = "Gt"
    ds["ice_discharge"] *= -1

    return ds


def process_row(
    row,
    m_var,
    norm_year,
    norm_vars=[
        "cumulative_mass_balance",
        "cumulative_surface_mass_balance",
        "cumulative_ice_discharge",
        "cumulative_mass_balance_uncertainty",
        "cumulative_ice_discharge_uncertainty",
        "cumulative_surface_mass_balance_uncertainty",
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
    df["year"] = range(1972, 2019)
    df["basin"] = row.values[0]
    df["time"] = pd.date_range(start="1972-01-01", end="2018-01-01", freq="YS")
    if (norm_year is not None) & (m_var in norm_vars):
        df[m_var] -= df[df["year"] == norm_year][m_var].values

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
