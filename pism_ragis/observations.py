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
Module provides functions to deal with observations.
"""

from functools import reduce
from pathlib import Path

import pandas as pd
import xarray as xr


def load_grace(
    url: Path | str = Path("../data/grace/gsfc_grace_mouginot_basins.nc"),
) -> xr.Dataset:
    """
    Load GRACE dataset from the specified URL or file path.

    Parameters
    ----------
    url : Path | str, optional
        The URL or file path to the GRACE dataset, by default "../data/grace/gsfc_grace_mouginot_basins.nc".

    Returns
    -------
    xr.Dataset
        The loaded GRACE dataset.

    Examples
    --------
    >>> ds = load_grace()
    >>> print(ds)
    """
    ds = xr.open_dataset(url)

    return ds


def load_imbie_2021(
    url: (
        str | Path
    ) = "https://ramadda.data.bas.ac.uk/repository/entry/get/imbie_greenland_2021_Gt.csv?entryid=synth:77b64c55-7166-4a06-9def-2e400398e452:L2ltYmllX2dyZWVubGFuZF8yMDIxX0d0LmNzdg==",
) -> xr.Dataset:
    """
    Load the IMBIE Greenland 2021 dataset from a CSV file and return as an xarray Dataset.

    Parameters
    ----------
    url : str | Path, optional
        The URL or file path to the IMBIE Greenland 2021 dataset, by default "https://ramadda.data.bas.ac.uk/repository/entry/get/imbie_greenland_2021_Gt.csv?entryid=synth:77b64c55-7166-4a06-9def-2e400398e452:L2ltYmllX2dyZWVubGFuZF8yMDIxX0d0LmNzdg==".

    Returns
    -------
    xr.Dataset
        The loaded IMBIE Greenland 2021 dataset.

    Examples
    --------
    >>> ds = load_imbie_2021()
    >>> print(ds)
    """

    df = pd.read_csv(url, skiprows=13)
    df["time"] = pd.to_datetime(df["time"], format="%Y-%m-%d")
    df = df.set_index("time")

    ds = df.to_xarray()

    return ds


def load_imbie(
    url: str | Path = "http://imbie.org/wp-content/uploads/2012/11/imbie_dataset_greenland_dynamics-2020_02_28.xlsx",
) -> xr.Dataset:
    """
    Load the IMBIE Greenland dataset from an Excel file and return as an xarray Dataset.

    This function loads the IMBIE Greenland dataset from the specified URL or file path,
    processes the data, and returns it as an xarray Dataset.

    Parameters
    ----------
    url : str | Path, optional
        The URL or file path to the IMBIE Greenland dataset, by default "http://imbie.org/wp-content/uploads/2012/11/imbie_dataset_greenland_dynamics-2020_02_28.xlsx".

    Returns
    -------
    xr.Dataset
        The loaded IMBIE Greenland dataset.

    Examples
    --------
    >>> ds = load_imbie()
    >>> print(ds)
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
            "Rate of ice sheet mass change certainty (Gt/yr)": "mass_balance_uncertainty",
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
    url: str | Path = "https://www.pnas.org/doi/suppl/10.1073/pnas.1904242116/suppl_file/pnas.1904242116.sd02.xlsx",
    norm_year: float | None = None,
) -> xr.Dataset:
    """
    Load the Mouginot et al (2019) dataset.

    This function loads the Mouginot et al (2019) dataset from the provided URL. The dataset is an Excel file with two sheets, "(2) MB_GIS" for the main data and "(2) MB_GIS" for the uncertainty data. The function processes the data and returns it as an xarray Dataset.

    Parameters
    ----------
    url : str or Path, optional
        The URL or local path of the Excel file to load, by default "https://www.pnas.org/doi/suppl/10.1073/pnas.1904242116/suppl_file/pnas.1904242116.sd02.xlsx".
    norm_year : None or float, optional
        The year to normalize the data to, by default None.

    Returns
    -------
    xr.Dataset
        The processed data as an xarray Dataset.

    Examples
    --------
    >>> ds = load_mouginot()
    >>> print(ds)
    """
    b_vars = ["mass_balance", "surface_mass_balance", "grounding_line_flux"]
    bu_vars = [f"{v}_uncertainty" for v in b_vars]
    cb_vars = [
        "cumulative_mass_balance",
        "cumulative_surface_mass_balance",
        "cumulative_grounding_line_flux",
    ]
    cbu_vars = [f"{v}_uncertainty" for v in cb_vars]

    # Load the main data and the uncertainty data
    df_m = pd.read_excel(url, sheet_name="(2) MB_GIS", header=8, usecols="B,P:BJ", engine="openpyxl")
    df_u = pd.read_excel(url, sheet_name="(2) MB_GIS", header=8, usecols="B,CE:DY", engine="openpyxl")

    # Process the main data
    dfs = []
    for k, m_var in zip(
        [0, 12, 22, 34, 45, 57],
        [
            "grounding_line_flux",
            "surface_mass_balance",
            "mass_balance",
            "cumulative_mass_balance",
            "cumulative_grounding_line_flux",
            "cumulative_surface_mass_balance",
        ],
    ):
        p_dfs = [process_row(row, m_var, norm_year) for _, row in df_m.loc[k : k + 7].iterrows()]
        dfs.append(pd.concat(p_dfs).reset_index(drop=True))

    df = reduce(
        lambda left, right: pd.merge(left, right, on=["basin", "year", "time"], how="outer"),
        dfs,
    )

    # Process the uncertainty data
    dfs_u = []
    for k, m_var in zip(
        [0, 12, 22, 34, 45, 57],
        [
            "grounding_line_flux_uncertainty",
            "surface_mass_balance_uncertainty",
            "mass_balance_uncertainty",
            "cumulative_mass_balance_uncertainty",
            "cumulative_grounding_line_flux_uncertainty",
            "cumulative_surface_mass_balance_uncertainty",
        ],
    ):
        p_dfs = [process_row(row, m_var, norm_year) for _, row in df_u.loc[k : k + 7].iterrows()]
        dfs_u.append(pd.concat(p_dfs).reset_index(drop=True))

    df_u = reduce(
        lambda left, right: pd.merge(left, right, on=["basin", "year", "time"], how="outer"),
        dfs_u,
    )

    m_df = pd.merge(df, df_u)
    m_df.set_index(["time", "basin"], inplace=True)

    ds = xr.Dataset.from_dataframe(m_df)

    for v in b_vars + bu_vars:
        ds[v].attrs["units"] = "Gt year-1"
    for v in cb_vars + cbu_vars:
        ds[v].attrs["units"] = "Gt"
    ds["grounding_line_flux"] *= -1

    return ds


def process_row(
    row: pd.Series,
    m_var: str,
    norm_year: float | None,
    norm_vars: list[str] = [
        "cumulative_mass_balance",
        "cumulative_surface_mass_balance",
        "cumulative_ice_discharge",
        "cumulative_mass_balance_uncertainty",
        "cumulative_ice_discharge_uncertainty",
        "cumulative_surface_mass_balance_uncertainty",
    ],
) -> pd.DataFrame:
    """
    Helper function to process a row of the data.

    This function takes a row of the data and a variable name, and returns a DataFrame with the processed data for that row.

    Parameters
    ----------
    row : pd.Series
        The row of the data to process.
    m_var : str
        The variable name for the data in the row.
    norm_year : float | None
        The year to normalize the data to, by default None.
    norm_vars : list of str, optional
        List of variables to normalize, by default includes cumulative mass balance and uncertainties.

    Returns
    -------
    pd.DataFrame
        The processed data for the row as a DataFrame.

    Examples
    --------
    >>> row = pd.Series(["Basin1", 1, 2, 3, 4, 5])
    >>> df = process_row(row, "mass_balance", 2000)
    >>> print(df)
    """
    df = pd.DataFrame(row.values[1::], columns=[m_var], dtype=float)
    df["year"] = range(1972, 2019)
    df["basin"] = row.values[0]
    df["time"] = pd.date_range(start="1972-01-01", end="2018-01-01", freq="YS")
    if (norm_year is not None) & (m_var in norm_vars):
        df[m_var] -= df[df["year"] == norm_year][m_var].values

    return df
