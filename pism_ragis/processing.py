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

"""
Module for data processing
"""

import contextlib
import os
import pathlib
import re
import shutil
from calendar import isleap
from datetime import datetime
from pathlib import Path
from typing import Any, Hashable, List, Mapping, Union

import dask
import joblib
import numpy as np
import pandas as pd
import requests
import xarray as xr
from tqdm.auto import tqdm

kg2cmsle = 1 / 1e12 * 1.0 / 362.5 / 10.0
gt2cmsle = 1 / 362.5 / 10.0


def download_dataset(
    url: str = "https://dataverse.geus.dk/api/access/datafile/:persistentId?persistentId=doi:10.22008/FK2/OHI23Z/MRSBQR",
    chunk_size: int = 1024,
) -> xr.Dataset:
    """
    Download a dataset from the specified URL and return it as an xarray Dataset.

    Parameters
    ----------
    url : str, optional
        The URL of the dataset to download. Default is the mass balance dataset URL.
    chunk_size : int, optional
        The size of the chunks to download at a time, in bytes. Default is 1024 bytes.

    Returns
    -------
    xr.Dataset
        The downloaded dataset as an xarray Dataset.

    Examples
    --------
    >>> dataset = download_mass_balance()
    >>> print(dataset)
    """
    # Get the file size from the headers
    response = requests.head(url, timeout=10)
    file_size = int(response.headers.get("content-length", 0))

    # Initialize the progress bar
    progress = tqdm(total=file_size, unit="iB", unit_scale=True)

    # Download the file in chunks and update the progress bar
    print(f"Downloading {url}")
    with requests.get(url, stream=True, timeout=10) as r:
        r.raise_for_status()
        with open("temp.nc", "wb") as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                f.write(chunk)
                progress.update(len(chunk))
    progress.close()

    # Open the downloaded file with xarray
    return xr.open_dataset("temp.nc")


def days_in_year(year: int) -> int:
    """
    Calculate the number of days in a given year.

    Parameters
    ----------
    year : int
        The year for which to calculate the number of days.

    Returns
    -------
    int
        The number of days in the specified year. Returns 366 if the year is a leap year, otherwise returns 365.
    """
    if isleap(year):
        return 366
    else:
        return 365


def calculate_area(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """
    Calculate the area of each grid cell given arrays of latitudes and longitudes.

    Parameters
    ----------
    lat : np.ndarray
        Array of latitude values in degrees.
    lon : np.ndarray
        Array of longitude values in degrees.

    Returns
    -------
    np.ndarray
        2D array of grid cell areas in square meters.
    """
    lat_rad = np.deg2rad(lat)
    lon_rad = np.deg2rad(lon)

    # Calculate the distances between adjacent latitudes and longitudes
    dlon = np.diff(lon_rad)

    # Calculate the area of each grid cell
    R = 6371000  # Radius of the Earth in meters
    area = np.zeros((len(lat) - 1, len(lon) - 1))

    for i in range(len(lat) - 1):
        for j in range(len(lon) - 1):
            area[i, j] = (
                (R**2)
                * np.abs(np.sin(lat_rad[i + 1]) - np.sin(lat_rad[i]))
                * np.abs(dlon[j])
            )

    return area


def preprocess_nc(
    ds,
    regexp: str = "id_(.+?)_",
    dim: str = "exp_id",
    drop_vars: Union[List[str], None] = None,
    drop_dims: List[str] = ["nv4"],
):
    """
    Add experiment 'exp_id'
    """
    m_id_re = re.search(regexp, ds.encoding["source"])
    ds.expand_dims(dim)
    assert m_id_re is not None
    m_id: Union[str, int]
    try:
        m_id = int(m_id_re.group(1))
    except:
        m_id = str(m_id_re.group(1))
    ds[dim] = m_id

    return ds.drop_vars(drop_vars, errors="ignore").drop_dims(
        drop_dims, errors="ignore"
    )


def compute_basin(
    ds: xr.Dataset, name: str = "basin", dim: List = ["x", "y"]
) -> xr.Dataset:
    """
    Compute the sum of the dataset over the 'x' and 'y' dimensions and add a new dimension 'basin'.

    Parameters
    ----------
    ds : xr.Dataset
        The input dataset.
    name : str
        The name to assign to the new 'basin' dimension.

    Returns
    -------
    xr.Dataset
        The computed dataset with the new 'basin' dimension.

    Examples
    --------
    >>> ds = xr.Dataset({'var': (('x', 'y'), np.random.rand(5, 5))})
    >>> compute_basin(ds, 'new_basin')
    """
    ds = ds.sum(dim=dim).expand_dims("basin")
    ds["basin"] = [name]
    return ds.compute()


@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""

    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        """TQDM Callback"""

        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


def to_decimal_year(date):
    """Convert datetime date to decimal year"""
    year = date.year
    start_of_this_year = datetime(year=year, month=1, day=1)
    start_of_next_year = datetime(year=year + 1, month=1, day=1)
    year_elapsed = (date - start_of_this_year).total_seconds()
    year_duration = (start_of_next_year - start_of_this_year).total_seconds()
    fraction = year_elapsed / year_duration

    return date.year + fraction


def check_file(
    infile: Union[str, pathlib.Path], norm_year: Union[None, float] = None
) -> bool:
    """Check netCDF file"""
    with xr.open_dataset(infile) as ds:
        is_ok: bool = False
        if "time" in ds.indexes:
            datetimeindex = ds.indexes["time"]
            years = np.array(
                [to_decimal_year(x.to_pydatetime()) for x in datetimeindex]
            )
            monotonically_increasing = np.all(
                years.reshape(1, -1)[:, 1:] >= years.reshape(1, -1)[:, :-1], axis=1
            )[0]
            if norm_year:
                if (years[-1] >= norm_year) and monotonically_increasing:
                    is_ok = True
            else:
                print(
                    f"{infile} time-series too short or not monotonically-increasing."
                )
        return is_ok


def check_paleo_file(
    infile: Union[str, pathlib.Path], norm_year: Union[None, float] = None
) -> bool:
    """Check netCDF file"""
    with xr.open_dataset(infile) as ds:
        is_ok: bool = False
        if "time" in ds.indexes:
            datetimeindex = ds.indexes["time"]
            years = datetimeindex.year
            monotonically_increasing = np.all(
                years.reshape(1, -1)[:, 1:] >= years.reshape(1, -1)[:, :-1], axis=1
            )[0]
            if norm_year:
                if (years[-1] >= norm_year) and monotonically_increasing:
                    is_ok = True
            else:
                print(
                    f"{infile} time-series too short or not monotonically-increasing."
                )
        return is_ok


def copy_file(
    infile: Union[str, pathlib.Path], outdir: Union[str, pathlib.Path]
) -> None:
    """
    Copy infile to outdir
    """
    if infile is not pathlib.Path:
        in_path = pathlib.Path(infile)
    else:
        in_path = infile  # type: ignore
    if outdir is not pathlib.Path:
        out_path = pathlib.Path(outdir)
    else:
        out_path = outdir  # type: ignore
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    shutil.copy(in_path, out_path)


@xr.register_dataset_accessor("utils")
class UtilsMethods:
    """
    Utils methods for xarray Dataset.

    This class is used to add custom methods to xarray Dataset objects. The methods can be accessed via the 'utils' attribute.

    Parameters
    ----------

    xarray_obj : xr.Dataset
      The xarray Dataset to which to add the custom methods.
    """

    def __init__(self, xarray_obj: xr.Dataset):
        """
        Initialize the UtilsMethods class.

        Parameters
        ----------

        xarray_obj : xr.Dataset
            The xarray Dataset to which to add the custom methods.
        """
        self._obj = xarray_obj

    def init(self):
        """
        Do-nothing method.

        This method is needed to work with joblib Parallel.
        """

    def drop_nonnumeric_vars(self, errors: str = "ignore") -> xr.Dataset:
        """
        Drop non-numeric variables from the xarray Dataset.

        This method removes all variables from the xarray Dataset that do not have a numeric data type.

        Parameters
        ----------
        errors : {'ignore', 'raise'}, optional
            If 'ignore', suppress error and only drop existing variables.
            If 'raise', raise an error if any of the variables are not found in the dataset.
            Default is 'ignore'.

        Returns
        -------
        xarray.Dataset
            A new xarray Dataset with only numeric variables.

        Examples
        --------
        >>> import xarray as xr
        >>> import numpy as np
        >>> data = xr.Dataset({
        ...     'temperature': (('x', 'y'), [[15.5, 16.2], [14.8, 15.1]]),
        ...     'humidity': (('x', 'y'), [[80, 85], [78, 82]]),
        ...     'location': (('x', 'y'), [['A', 'B'], ['C', 'D']])
        ... })
        >>> processor = DataProcessor(data)
        >>> numeric_data = processor.drop_nonnumeric_vars()
        >>> print(numeric_data)
        <xarray.Dataset>
        Dimensions:     (x: 2, y: 2)
        Dimensions without coordinates: x, y
        Data variables:
            temperature  (x, y) float64 15.5 16.2 14.8 15.1
            humidity     (x, y) int64 80 85 78 82
        """
        nonnumeric_vars = [
            v
            for v in self._obj.data_vars
            if not np.issubdtype(self._obj[v].dtype, np.number)
        ]

        return self._obj.drop_vars(nonnumeric_vars, errors=errors)


def load_ensemble(
    filenames: List[Union[Path, str]],
    parallel: bool = True,
) -> xr.Dataset:
    """
    Load an ensemble of NetCDF files into an xarray Dataset.

    Parameters
    ----------
    filenames : List[Union[Path, str]]
        A list of file paths or strings representing the NetCDF files to be loaded.
    parallel : bool, optional
        Whether to load the files in parallel using Dask. Default is True.

    Returns
    -------
    xr.Dataset
        The loaded xarray Dataset containing the ensemble data.

    Notes
    -----
    This function uses Dask to load the dataset in parallel and handle large chunks efficiently.
    It sets the Dask configuration to split large chunks during array slicing.
    """
    with dask.config.set(**{"array.slicing.split_large_chunks": True}):
        print("Loading ensemble files... ", end="", flush=True)
        ds = xr.open_mfdataset(filenames, parallel=parallel, chunks="auto").drop_vars(
            ["spatial_ref", "mapping"], errors="ignore"
        )
        if "time" in ds["pism_config"].coords:
            ds["pism_config"] = ds["pism_config"].isel(time=0).drop_vars("time")
        print("Done.")
        return ds


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


def standardize_variable_names(
    ds: xr.Dataset, name_dict: Union[Mapping[Any, Hashable], None]
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


def convert_column_to_numeric(column):
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
