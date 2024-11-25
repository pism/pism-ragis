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

# pylint: disable=too-many-positional-arguments

"""
Module for data processing
"""

import contextlib
import datetime
import os
import pathlib
import re
import shutil
import zipfile
from calendar import isleap
from pathlib import Path
from typing import Any, Dict, Hashable, List, Mapping, Union

import joblib
import numpy as np
import pandas as pd
import xarray as xr
from tqdm.auto import tqdm

from pism_ragis.decorators import timeit
from pism_ragis.logger import get_logger

logger = get_logger(__name__)


def unzip_file(zip_path: str, extract_to: str, overwrite: bool = False) -> None:
    """
    Unzip a file to a specified directory with a progress bar and optional overwrite.

    Parameters
    ----------
    zip_path : str
        The path to the ZIP file.
    extract_to : str
        The directory where the contents will be extracted.
    overwrite : bool, optional
        Whether to overwrite existing files, by default False.
    """
    # Ensure the extract_to directory exists
    Path(extract_to).mkdir(parents=True, exist_ok=True)

    # Open the ZIP file
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        # Get the list of file names in the zip file
        file_list = zip_ref.namelist()

        # Iterate over the file names with a progress bar
        for file in tqdm(file_list, desc="Extracting files", unit="file"):
            file_path = Path(extract_to) / file
            if not file_path.exists() or overwrite:
                zip_ref.extract(member=file, path=extract_to)


def decimal_year_to_datetime(decimal_year: float) -> datetime.datetime:
    """
    Convert a decimal year to a datetime object.

    Parameters
    ----------
    decimal_year : float
        The decimal year to be converted.

    Returns
    -------
    datetime.datetime
        The corresponding datetime object.

    Notes
    -----
    The function calculates the date by determining the start of the year and adding
    the fractional part of the year as days. If the resulting date has an hour value
    of 12 or more, it rounds up to the next day and sets the time to midnight.
    """
    year = int(decimal_year)
    remainder = decimal_year - year
    start_of_year = datetime.datetime(year, 1, 1)
    days_in_year = (datetime.datetime(year + 1, 1, 1) - start_of_year).days
    date = start_of_year + datetime.timedelta(days=remainder * days_in_year)
    if date.hour >= 12:
        date = date + datetime.timedelta(days=1)
    return date.replace(hour=0, minute=0, second=0, microsecond=0)


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


def preprocess_time(
    ds,
    regexp: str = "ERA5-(.+?).nc",
    freq: str = "MS",
    periods: int | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    drop_vars: List[str] | None = None,
    drop_dims: List[str] = ["nv4"],
):
    """
    Add correct time and time_bounds to the dataset.

    This function processes the time coordinates of the dataset by extracting the year
    from the filename using a regular expression, creating a time range, centering the time,
    and adding time bounds. It also drops specified variables and dimensions from the dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        The input dataset to be processed.
    regexp : str, optional
        The regular expression pattern to extract the year from the filename, by default "ERA5-(.+?).nc".
    freq : str, optional
        The frequency string to create the time range, by default "MS".
    drop_vars : Union[List[str], None], optional
        A list of variable names to be dropped from the dataset, by default None.
    drop_dims : List[str], optional
        A list of dimension names to be dropped from the dataset, by default ["nv4"].

    Returns
    -------
    xarray.Dataset
        The processed dataset with updated time coordinates and bounds, and specified variables and dimensions dropped.

    Raises
    ------
    AssertionError
        If the regular expression does not match any part of the filename.
    """

    if "time" not in ds.coords:
        nt = 1
        ds = ds.expand_dims(dim="time", axis=0)
    else:
        nt = ds.time.size

    if start_date and end_date and periods:
        time = xr.cftime_range(start_date, end_date, periods=periods)
    else:
        m_year_re = re.search(regexp, ds.encoding["source"])
        assert m_year_re is not None
        m_year = m_year_re.group(1)
        time = xr.cftime_range(m_year, freq=freq, periods=nt + 1)
    time_centered = time[:-1] + (time[1:] - time[:-1]) / 2
    ds = ds.assign_coords(time=time_centered)

    time_bounds = xr.DataArray(
        np.vstack([time[:-1], time[1:]]).T, dims=["time", "bounds"]
    )
    # Add bounds to the dataset
    ds = ds.assign_coords(time_bounds=time_bounds)

    return ds.drop_vars(drop_vars, errors="ignore").drop_dims(
        drop_dims, errors="ignore"
    )


def preprocess_nc(
    ds,
    regexp: str = "id_(.+?)_",
    dim: str = "exp_id",
    drop_vars: Union[List[str], None] = None,
    drop_dims: List[str] = ["nv4"],
):
    """
    Add experiment identifier to the dataset.

    This function processes the dataset by extracting an experiment identifier from the filename
    using a regular expression, adding it as a new dimension, and optionally dropping specified
    variables and dimensions from the dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        The input dataset to be processed.
    regexp : str, optional
        The regular expression pattern to extract the experiment identifier from the filename, by default "id_(.+?)_".
    dim : str, optional
        The name of the new dimension to be added to the dataset, by default "exp_id".
    drop_vars : Union[List[str], None], optional
        A list of variable names to be dropped from the dataset, by default None.
    drop_dims : List[str], optional
        A list of dimension names to be dropped from the dataset, by default ["nv4"].

    Returns
    -------
    xarray.Dataset
        The processed dataset with the experiment identifier added as a new dimension, and specified variables and dimensions dropped.

    Raises
    ------
    AssertionError
        If the regular expression does not match any part of the filename.
    """
    m_id_re = re.search(regexp, ds.encoding["source"])
    ds = ds.expand_dims(dim)
    assert m_id_re is not None
    m_id: Union[str, int]
    try:
        m_id = int(m_id_re.group(1))
    except:
        m_id = str(m_id_re.group(1))
    ds[dim] = [m_id]

    return ds.drop_vars(drop_vars, errors="ignore").drop_dims(
        drop_dims, errors="ignore"
    )


def preprocess_scalar_nc(
    ds,
    regexp: str = "id_(.+?)_",
    dim: str = "exp_id",
    ensemble_dim: str = "ensemble_id",
    basin_dim: str = "basin",
    ensemble_id: str = "RAGIS",
    basin: str = "GIS",
    drop_vars: Union[List[str], None] = None,
    drop_dims: List[str] = ["nv4"],
):
    """
    Add experiment identifier and additional dimensions to the dataset.

    This function processes the dataset by extracting an experiment identifier from the filename
    using a regular expression, adding it along with ensemble and basin identifiers as new dimensions.
    It also processes specific variables, merges configuration and statistics data, and optionally
    drops specified variables and dimensions from the dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        The input dataset to be processed.
    regexp : str, optional
        The regular expression pattern to extract the experiment identifier from the filename, by default "id_(.+?)_".
    dim : str, optional
        The name of the new experiment identifier dimension to be added to the dataset, by default "exp_id".
    ensemble_dim : str, optional
        The name of the new ensemble identifier dimension to be added to the dataset, by default "ensemble_id".
    basin_dim : str, optional
        The name of the new basin identifier dimension to be added to the dataset, by default "basin".
    ensemble_id : str, optional
        The value of the ensemble identifier to be added, by default "RAGIS".
    basin : str, optional
        The value of the basin identifier to be added, by default "GIS".
    drop_vars : Union[List[str], None], optional
        A list of variable names to be dropped from the dataset, by default None.
    drop_dims : List[str], optional
        A list of dimension names to be dropped from the dataset, by default ["nv4"].

    Returns
    -------
    xarray.Dataset
        The processed dataset with the experiment, ensemble, and basin identifiers added as new dimensions,
        and specified variables and dimensions dropped.

    Raises
    ------
    AssertionError
        If the regular expression does not match any part of the filename.
    """
    config = ds["pism_config"]
    stats = ds["run_stats"]
    encoding = ds.encoding

    pism_config = xr.DataArray(
        list(config.attrs.values()),
        dims=["pism_config_axis"],
        coords={"pism_config_axis": list(config.attrs.keys())},
        name="pism_config",
    )
    run_stats = xr.DataArray(
        list(stats.attrs.values()),
        dims=["run_stats_axis"],
        coords={"run_stats_axis": list(stats.attrs.keys())},
        name="run_stats",
    )
    if "ice_mass" in ds:
        ds["ice_mass"] /= 1e12
        ds["ice_mass"].attrs["units"] = "Gt"
    if "ice_mass_glacierized" in ds:
        ds["ice_mass_glacierized"] /= 1e12
        ds["ice_mass_glacierized"].attrs["units"] = "Gt"
    ds = xr.merge([ds.drop_vars(["pism_config", "run_stats"]), pism_config, run_stats])
    ds.encoding.update(encoding)
    m_id_re = re.search(regexp, ds.encoding["source"])
    ds = ds.expand_dims(dim)
    ds = ds.expand_dims(ensemble_dim)
    ds[ensemble_dim] = [ensemble_id]
    ds = ds.expand_dims(basin_dim)
    ds[basin_dim] = [basin]
    assert m_id_re is not None
    m_id: Union[str, int]
    try:
        m_id = int(m_id_re.group(1))
    except:
        m_id = str(m_id_re.group(1))
    ds[dim] = [m_id]

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
    ds = ds.sum(dim=dim).expand_dims("basin", axis=1)
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
    start_of_this_year = datetime.datetime(year=year, month=1, day=1)
    start_of_next_year = datetime.datetime(year=year + 1, month=1, day=1)
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


@timeit
def load_ensemble(
    filenames: List[Union[Path, str]], parallel: bool = True, engine: str = "netcdf4"
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
    """
    print("Loading ensemble files... ", end="", flush=True)
    ds = xr.open_mfdataset(
        filenames,
        parallel=parallel,
        chunks={"exp_id": -1, "pism_config_axis": -1},
        engine=engine,
    ).drop_vars(["spatial_ref", "mapping"], errors="ignore")
    print("Done.")
    return ds


@timeit
def normalize_cumulative_variables(
    ds: xr.Dataset, variables, reference_date: str = "1992-01-01"
) -> xr.Dataset:
    """
    Normalize cumulative variables in an xarray Dataset by subtracting their values at a reference year.

    Parameters
    ----------
    ds : xr.Dataset
        The xarray Dataset containing the cumulative variables to be normalized.
    variables : str or list of str
        The name(s) of the cumulative variables to be normalized.
    reference_year : float, optional
        The reference year to use for normalization. Default is 1992.0.

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
    ds[variables] -= ds[variables].sel(time=reference_date, method="nearest")
    return ds


@timeit
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


def select_experiment(ds: xr.Dataset, exp_id: str, n: int) -> xr.Dataset:
    """
    Reset the experiment id.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset containing the experiment data.
    exp_id : str
        The experiment id to select.
    n : int
        The new experiment id to set.

    Returns
    -------
    xr.Dataset
        The dataset with the updated experiment id.
    """
    exp = ds.sel(exp_id=exp_id)
    exp["exp_id"] = n
    return exp


def simplify_path(my_str: str) -> str:
    """
    Simplify string by extracting the file name.

    Parameters
    ----------
    my_str : str
        The input string representing a file path.

    Returns
    -------
    str
        The simplified string (file name).
    """
    return Path(my_str).name


def simplify_climate(my_str: str) -> str:
    """
    Simplify climate string.

    This function simplifies the input climate string by returning a standardized
    climate model name based on the presence of specific substrings.

    Parameters
    ----------
    my_str : str
        The input climate string.

    Returns
    -------
    str
        The standardized climate model name based on the input string.

    Examples
    --------
    >>> simplify_climate("MAR_2020")
    'MAR'
    >>> simplify_climate("RACMO_2020")
    'RACMO'
    >>> simplify_climate("Other_2020")
    'HIRHAM'
    """
    climate_mapping: Dict[str, str] = {
        "MAR": "MAR",
        "RACMO": "RACMO",
        "HIRHAM": "HIRHAM",
    }

    for key in climate_mapping:  # pylint: disable=consider-using-dict-items
        if key in my_str:
            return climate_mapping[key]

    return "HIRHAM"


def simplify_ocean(my_str: str) -> str:
    """
    Simplify ocean string.

    Parameters
    ----------
    my_str : str
        The input ocean string.

    Returns
    -------
    str
        The simplified ocean string.
    """
    return "-".join(my_str.split("_")[1:2])


def simplify_calving(my_str: str) -> int:
    """
    Simplify calving string.

    Parameters
    ----------
    my_str : str
        The input calving string.

    Returns
    -------
    int
        The simplified calving value.
    """
    return int(my_str.split("_")[3])


def convert_column_to_numeric(column: pd.Series) -> pd.Series:
    """
    Convert column to numeric if possible.

    Parameters
    ----------
    column : pd.Series
        The column to convert.

    Returns
    -------
    pd.Series
        The converted column, or the original column if conversion fails.
    """
    try:
        return pd.to_numeric(column, errors="raise")
    except ValueError:
        return column


def transpose_dataframe(df: pd.DataFrame, exp_id: str) -> pd.DataFrame:
    """
    Transpose dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        The input dataframe.
    exp_id : str
        The experiment id to add to the transposed dataframe.

    Returns
    -------
    pd.DataFrame
        The transposed dataframe with the experiment id.
    """
    param_names = df["pism_config_axis"]
    df = df[["pism_config"]].T
    df.columns = param_names
    df["exp_id"] = exp_id
    return df
