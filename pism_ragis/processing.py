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
Module for data processing.
"""

import contextlib
import datetime
import os
import pathlib
import re
import shutil
import zipfile
from collections import OrderedDict
from collections.abc import Callable, Hashable, Mapping
from pathlib import Path
from typing import Any

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
            area[i, j] = (R**2) * np.abs(np.sin(lat_rad[i + 1]) - np.sin(lat_rad[i])) * np.abs(dlon[j])

    return area


def preprocess_time(
    ds,
    regexp: str = "ERA5-(.+?).nc",
    freq: str = "MS",
    periods: int | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    drop_vars: list[str] | None = None,
    drop_dims: list[str] = ["nv4"],
) -> xr.Dataset:
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
    periods : int or None, optional
        The number of periods in the time range, by default None.
    start_date : str or None, optional
        The start date for the time range, by default None.
    end_date : str or None, optional
        The end date for the time range, by default None.
    drop_vars : list[str] or None, optional
        A list of variable names to be dropped from the dataset, by default None.
    drop_dims : list[str], optional
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

    time_bounds = xr.DataArray(np.vstack([time[:-1], time[1:]]).T, dims=["time", "bounds"])
    # Add bounds to the dataset
    ds = ds.assign_coords(time_bounds=time_bounds)

    return ds.drop_vars(drop_vars, errors="ignore").drop_dims(drop_dims, errors="ignore")


def preprocess_nc(
    ds,
    regexp: str = "id_(.+?)_",
    dim: str = "exp_id",
    drop_vars: list[str] | None = None,
    drop_dims: list[str] = ["nv4"],
) -> xr.Dataset:
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
    drop_vars : list[str] | None, optional
        A list of variable names to be dropped from the dataset, by default None.
    drop_dims : list[str], optional
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
    m_id: str | int
    try:
        m_id = int(m_id_re.group(1))
    except:
        m_id = str(m_id_re.group(1))
    ds[dim] = [m_id]

    return ds.drop_vars(drop_vars, errors="ignore").drop_dims(drop_dims, errors="ignore")


def preprocess_config(
    ds,
    regexp: str = "id_(.+?)_",
    dim: str = "exp_id",
    drop_vars: list[str] | None = None,
    drop_dims: list[str] = ["nv4"],
) -> xr.Dataset:
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
    drop_vars : list[str]| None, optional
        A list of variable names to be dropped from the dataset, by default None.
    drop_dims : list[str], optional
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

    if dim not in ds.dims:
        m_id_re = re.search(regexp, ds.encoding["source"])
        ds = ds.expand_dims(dim)
        assert m_id_re is not None
        m_id: str | int
        try:
            m_id = int(m_id_re.group(1))
        except:
            m_id = str(m_id_re.group(1))
        ds[dim] = [m_id]

    p_config = ds["pism_config"]

    # List of suffixes to exclude
    suffixes_to_exclude = ["_doc", "_type", "_units", "_option", "_choices"]

    # Filter the dictionary
    config = {k: v for k, v in p_config.attrs.items() if not any(k.endswith(suffix) for suffix in suffixes_to_exclude)}
    if "geometry.front_retreat.prescribed.file" not in config.keys():
        config["geometry.front_retreat.prescribed.file"] = "false"

    config_sorted = OrderedDict(sorted(config.items()))

    pc_keys = np.array(list(config_sorted.keys()))
    pc_vals = np.array(list(config_sorted.values()))

    pism_config = xr.DataArray(
        pc_vals.reshape(-1, 1),
        dims=["pism_config_axis", dim],
        coords={"pism_config_axis": pc_keys, dim: [m_id]},
        name="pism_config",
    )
    ds = xr.merge(
        [
            ds.drop_vars(["pism_config", "run_stats"], errors="ignore").drop_dims(
                ["pism_config_axis", "run_stats_axis"], errors="ignore"
            ),
            pism_config,
        ]
    )
    return ds.drop_vars(drop_vars, errors="ignore").drop_dims(drop_dims, errors="ignore")


def preprocess_scalar_nc(
    ds,
    regexp: str = "id_(.+?)_",
    dim: str = "exp_id",
    ensemble_dim: str = "ensemble_id",
    basin_dim: str = "basin",
    ensemble_id: str = "RAGIS",
    basin: str = "GIS",
    drop_vars: list[str] | None = None,
    drop_dims: list[str] = ["nv4"],
) -> xr.Dataset:
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
    drop_vars : list[str] | None, optional
        A list of variable names to be dropped from the dataset, by default None.
    drop_dims : list[str], optional
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
    m_id: str | int
    try:
        m_id = int(m_id_re.group(1))
    except:
        m_id = str(m_id_re.group(1))
    ds[dim] = [m_id]

    return ds.drop_vars(drop_vars, errors="ignore").drop_dims(drop_dims, errors="ignore")


def compute_basin(ds: xr.Dataset, name: str = "basin", dim: list = ["x", "y"]) -> xr.Dataset:
    """
    Compute the sum of the dataset over the 'x' and 'y' dimensions and add a new dimension 'basin'.

    Parameters
    ----------
    ds : xr.Dataset
        The input dataset.
    name : str
        The name to assign to the new 'basin' dimension.
    dim : List
        The dimensions to sum over.

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
    return ds


@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """
    Context manager to patch joblib to report into tqdm progress bar given as argument.

    Parameters
    ----------
    tqdm_object : tqdm.tqdm
        The tqdm progress bar object to use for reporting progress.
    """
    # ...existing code...

    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        """
        TQDM Callback.

        This callback updates the tqdm progress bar with the batch size.
        """

        def __call__(self, *args, **kwargs):
            """
            Call the TQDM callback.

            Parameters
            ----------
            *args : tuple
                Positional arguments.
            **kwargs : dict
                Keyword arguments.

            Returns
            -------
            Any
                The result of the super class __call__ method.
            """
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


def to_decimal_year(date: datetime.datetime) -> float:
    """
    Convert datetime date to decimal year.

    Parameters
    ----------
    date : datetime.datetime
        The date to convert.

    Returns
    -------
    float
        The decimal year representation of the date.
    """
    year = date.year
    start_of_this_year = datetime.datetime(year=year, month=1, day=1)
    start_of_next_year = datetime.datetime(year=year + 1, month=1, day=1)
    year_elapsed = (date - start_of_this_year).total_seconds()
    year_duration = (start_of_next_year - start_of_this_year).total_seconds()
    fraction = year_elapsed / year_duration

    return date.year + fraction


def check_file(infile: str | pathlib.Path, norm_year: None | float = None) -> bool:
    """
    Check netCDF file.

    Parameters
    ----------
    infile : str | pathlib.Path
        The path to the netCDF file.
    norm_year : None | float, optional
        The normalization year, by default None.

    Returns
    -------
    bool
        True if the file is valid, False otherwise.
    """
    with xr.open_dataset(infile) as ds:
        is_ok: bool = False
        if "time" in ds.indexes:
            datetimeindex = ds.indexes["time"]
            years = np.array([to_decimal_year(x.to_pydatetime()) for x in datetimeindex])
            monotonically_increasing = np.all(years.reshape(1, -1)[:, 1:] >= years.reshape(1, -1)[:, :-1], axis=1)[0]
            if norm_year:
                if (years[-1] >= norm_year) and monotonically_increasing:
                    is_ok = True
            else:
                print(f"{infile} time-series too short or not monotonically-increasing.")
        return is_ok


def check_paleo_file(infile: str | pathlib.Path, norm_year: None | float = None) -> bool:
    """
    Check netCDF file.

    Parameters
    ----------
    infile : str | pathlib.Path
        The path to the netCDF file.
    norm_year : None, float, optional
        The normalization year, by default None.

    Returns
    -------
    bool
        True if the file is valid, False otherwise.
    """
    with xr.open_dataset(infile) as ds:
        is_ok: bool = False
        if "time" in ds.indexes:
            datetimeindex = ds.indexes["time"]
            years = datetimeindex.year
            monotonically_increasing = np.all(years.reshape(1, -1)[:, 1:] >= years.reshape(1, -1)[:, :-1], axis=1)[0]
            if norm_year:
                if (years[-1] >= norm_year) and monotonically_increasing:
                    is_ok = True
            else:
                print(f"{infile} time-series too short or not monotonically-increasing.")
        return is_ok


def copy_file(infile: str | pathlib.Path, outdir: str | pathlib.Path) -> None:
    """
    Copy infile to outdir.

    Parameters
    ----------
    infile : str | pathlib.Path
        The input file path.
    outdir : str, pathlib.Path
        The output directory path.
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
        nonnumeric_vars = [v for v in self._obj.data_vars if not np.issubdtype(self._obj[v].dtype, np.number)]

        return self._obj.drop_vars(nonnumeric_vars, errors=errors)


@timeit
def load_ensemble(
    filenames: list[Path | str],
    parallel: bool = True,
    engine: str = "h5netcdf",
    preprocess: Callable | None = None,
) -> xr.Dataset:
    """
    Load an ensemble of NetCDF files into an xarray Dataset.

    Parameters
    ----------
    filenames : list[Path | str]
        A list of file paths or strings representing the NetCDF files to be loaded.
    parallel : bool, optional
        Whether to load the files in parallel using Dask. Default is True.
    engine : str, optional
        The engine to use for loading the NetCDF files. Default is "h5netcdf".
    preprocess : Callable or None, optional
        A preprocessing function to apply to each dataset before concatenation. Default is None.

    Returns
    -------
    xr.Dataset
        The loaded xarray Dataset containing the ensemble data.
    """

    time_coder = xr.coders.CFDatetimeCoder()

    print("Loading ensemble files... ", end="", flush=True)
    ds = xr.open_mfdataset(
        filenames,
        parallel=parallel,
        preprocess=preprocess,
        engine=engine,
        decode_times=time_coder,
        decode_timedelta=True,
    ).drop_vars(["spatial_ref", "mapping"], errors="ignore")
    print("Done.")
    return ds


def normalize_cumulative_variables(
    ds: xr.Dataset,
    variables: str | list[str] | None = None,
    reference_date: str = "1992-01-01",
) -> xr.Dataset:
    """
    Normalize cumulative variables in an xarray Dataset by subtracting their values at a reference year.

    Parameters
    ----------
    ds : xr.Dataset
        The xarray Dataset containing the cumulative variables to be normalized.
    variables : str or list of str
        The name(s) of the cumulative variables to be normalized.
    reference_date : str, optional
        The reference date to use for normalization. Default is "1992-01-01".

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
    >>> normalize_cumulative_variables(data, "cumulative_var", reference_date="1992-01-01")
    <xarray.Dataset>
    Dimensions:         (time: 6)
    Coordinates:
      * time            (time) datetime64[ns] 1990-12-31 1991-12-31 ... 1995-12-31
    Data variables:
        cumulative_var  (time) int64 0 10 20 30 40 50
    """

    if variables is not None:
        ds[variables] -= ds[variables].sel(time=reference_date, method="nearest")
    else:
        pass
    return ds


def standardize_variable_names(ds: xr.Dataset, name_dict: Mapping[Any, Hashable] | None) -> xr.Dataset:
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
    >>> standardize_variable_names(ds, name_dict)
    <xarray.Dataset>
    Dimensions:      (x: 3)
    Dimensions without coordinates: x
    Data variables:
        temperature   (x) int64 1 2 3
        precipitation (x) int64 4 5 6
    """
    return ds.rename_vars(name_dict)


def select_experiments(df: pd.DataFrame, ids_to_select: list[int]) -> pd.DataFrame:
    """
    Select rows from a DataFrame based on a list of experiment IDs, including duplicates.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing experiment data.
    ids_to_select : list[int]
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
    repeated_indices = selected_rows.index.repeat([ids_to_select.count(id) for id in selected_rows["exp_id"]])

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
    climate_mapping: dict[str, str] = {
        "MAR": "MAR",
        "RACMO": "RACMO",
        "HIRHAM": "HIRHAM",
    }

    for key in climate_mapping:  # pylint: disable=consider-using-dict-items
        if key in my_str:
            return climate_mapping[key]

    return "HIRHAM"


def simplify_retreat(my_str: str) -> str:
    """
    Simplify retreat string.

    This function simplifies the input retreat string by returning a standardized
    retreat model name based on the presence of specific substrings.

    Parameters
    ----------
    my_str : str
        The input retreat string.

    Returns
    -------
    str
        The standardized retreat model name based on the input string.

    Examples
    --------
    >>> simplify_retreat("false")
    'Free'
    >>> simplify_retreat("true")
    'Prescribed'
    """

    if my_str in ("false", ""):
        short_str = "Free"
    else:
        short_str = "Prescribed"

    return short_str


def simplify_ocean(my_str: str) -> str:
    """
    Simplify ocean string.

    Parameters
    ----------
    my_str : str
        The input ocean string.

    Returns
    -------
    int
        The simplified ocean value.
    """
    gcms: dict[str, str] = {
        "ACCESS1-3_rcp85": "ACCESS1-3",
        "CNRM-CM6_ssp126": "CNRM-CM6",
        "CNRM-ESM2_ssp585": "CNRM-ESM2",
        "CSIRO-Mk3.6_rcp85": "CSIRO-Mk3.6",
        "HadGEM2-ES_rcp85": "HadGEM2-ES",
        "IPSL-CM5-MR_rcp85": "IPSL-CM5",
        "MIROC-ESM-CHEM_rcp26": "MIROC-ESM",
        "NorESM1-M_rcp85": "NorESM1-M",
        "UKESM1-CM6_ssp585": "UKESM1-CM6",
    }

    gcm = "_".join(my_str.split("_")[1:3])
    return gcms[gcm]


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


def convert_column_to_category(column: pd.Series) -> pd.Series:
    """
    Convert column to category if possible.

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
        return column.astype("category")
    except ValueError:
        return column


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


def filter_config(da: xr.DataArray, params: list[str]) -> xr.DataArray:
    """
    Filter the configuration parameters from the dataset.

    This function selects the specified configuration parameters from a DataArray
    and returns them as a DataArray.

    Parameters
    ----------
    da : xr.DataArray
        The input data array containing the configuration parameters.
    params : List[str]
        A list of configuration parameter names to be selected.

    Returns
    -------
    xr.DataArray
        The selected configuration parameters as a DataArray.

    Examples
    --------
    >>> ds = xr.Dataset({'pism_config': (('pism_config_axis',), [1, 2, 3])},
                        coords={'pism_config_axis': ['param1', 'param2', 'param3']})
    >>> filter_config(ds, ['param1', 'param3'])
    <xarray.DataArray 'pism_config' (pism_config_axis: 2)>
    array([1, 3])
    Coordinates:
      * pism_config_axis  (pism_config_axis) <U6 'param1' 'param3'
    """
    config = da.sel(pism_config_axis=params)
    return config


@timeit
def config_to_dataframe(
    config: xr.DataArray,
    ensemble: str | None = None,
    aux_dim_name: str = "aux_id",
    config_axis_name: str = "pism_config_axis",
) -> pd.DataFrame:
    """
    Convert an xarray DataArray configuration into a pandas DataFrame.

    This function pivots a DataArray containing configuration values (e.g., for
    ensemble modeling) into a wide-format DataFrame. It supports multi-dimensional
    sample spaces and handles non-unique coordinate combinations by introducing an
    auxiliary identifier dimension to ensure uniqueness during reshaping.

    Parameters
    ----------
    config : xr.DataArray
        An xarray DataArray with a configuration axis (e.g., 'pism_config_axis') and
        one or more sampling dimensions (e.g., 'exp_id', 'basin', 'filtered_by').
    ensemble : str or None, optional
        Optional label for the ensemble to include as a column in the output DataFrame.
    aux_dim_name : str, default "aux_id"
        The name of the auxiliary dimension added to create a unique index for reshaping.
    config_axis_name : str, default "pism_config_axis"
        The name of the dimension corresponding to configuration parameters (used as
        columns in the output DataFrame).

    Returns
    -------
    pd.DataFrame
        A DataFrame with a row for each sample (including coordinates as columns),
        and configuration parameters as additional columns.

    Raises
    ------
    ValueError
        If `config_axis_name` is not a dimension of the DataArray.

    Examples
    --------
    >>> config_to_dataframe(config_da)
    >>> config_to_dataframe(config_da, ensemble="Prior", aux_dim_name="uid", config_axis_name="params")
    """

    if config_axis_name not in config.dims:
        raise ValueError(f"'{config_axis_name}' must be a dimension of the input DataArray.")

    # Identify dimensions that define samples (excluding the config axis)
    sample_dims = [dim for dim in config.dims if dim != config_axis_name]
    n_samples = np.prod([config.sizes[dim] for dim in sample_dims])

    # Create a unique auxiliary dimension
    aux_dim = xr.DataArray(np.arange(n_samples).reshape(*[config.sizes[dim] for dim in sample_dims]), dims=sample_dims)
    config = config.copy()
    config.coords[aux_dim_name] = aux_dim

    # Convert to DataFrame
    df = config.to_dataframe(name="pism_config").reset_index()

    # Pivot to wide format
    wide = df.pivot(index=sample_dims + [aux_dim_name], columns=config_axis_name, values="pism_config").reset_index()

    # Restore any original coordinates that are not 1D
    for name, coord in config.coords.items():
        if name in sample_dims or name in [aux_dim_name, config_axis_name]:
            continue
        if set(coord.dims).issubset(sample_dims):
            try:
                values = coord.values.reshape(n_samples)
            except ValueError:
                values = xr.DataArray(coord).broadcast_like(config).values.reshape(n_samples)
            wide[name] = values

    if ensemble is not None:
        wide["ensemble"] = ensemble

    return wide


@timeit
def filter_by_retreat_method(ds: xr.Dataset, retreat_method: str, compute: bool = False) -> xr.Dataset:
    """
    Filter retreat experiments based on the retreat method.

    This function filters the dataset to include only the experiments that match the specified
    retreat method. The retreat method can be "free", "prescribed", or "all".

    Parameters
    ----------
    ds : xr.Dataset
        The input dataset containing the retreat experiments.
    retreat_method : {"Free", "Prescribed", "All"}
        The retreat method to filter by. "Free" selects experiments with no prescribed retreat,
        "Prescribed" selects experiments with a prescribed retreat, and "All" selects all experiments.
    compute : bool, optional
        Whether to compute the dataset immediately, by default False.

    Returns
    -------
    xr.Dataset
        The filtered dataset containing only the experiments that match the specified retreat method.

    Examples
    --------
    >>> ds = xr.Dataset({'pism_config': (('exp_id', 'pism_config_axis'), [[1, 2], [3, 4]])},
    ...                 coords={'exp_id': [0, 1], 'pism_config_axis': ['param1', 'geometry.front_retreat.prescribed.file']})
    >>> filter_by_retreat_method(ds, 'Free')
    <xarray.Dataset>
    Dimensions:         (exp_id: 1, pism_config_axis: 2)
    Coordinates:
      * exp_id          (exp_id) int64 0
      * pism_config_axis (pism_config_axis) <U36 'param1' 'geometry.front_retreat.prescribed.file'
    Data variables:
        pism_config     (exp_id, pism_config_axis) int64 1 2
    """
    # Select the relevant pism_config_axis
    retreat = ds.sel(pism_config_axis="geometry.front_retreat.prescribed.file")
    if compute:
        retreat = retreat.compute()

    if retreat_method == "Free":
        retreat_exp_ids = retreat.where(retreat["pism_config"] == "false", drop=True).exp_id.values
    elif retreat_method == "Prescribed":
        retreat_exp_ids = retreat.where(retreat["pism_config"] != "false", drop=True).exp_id.values
    else:
        retreat_exp_ids = ds.exp_id

    # Select the Dataset with the filtered exp_ids
    ds = ds.sel(exp_id=retreat_exp_ids)

    return ds


def sort_columns(df: pd.DataFrame, sorted_columns: list[str]) -> pd.DataFrame:
    """
    Sort columns of a DataFrame.

    This function sorts the columns of a DataFrame such that the columns specified in
    `sorted_columns` appear in the specified order, while all other columns appear before
    the sorted columns in their original order.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame to be sorted.
    sorted_columns : List[str]
        A list of column names to be sorted.

    Returns
    -------
    pd.DataFrame
        The DataFrame with columns sorted as specified.
    """
    # Identify columns that are not in the list
    other_columns = [col for col in df.columns if col not in sorted_columns]

    # Concatenate other columns with the sorted columns
    new_column_order = other_columns + sorted_columns

    # Reindex the DataFrame
    return df.reindex(columns=new_column_order)


def add_prefix_coord(sensitivity_indices: xr.Dataset, parameter_groups: dict[str, str]) -> xr.Dataset:
    """
    Add prefix coordinates to an xarray Dataset.

    This function extracts the prefix from each coordinate value in the 'pism_config_axis'
    and adds it as a new coordinate. It also maps the prefixes to their corresponding
    sensitivity indices groups.

    Parameters
    ----------
    sensitivity_indices : xr.Dataset
        The input dataset containing sensitivity indices.
    parameter_groups : dict[str, str]
        A dictionary mapping parameter names to their corresponding groups.

    Returns
    -------
    xr.Dataset
        The dataset with added prefix coordinates and sensitivity indices groups.
    """
    prefixes = [name.split(".")[0] for name in sensitivity_indices.pism_config_axis.values]

    sensitivity_indices = sensitivity_indices.assign_coords(prefix=("pism_config_axis", prefixes))
    si_prefixes = [parameter_groups[name] for name in sensitivity_indices.prefix.values]

    sensitivity_indices = sensitivity_indices.assign_coords(sensitivity_indices_group=("pism_config_axis", si_prefixes))
    return sensitivity_indices


def convert_category_to_integer(
    df: pd.DataFrame,
    params: list[str] = [
        "surface.given.file",
        "ocean.th.file",
        "geometry.front_retreat.prescribed.file",
    ],
) -> pd.DataFrame:
    """
    Prepare the input DataFrame by converting columns to numeric and mapping unique values to integers.

    This function processes the input DataFrame by converting specified columns to numeric values,
    dropping specified columns, and mapping unique values in the specified parameters to integers.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame to be processed.
    params : list[str], optional
        A list of column names to be processed. Unique values in these columns will be mapped to integers.
        By default, the list includes:
        ["surface.given.file", "ocean.th.file", "geometry.front_retreat.prescribed.file"].

    Returns
    -------
    pd.DataFrame
        The processed DataFrame with specified columns converted to numeric and unique values mapped to integers.

    Examples
    --------
    >>> df = pd.DataFrame({
    ...     "surface.given.file": ["file1", "file2", "file1"],
    ...     "ocean.th.file": ["fileA", "fileB", "fileA"],
    ...     "geometry.front_retreat.prescribed.file": ["fileM", "fileN", "fileM"],
    ...     "ensemble": [1, 2, 3],
    ...     "exp_id": [101, 102, 103]
    ... })
    >>> convert_category_to_integer(df)
       surface.given.file  ocean.th.file  geometry.front_retreat.prescribed.file
    0                   0              0                                      0
    1                   1              1                                      1
    2                   0              0                                      0
    """
    df = df.apply(convert_column_to_numeric).drop(columns=["ensemble", "exp_id"], errors="ignore")

    for param in params:
        m_dict: dict[str, int] = {v: k for k, v in enumerate(df[param].unique())}
        df[param] = df[param].map(m_dict)

    return df


@timeit
def prepare_simulations(
    filenames: list[Path | str],
    config: dict[str, Any],
    reference_date: str,
    parallel: bool = True,
    preprocess: Callable | None = None,
    engine: str = "h5netcdf",
) -> xr.Dataset:
    """
    Prepare simulations by loading and processing ensemble datasets.

    This function loads ensemble datasets from the specified filenames, processes them
    according to the provided configuration, and returns the processed dataset. The
    processing steps include sorting, converting byte strings to strings, dropping NaNs,
    standardizing variable names, calculating cumulative variables, and normalizing
    cumulative variables.

    Parameters
    ----------
    filenames : list[[Path | str]
        A list of file paths to the ensemble datasets.
    config : dict[str, Any]
        A dictionary containing configuration settings for processing the datasets.
    reference_date : str
        The reference date for normalizing cumulative variables.
    parallel : bool, optional
        Whether to load the datasets in parallel, by default True.
    preprocess : Callable, optional
        Pass a preprocess function to xr.open_mfdataset.
    engine : str, optional
        The engine to use for loading the datasets, by default "h5netcdf".

    Returns
    -------
    xr.Dataset
        The processed xarray dataset.

    Examples
    --------
    >>> filenames = ["file1.nc", "file2.nc"]
    >>> config = {
    ...     "PISM Spatial": {...},
    ...     "Cumulative Variables": {
    ...         "cumulative_grounding_line_flux": "cumulative_gl_flux",
    ...         "cumulative_smb_flux": "cumulative_smb_flux"
    ...     },
    ...     "Flux Variables": {
    ...         "grounding_line_flux": "gl_flux",
    ...         "smb_flux": "smb_flux"
    ...     }
    ... }
    >>> reference_date = "2000-01-01"
    >>> ds = prepare_simulations(filenames, config, reference_date)
    """
    ds = load_ensemble(filenames, preprocess=preprocess, parallel=parallel, engine=engine).dropna(dim="exp_id")

    ds = xr.apply_ufunc(np.vectorize(convert_bstrings_to_str), ds, dask="parallelized")

    ds = standardize_variable_names(ds, config["PISM Spatial"])
    ds[config["Cumulative Variables"]["cumulative_grounding_line_flux"]] = ds[
        config["Flux Variables"]["grounding_line_flux"]
    ].cumsum() / len(ds.time)
    ds[config["Cumulative Variables"]["cumulative_smb_flux"]] = ds[config["Flux Variables"]["smb_flux"]].cumsum() / len(
        ds.time
    )
    ds = normalize_cumulative_variables(
        ds,
        variables=list(config["Cumulative Variables"].values()),
        reference_date=reference_date,
    )
    return ds


@timeit
def prepare_observations(
    url: Path | str,
    config: dict[str, Any],
    reference_date: str,
    engine: str = "h5netcdf",
) -> xr.Dataset:
    """
    Prepare observation datasets by normalizing cumulative variables.

    This function loads observation datasets from the specified URLs, sorts them by basin,
    normalizes the cumulative variables, and returns the processed datasets.

    Parameters
    ----------
    url : Path or str
        The URL or path to the basin observation dataset.
    config : dict[str, Any]
        A dictionary containing configuration settings for processing the datasets.
    reference_date : str
        The reference date for normalizing cumulative variables.
    engine : str, optional
        The engine to use for loading the datasets, by default "h5netcdf".

    Returns
    -------
    xr.Dataset
        A observation datasets.

    Examples
    --------
    >>> config = {
    ...     "Cumulative Variables": {"cumulative_mass_balance": "mass_balance"},
    ...     "Cumulative Uncertainty Variables": {"cumulative_mass_balance_uncertainty": "mass_balance_uncertainty"}
    ... }
    >>> prepare_observations("basin.nc", config, "2000-01-1")
    <xarray.Dataset>
    """
    time_coder = xr.coders.CFDatetimeCoder()

    ds = xr.open_dataset(url, engine=engine, chunks=-1, decode_times=time_coder, decode_timedelta=True)
    ds = ds.sortby("basin")

    cumulative_vars = config["Cumulative Variables"]
    cumulative_uncertainty_vars = config["Cumulative Uncertainty Variables"]

    ds = normalize_cumulative_variables(
        ds,
        list(cumulative_vars.values()) + list(cumulative_uncertainty_vars.values()),
        reference_date,
    )

    return ds


def convert_bstrings_to_str(element: Any) -> Any:
    """
    Convert byte strings to regular strings.

    Parameters
    ----------
    element : Any
        The element to be checked and potentially converted. If the element is a byte string,
        it will be converted to a regular string. Otherwise, the element will be returned as is.

    Returns
    -------
    Any
        The converted element if it was a byte string, otherwise the original element.
    """
    if isinstance(element, bytes):
        return element.decode("utf-8")
    return element


def prepare_liafr(
    obs_ds: xr.Dataset,
    sim_ds: xr.Dataset,
    obs_mean_var,
    obs_std_var,
    sim_var: str,
) -> tuple[xr.Dataset, xr.Dataset]:
    """
    Prepare land ice area fraction retreat data for analysis.

    Parameters
    ----------
    obs_ds : xr.Dataset
        The observed dataset.
    sim_ds : xr.Dataset
        The simulated dataset.
    obs_mean_var : str
        The variable name for the observed mean data.
    obs_std_var : str
        The variable name for the observed standard deviation data.
    sim_var : str
        The variable name for the simulated data.

    Returns
    -------
    tuple[xr.Dataset, xr.Dataset]
        A tuple containing the prepared observed and simulated datasets.
    """
    thk_mask = (sim_ds["thk"] > 10).persist()
    s_liafr = thk_mask.resample(time="YE").mean()
    s_liafr.name = sim_var
    o_liafr = obs_ds[obs_mean_var].interp_like(s_liafr, method="nearest")
    s_liafr_b = s_liafr
    o_liafr_b = o_liafr
    sim = s_liafr_b.to_dataset()

    o_liafr_b_uncertainty = xr.ones_like(o_liafr_b)
    o_liafr_b_uncertainty.name = obs_std_var
    obs = xr.merge([o_liafr_b, o_liafr_b_uncertainty])

    return obs, sim


def prepare_dhdt(
    obs_ds: xr.Dataset,
    sim_ds: xr.Dataset,
    obs_mean_var,
    obs_std_var,
    sim_var: str,
    coarsen: dict | None = None,
) -> tuple[xr.Dataset, xr.Dataset]:
    """
    Prepare dh/dt data for analysis.

    Parameters
    ----------
    obs_ds : xr.Dataset
        The observed dataset.
    sim_ds : xr.Dataset
        The simulated dataset.
    obs_mean_var : str
        The variable name for the observed mean data.
    obs_std_var : str
        The variable name for the observed standard deviation data.
    sim_var : str
        The variable name for the simulated data.
    coarsen : dict or None, optional
        Dictionary specifying the dimensions and factors for coarsening the simulated data, by default None.

    Returns
    -------
    tuple[xr.Dataset, xr.Dataset]
        A tuple containing the prepared observed and simulated datasets.
    """
    sim_ds = sim_ds.pint.quantify()
    sim_ds[sim_var] = sim_ds["dHdt"] * 1000.0 / 910.0
    sim_ds[sim_var] = sim_ds[sim_var].pint.to("m year^-1")
    sim_ds = sim_ds.pint.dequantify()

    sim_retreat_resampled = (
        sim_ds.drop_vars(["pism_config", "run_stats"])
        .drop_dims(["pism_config_axis", "run_stats_axis"])
        .resample({"time": "YS"})
        .mean(dim="time")
    )

    obs = obs_ds.pint.quantify()
    for v in [obs_mean_var, obs_std_var]:
        obs[v] = obs[v].pint.to("m year^-1")
    obs = obs.pint.dequantify()

    if coarsen is not None:
        sim = sim_retreat_resampled.coarsen(coarsen).mean()
    else:
        sim = sim_retreat_resampled
    obs = obs.interp_like(sim)

    obs_dhdt = obs[obs_mean_var]
    obs_mask = obs_dhdt.isnull()
    obs_mask = obs_mask.any(dim="time")

    sim = sim.where(~obs_mask)[[sim_var]]

    return obs, sim


def prepare_v(
    obs_ds: xr.Dataset,
    sim_ds: xr.Dataset,
    obs_mean_var: str | None = None,
    obs_std_var: str | None = None,
    coarsen: dict | None = None,
) -> tuple[xr.Dataset, xr.Dataset]:
    """
    Prepare dh/dt data for analysis.

    Parameters
    ----------
    obs_ds : xr.Dataset
        The observed dataset.
    sim_ds : xr.Dataset
        The simulated dataset.
    obs_mean_var : str
        The variable name for the observed mean data.
    obs_std_var : str
        The variable name for the observed standard deviation data.
    coarsen : dict or None, optional
        Dictionary specifying the dimensions and factors for coarsening the simulated data, by default None.

    Returns
    -------
    tuple[xr.Dataset, xr.Dataset]
        A tuple containing the prepared observed and simulated datasets.
    """

    sim_retreat_resampled = (
        sim_ds.drop_vars(["pism_config", "run_stats"])
        .drop_dims(["pism_config_axis", "run_stats_axis"])
        .resample({"time": "YS"})
        .mean(dim="time")
    )
    obs = obs_ds.pint.quantify()
    for v in [obs_mean_var, obs_std_var]:
        obs[v] = obs[v].pint.to("m year^-1")
    obs = obs.pint.dequantify()
    obs = obs.where(obs["ice"])
    if coarsen is not None:
        sim = sim_retreat_resampled.coarsen(coarsen).mean()
    else:
        sim = sim_retreat_resampled
    obs = obs.interp_like(sim)

    return obs, sim


def prepare_grace(
    obs_ds: xr.Dataset,
    sim_ds: xr.Dataset,
    sim_var: str | None = None,
) -> tuple[xr.Dataset, xr.Dataset]:
    """
    Prepare dh/dt data for analysis.

    Parameters
    ----------
    obs_ds : xr.Dataset
        The observed dataset.
    sim_ds : xr.Dataset
        The simulated dataset.
    sim_var : str
        The variable name for the simulated data.

    Returns
    -------
    tuple[xr.Dataset, xr.Dataset]
        A tuple containing the prepared observed and simulated datasets.
    """

    obs = obs_ds.where(obs_ds["land_mask"])
    obs = obs.expand_dims({"basin": ["GIS"]})

    obs = obs.rio.set_spatial_dims(x_dim="lon", y_dim="lat")
    obs.rio.write_crs("EPSG:4326", inplace=True)

    sim = (
        sim_ds[[sim_var]]
        .rio.set_spatial_dims(x_dim="x", y_dim="y")
        .drop_vars(["time_bounds", "timestamp"], errors="ignore")
    )
    sim.rio.write_crs("EPSG:3413", inplace=True)

    print("Reprojecting simulated data to EPSG:4326")
    sims = []
    for s in tqdm(
        sim["exp_id"].values,
        total=len(sim["exp_id"].values),
        desc="Reprojecting simulated data",
    ):
        sim_s = sim.sel(exp_id=s)
        sim_s = sim_s.rio.reproject_match(obs)
        sim_s = sim_s.rename({"x": "lon", "y": "lat"})
        sims.append(sim_s)
    sim = xr.concat(sims, dim="exp_id")

    sim = sim.where(obs["land_mask"])

    return obs, sim
