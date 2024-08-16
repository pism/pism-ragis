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
from typing import List, Union

import joblib
import numpy as np
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
