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

import tarfile
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path
from urllib.request import urlopen

import earthaccess
import numpy as np
import requests
import xarray as xr
from dask.diagnostics import ProgressBar
from tqdm.auto import tqdm


def unzip_files(
    files=list[str | Path],
    output_dir: str | Path = ".",
    overwrite: bool = False,
    max_workers: int = 4,
) -> list[Path]:
    """
    Unzip files in parallel.

    Parameters
    ----------
    files : List[Union[str, Path]]
        List of file paths to unzip.
    output_dir : Union[str, Path], optional
        The directory where the unzipped files will be saved, by default ".".
    overwrite : bool, optional
        Whether to overwrite existing files, by default False.
    max_workers : int, optional
        The maximum number of threads to use for unzipping, by default 4.

    Returns
    -------
    List[Path]
        List of paths to the unzipped files.
    """
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for f in files:
            futures.append(
                executor.submit(unzip_file, f, str(output_dir), overwrite=overwrite)
            )
        for future in as_completed(futures):
            try:
                future.result()
            except (IOError, ValueError) as e:
                print(f"An error occurred: {e}", unzip_file)

    responses = list(Path(output_dir).rglob("*.nc"))
    return responses


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


def save_netcdf(
    ds: xr.Dataset,
    output_filename: str | Path = "GRE_G0240_1985_2018_IDW_EXP_1.nc",
    comp={"zlib": True, "complevel": 2},
):
    """
    Save the xarray dataset to a NetCDF file with specified compression.

    Parameters
    ----------
    ds : xarray.Dataset
        The dataset to be saved.
    output_filename : Union[str, Path], optional
        The output filename for the NetCDF file, by default "GRE_G0240_1985_2018_IDW_EXP_1.nc".
    comp : dict, optional
        Compression settings for the NetCDF file, by default {"zlib": True, "complevel": 2}.
    """
    encoding = {
        var: comp for var in ds.data_vars if np.issubdtype(ds[var].dtype, np.number)
    }
    with ProgressBar():
        ds.to_netcdf(output_filename, encoding=encoding)


def download_archive(url: str) -> tarfile.TarFile | zipfile.ZipFile:
    """
    Download an archive file from a URL and return it as a tarfile or ZipFile object.

    Parameters
    ----------
    url : str
        The URL of the archive file to download. The file can be either a .tar.gz or a .zip file.

    Returns
    -------
    Union[tarfile.TarFile, ZipFile]
        The downloaded archive file as a tarfile.TarFile object if the file is a .tar.gz,
        or as a ZipFile object if the file is a .zip.
    """
    with urlopen(url) as req:
        total_size = int(req.info().get("Content-Length").strip())
        buffer = BytesIO()
        for chunk in tqdm(
            iter(lambda: req.read(1024), b""), total=total_size // 1024, unit="KB"
        ):
            buffer.write(chunk)
        buffer.seek(0)

    if url.endswith("tar.gz"):
        return tarfile.open(fileobj=buffer, mode="r|gz")
    else:
        return zipfile.ZipFile(buffer)


def download_earthaccess(
    filter_str: str | None = None, result_dir: Path | str = ".", **kwargs
) -> list:
    """
    Download datasets via Earthaccess.

    Parameters
    ----------
    filter_str : str, optional
        A string to filter the search results. Default is None.
    result_dir : Union[Path, str], optional
        The directory where the downloaded files will be saved. Default is ".".
    **kwargs : dict
        Additional keyword arguments to pass to the Earthaccess search function.

    Returns
    -------
    List
        A list of paths to the downloaded files.
    """
    p = Path(result_dir)
    p.mkdir(parents=True, exist_ok=True)

    earthaccess.login()
    results = earthaccess.search_data(**kwargs)
    if filter_str is not None:
        results = [
            granule
            for granule in results
            if filter_str
            in granule["umm"]["DataGranule"]["Identifiers"][0]["Identifier"]
        ]
    earthaccess.get_s3_credentials(results=results)
    return earthaccess.download(results, p)


def download_netcdf(
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
    >>> dataset = download_dataset()
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
