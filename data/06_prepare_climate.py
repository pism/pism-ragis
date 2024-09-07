# Copyright (C) 2024 Andy Aschwanden
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
Prepare ITS_LIVE.
"""
# pylint: disable=unused-import

from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Union

import cf_xarray
import numpy as np
import requests
import xarray as xr
from dask.diagnostics import ProgressBar
from tqdm import tqdm

from pism_ragis.processing import preprocess_mar


def download_file(url: str, output_path: Path) -> None:
    """
    Download a file from a URL with a progress bar.
    Parameters
    ----------
    url : str
        The URL of the file to download.
    output_path : str
        The local path where the downloaded file will be saved.
    """

    if output_path.exists():
        return
    response = requests.get(url, stream=True, timeout=10)
    total_size = int(response.headers.get("content-length", 0))
    block_size = 1024  # 1 Kilobyte
    with open(output_path, "wb") as file, tqdm(
        total=total_size, unit="iB", unit_scale=True
    ) as progress_bar:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)


def download_racmo(
    output_dir: Union[str, Path] = ".",
    max_workers: int = 4,
) -> List[Path]:
    """
    Download files in parallel.
    Parameters
    ----------
    max_workers : int, optional
        The maximum number of threads to use for downloading, by default 4.
    """
    responses = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for var in [
            "precip_monthlyS_FGRN055_BN_RACMO2.3p2_ERA5_3h_1940_FGRN055_193909_202303.nc",
            "runoff_monthlyS_FGRN055_BN_RACMO2.3p2_ERA5_3h_1940_FGRN055_193909_202303.nc",
            "smb_monthlyS_FGRN055_BN_RACMO2.3p2_ERA5_3h_1940_FGRN055_193909_202303.nc",
            "t2m_monthlyA_FGRN055_BN_RACMO2.3p2_ERA5_3h_1940_FGRN055_193909_202303.nc",
        ]:
            url = f"https://surfdrive.surf.nl/files/index.php/s/No8LoNA18eS1v72/download?path=%2FMonthly&files={var}"
            output_path = output_dir / Path(var)
            futures.append(executor.submit(download_file, url, output_path))
            responses.append(output_path)
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"An error occurred: {e}")
        return responses


def download_mar(
    base_url: str,
    start_year: int,
    end_year: int,
    output_dir: Union[str, Path] = ".",
    max_workers: int = 4,
) -> List[Path]:
    """
    Download files in parallel.
    Parameters
    ----------
    start_year : int
        The starting year of the files to download.
    end_year : int
        The ending year of the files to download.
    max_workers : int, optional
        The maximum number of threads to use for downloading, by default 4.
    """
    responses = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for year in range(start_year, end_year + 1):
            year_file = f"MARv3.14-monthly-ERA5-{year}.nc"
            url = base_url + year_file
            output_path = output_dir / Path(year_file)
            futures.append(executor.submit(download_file, url, output_path))
            responses.append(output_path)
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"An error occurred: {e}")

    return responses


hirham_url = "http://ensemblesrt3.dmi.dk/data/prudence/temp/nichan/Daily2D_GrIS/"
mar_url = "http://ftp.climato.be/fettweis/MARv3.14/Greenland/ERA5-1km-monthly/"
xr.set_options(keep_attrs=True)


if __name__ == "__main__":
    __spec__ = None

    # set up the option parser
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.description = "Prepare climate forcing."
    options = parser.parse_args()

    result_dir = Path("climate")
    result_dir.mkdir(parents=True, exist_ok=True)
    hirham_dir = result_dir / Path("hirham")
    hirham_dir.mkdir(parents=True, exist_ok=True)
    racmo_dir = result_dir / Path("racmo")
    racmo_dir.mkdir(parents=True, exist_ok=True)
    mar_dir = result_dir / Path("mar")
    mar_dir.mkdir(parents=True, exist_ok=True)

    mar_vars_dict = {
        "STcorr": {"pism_name": "ice_surface_temp", "units": "degC"},
        "T2Mcorr": {"pism_name": "air_temp", "units": "degC"},
        "RUcorr": {"pism_name": "water_input_rate", "units": "kg m-2 month-1"},
        "SMBcorr": {"pism_name": "climatic_mass_balance", "units": "kg m-2 month"},
    }
    racmo_vars_dict = {
        "t2m": {"pism_name": "ice_surface_temp", "units": "K"},
        "runoff": {"pism_name": "water_input_rate", "units": "kg m-2 month-1"},
        "precip": {"pism_name": "precipitation", "units": "kg m-2 month"},
        "smb": {"pism_name": "climatic_mass_balance", "units": "kg m-2 month"},
    }

    print("Processing RACMO")
    responses = download_racmo(output_dir=racmo_dir)
    ds = (
        xr.open_mfdataset(
            responses,
            chunks="auto",
            parallel=True,
            compat="override",
            engine="h5netcdf",
        )
        .squeeze()
        .drop_vars(["block1", "block2", "assigned"])
    )
    for k, v in racmo_vars_dict.items():
        ds[k].attrs["units"] = v["units"]
    ds = ds.rename_vars({k: v["pism_name"] for k, v in racmo_vars_dict.items()})
    nt = ds.time.size
    times = xr.date_range("1939-10-01", freq="MS", periods=nt + 1)
    ds.time_bnds.values = np.array([times[:-1], times[1:]]).T
    ds = ds.sel(time=slice("1940", "2023"))

    encoding = {}
    comp = {"zlib": True, "complevel": 2}
    encoding_compression = {
        var: comp for var in ds.data_vars if var not in ("time", "time_bounds")
    }
    encoding.update(encoding_compression)
    output_file = Path("RACMO2.3p2_ERA5_FGRN055_1940_2023.nc")
    with ProgressBar():
        ds.to_netcdf(output_file, encoding=encoding)

    print("Processing MAR")
    responses = download_mar(
        mar_url,
        1975,
        2023,
        output_dir=mar_dir,
    )
    ds = xr.open_mfdataset(
        responses,
        preprocess=preprocess_mar,
        chunks="auto",
        parallel=True,
        engine="h5netcdf",
        attrs_file=responses[0],
        decode_cf=False,
    ).squeeze()
    # Fix global attribute encoding, xr.open_mfdataset seems to cause issues with some characters
    for k, v in ds.attrs.items():
        ds.attrs[k] = v.encode("ASCII", "surrogateescape").decode("UTF-8")
    ds = ds[mar_vars_dict.keys()]
    for k, v in mar_vars_dict.items():
        ds[k].attrs["units"] = v["units"]
    ds = ds.rename_vars({k: v["pism_name"] for k, v in mar_vars_dict.items()})
    encoding = {}
    comp = {"zlib": True, "complevel": 2}
    encoding_compression = {
        var: comp for var in ds.data_vars if var not in ("time", "time_bounds")
    }
    encoding.update(encoding_compression)
    output_file = Path("MARv3.14-monthly-ERA5_1975_2023.nc")
    with ProgressBar():
        ds.to_netcdf(output_file, encoding=encoding)

    print("Processing HIRHAM")
    # responses = download_files_in_parallel_years(
    #     hirham_url, 1980, 2021, file_template="{year}.zip", output_dir=hirham_dir
    # )
