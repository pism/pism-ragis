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
from typing import Union
from ast import literal_eval

import requests
import xarray as xr
from tqdm import tqdm


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


def download_files_in_parallel(
    base_url: str,
    start_year: int,
    end_year: int,
    file_template: str = "MARv3.14-monthly-ERA5-${year}.nc",
    output_dir: Union[str, Path] = ".",
    max_workers: int = 4,
) -> None:
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
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for year in range(start_year, end_year + 1):
            year_file = eval(f'f"{file_template}"')
            url = base_url + year_file
            output_path = output_dir / Path(year_file)
            futures.append(executor.submit(download_file, url, output_path))
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"An error occurred: {e}")


hirham_url = "http://ensemblesrt3.dmi.dk/data/prudence/temp/nichan/Daily2D_GrIS/"
mar_url = "http://ftp.climato.be/fettweis/MARv3.14/Greenland/ERA5-1km-monthly"
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

    # Download files from 1980 to 2020 in parallel
    download_files_in_parallel(
        mar_url,
        1975,
        2023,
        file_template="MARv3.14-monthly-ERA5-${year}.nc",
        output_dir=mar_dir,
    )
    download_files_in_parallel(
        hirham_url, 1980, 2021, file_template="{year}.zip", output_dir=hirham_dir
    )
