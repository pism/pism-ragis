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
Prepare Climate.
"""
# pylint: disable=unused-import,broad-exception-caught
# mypy: ignore-errors

import time
import zipfile
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Union

import cf_xarray
import numpy as np
import requests
import xarray as xr
from cdo import Cdo
from dask.diagnostics import ProgressBar
from tqdm import tqdm

from pism_ragis.processing import preprocess_time

cdo = Cdo()
cdo.debug = True


def process_year(
    year: int, output_dir: Path, vars_dict: Dict, start_date: str = "1975-01-01"
) -> Path:
    """
    Process NetCDF files for a given year and generate a monthly averaged dataset.

    Parameters
    ----------
    year : int
        The year to process.
    output_dir : Path
        The directory where the output file will be saved.
    start_date : str, optional
        The reference start date for the time encoding, by default "1975-01-01".

    Returns
    -------
    Path
        The path to the generated NetCDF file.

    Notes
    -----
    This function reads daily NetCDF files for the specified year, processes the data,
    and generates a monthly averaged dataset. The output file is saved in the specified
    output directory with the name format 'monthly_{year}.nc'.
    """
    output_file = output_dir / Path(f"monthly_{year}.nc")
    p = output_dir / Path(str(year))
    responses = sorted(p.glob("*.nc"))
    ds = xr.open_mfdataset(
        responses, parallel=True, chunks={"time": -1}, engine="netcdf4"
    )
    ds = ds[list(vars_dict.keys()) + ["lat", "lon"]]

    for v in ["lat", "lon", "gld", "rogl"]:
        ds[v] = ds[v].swap_dims({"x": "rlon", "y": "rlat"})
        if "_CoordinateAxisType" in ds[v].attrs:
            del ds[v].attrs["_CoordinateAxisType"]

    for k, v in vars_dict.items():
        if k in ds:
            ds[k].attrs["units"] = v["units"]
            if "missing_value" in ds[k].attrs:
                del ds[k].attrs["missing_value"]
    time = xr.date_range(str(year), freq="D", periods=ds.time.size + 1)
    time_centered = time[:-1] + (time[1:] - time[:-1]) / 2
    ds = ds.assign_coords(time=time_centered)

    ds = ds.resample({"time": "MS"}).mean()
    time = xr.date_range(str(year), freq="MS", periods=ds.time.size + 1)
    time_centered = time[:-1] + (time[1:] - time[:-1]) / 2
    ds = ds.assign_coords(time=time_centered)
    ds = ds.cf.add_bounds("time")

    for v in ds.data_vars:
        if "cell_methods" in ds[v].attrs:
            del ds[v].attrs["cell_methods"]
        if "_FillValue" in ds[v].attrs:
            del ds[v].attrs["_FillValue"]
    ds["time"].encoding = {
        "units": f"hours since {start_date}",
    }
    ds["time"].attrs.update(
        {
            "axis": "T",
            "long_name": "time",
        }
    )
    ds.attrs["Conventions"] = "CF-1.8"

    encoding = {var: {"_FillValue": False} for var in ["rlat", "rlon", "lon", "lat"]}
    comp = {"zlib": True, "complevel": 2}

    encoding_compression = {
        var: comp
        for var in ds.data_vars
        if var not in ("time", "time_bounds", "time_bnds")
    }
    encoding.update(encoding_compression)

    with ProgressBar():
        ds.to_netcdf(output_file, encoding=encoding)
    _ = [f.unlink() for f in p.glob("*.nc")]
    p.rmdir()
    return output_file


def process_hirham_cdo(
    data_dir: Union[str, Path],
    output_file: Union[str, Path],
    base_url: str,
    vars_dict: Dict,
    overwrite: bool = False,
    max_workers: int = 4,
    start_year: int = 1980,
    end_year: int = 2021,
) -> None:
    """
    Process HIRHAM data and save the output to a NetCDF file.

    Parameters
    ----------
    data_dir : str
        Directory containing the input data.
    output_file : str
        Path to the output NetCDF file.

    Returns
    -------
    None
    """
    print("Processing HIRHAM")

    hirham_dir = data_dir / Path("hirham")
    hirham_dir.mkdir(parents=True, exist_ok=True)
    hirham_nc_dir = hirham_dir / Path("nc")
    hirham_nc_dir.mkdir(parents=True, exist_ok=True)
    hirham_zip_dir = hirham_dir / Path("zip")
    hirham_zip_dir.mkdir(parents=True, exist_ok=True)

    responses = download_hirham(
        base_url,
        start_year,
        end_year,
        output_dir=hirham_zip_dir,
        max_workers=max_workers,
    )

    responses = unzip_files(
        responses,
        output_dir=hirham_nc_dir,
        overwrite=overwrite,
        max_workers=max_workers,
    )

    # Initialize an empty list to store the parts of the string
    chname_parts = []

    # Iterate over the dictionary items
    for key, value in vars_dict.items():
        chname_parts.append(key)
        chname_parts.append(value["pism_name"])
    chname = ",".join(chname_parts)

    # Initialize an empty list to store the parts of the string
    setattribute_parts = []

    # Iterate over the dictionary items
    for key, value in vars_dict.items():
        setattribute_parts.append(f"""{key}@units='{value["units"]}'""")
    setattribute = ",".join(setattribute_parts)

    print("Merging daily files and calculate monthly means.")
    for year in range(start_year, end_year + 1):
        print(f"..processing {year}")
        p = hirham_nc_dir / Path(str(year))
        responses = sorted(p.glob("*.nc"))
        infiles = [str(p.absolute()) for p in responses]
        infiles = " ".join(infiles)
        ofile = hirham_nc_dir / Path(f"monthly_{year}.nc")
        outfile = str(ofile.absolute())

        start = time.time()
        cdo.setmisstodis(
            input=f"""-monmean -setreftime,1975-01-01 -settbounds,day -settaxis,"{year}-01-01" -chname,{chname} -setattribute,{setattribute} -setgrid,grids/grid_hirham.txt -selvar,{",".join(vars_dict.keys())} -mergetime """
            + infiles,
            output=outfile,
            options=f"-f nc4 -z zip_2 -P {max_workers}",
        )
        end = time.time()
        time_elapsed = end - start
        print(f"Time elapsed {time_elapsed:.0f}s")

    start = time.time()
    infiles = [
        str((hirham_nc_dir / Path(f"monthly_{year}.nc")).absolute())
        for year in range(start_year, end_year + 1)
    ]
    infiles = " ".join(infiles)
    merged_ofile = cdo.mergetime(
        input=infiles,
        options=f"-f nc4 -z zip_2 -P {max_workers}",
    )
    pre_1980 = cdo.settbounds(
        "1mon", input=" -settaxis,1975-01-01,,1mon -selyear,1981/1985 " + merged_ofile
    )
    merged = cdo.mergetime(
        input=pre_1980 + " " + merged_ofile,
        options=f"-f nc4 -z zip_2 -P {max_workers}",
    )
    ds = cdo.settunits(
        "days",
        input="-settbounds,1mon -settaxis,1975-01-01,,1mon " + merged,
        returnXDataset=True,
    )

    ds["rlat"].attrs.update({"standard_name": "grid_latitude"})
    ds["rlon"].attrs.update({"standard_name": "grid_longitude"})
    for v in ds.data_vars:
        if "cell_methods" in ds[v].attrs:
            del ds[v].attrs["cell_methods"]
        if "_FillValue" in ds[v].attrs:
            del ds[v].attrs["_FillValue"]

    ds.attrs["Conventions"] = "CF-1.8"

    encoding = {var: {"_FillValue": False} for var in ["rlat", "rlon"]}
    comp = {"zlib": True, "complevel": 2}

    encoding_compression = {
        var: comp
        for var in ds.data_vars
        if var not in ("time", "time_bounds", "time_bnds")
    }
    encoding.update(encoding_compression)
    print(f"Writing to {output_file}")
    with ProgressBar():
        ds.to_netcdf(output_file, encoding=encoding)

    end = time.time()
    time_elapsed = end - start
    print(f"Time elapsed {time_elapsed:.0f}s")


def process_hirham(
    data_dir: Union[str, Path],
    output_file: Union[str, Path],
    base_url: str,
    vars_dict: Dict,
    overwrite: bool = False,
    max_workers: int = 4,
    start_year: int = 1980,
    end_year: int = 2021,
) -> None:
    """
    Process HIRHAM data and save the output to a NetCDF file.

    Parameters
    ----------
    data_dir : str
        Directory containing the input data.
    output_file : str
        Path to the output NetCDF file.

    Returns
    -------
    None
    """
    print("Processing HIRHAM")

    start_date = "1975"

    hirham_dir = data_dir / Path("hirham")
    hirham_dir.mkdir(parents=True, exist_ok=True)
    hirham_nc_dir = hirham_dir / Path("nc")
    hirham_nc_dir.mkdir(parents=True, exist_ok=True)
    hirham_zip_dir = hirham_dir / Path("zip")
    hirham_zip_dir.mkdir(parents=True, exist_ok=True)

    responses = download_hirham(
        base_url,
        start_year,
        end_year,
        output_dir=hirham_zip_dir,
        max_workers=max_workers,
    )

    responses = unzip_files(
        responses,
        output_dir=hirham_nc_dir,
        overwrite=overwrite,
        max_workers=max_workers,
    )

    print("Merging daily files and calculate monthly means.")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        responses = []
        for year in range(start_year, end_year):
            futures.append(
                executor.submit(process_year, year, output_dir=hirham_nc_dir)
            )
        for future in as_completed(futures):
            try:
                f = future.result()
                responses.append(f)
            except Exception as e:
                print(f"An error occurred: {e}")

    infiles = [
        str((hirham_nc_dir / Path(f"monthly_{year}.nc")).absolute())
        for year in range(start_year, end_year)
    ]
    infiles = " ".join(infiles)
    print("Fill missing values")
    ds = cdo.setmisstodis(input="-mergetime " + infiles, returnXDataset=True).squeeze()
    ds = ds.rename_vars({"time_bnds": "time_bounds"})

    for k, v in vars_dict.items():
        if k in ds:
            ds[k].attrs["units"] = v["units"]
            if "missing_value" in ds[k].attrs:
                del ds[k].attrs["missing_value"]

    ds = ds.rename_vars({k: v["pism_name"] for k, v in vars_dict.items()})
    time = xr.date_range(start_year, freq="MS", periods=ds.time.size + 1)
    time_centered = time[:-1] + (time[1:] - time[:-1]) / 2

    ds = ds.assign_coords(time=time_centered)
    ds["time_bounds"].values = np.array([time[:-1], time[1:]]).T

    # Use 1981 to 1985 to match the calendar / month lengths from 75 to 79
    pre_ds = ds.sel(time=slice("1981", "1985"))
    time = xr.date_range("1975", freq="MS", periods=pre_ds.time.size + 1)
    time_centered = time[:-1] + (time[1:] - time[:-1]) / 2

    pre_ds = pre_ds.assign_coords(time=time_centered)
    pre_ds["time_bounds"].values = np.array([time[:-1], time[1:]]).T
    ds = xr.concat([pre_ds, ds], dim="time")

    ds["time"].attrs.update(
        {
            "axis": "T",
            "long_name": "time",
        }
    )
    ds.attrs["Conventions"] = "CF-1.8"

    time = xr.date_range("1975", freq="MS", periods=ds.time.size + 1)
    time_centered = time[:-1] + (time[1:] - time[:-1]) / 2

    ds = ds.assign_coords(time=time_centered)
    ds["time_bounds"].values = np.array([time[:-1], time[1:]]).T
    ds["time"].encoding = {
        "units": f"hours since {start_date}",
    }

    encoding = {var: {"_FillValue": False} for var in ["rlat", "rlon", "lon", "lat"]}
    comp = {"zlib": True, "complevel": 2}
    encoding_compression = {
        var: comp for var in ds.data_vars if var not in ("time", "time_bounds")
    }
    encoding.update(encoding_compression)
    print(f"Saving to {output_file}")
    with ProgressBar():
        ds.to_netcdf(output_file, encoding=encoding)


def process_racmo_cdo(
    data_dir: Union[str, Path],
    output_file: Union[str, Path],
    vars_dict: Dict,
    start_year: int = 1975,
    end_year: int = 2023,
    max_workers: int = 4,
) -> None:
    """
    Process RACMO data and save the output to a NetCDF file.

    Parameters
    ----------
    data_dir : str
        Directory containing the input data.
    output_file : str
        Path to the output NetCDF file.

    Returns
    -------
    None
    """
    print("Processing RACMO")

    racmo_dir = data_dir / Path("racmo")
    racmo_dir.mkdir(parents=True, exist_ok=True)
    responses = download_racmo(output_dir=racmo_dir, max_workers=max_workers)

    infiles = [str(p.absolute()) for p in responses]
    infiles = " ".join(infiles)

    # Initialize an empty list to store the parts of the string
    chname_parts = []

    # Iterate over the dictionary items
    for key, value in vars_dict.items():
        chname_parts.append(key)
        chname_parts.append(value["pism_name"])
    chname = ",".join(chname_parts)

    # Initialize an empty list to store the parts of the string
    setattribute_parts = []

    # Iterate over the dictionary items
    for key, value in vars_dict.items():
        setattribute_parts.append(f"""{key}@units='{value["units"]}'""")
    setattribute = ",".join(setattribute_parts)
    start = time.time()
    ds = cdo.settbounds(
        "1mon",
        input=f""" -chname,{chname} -setattribute,{setattribute} -setgrid,grids/grid_racmo.txt -sellevel,0 -selvar,{",".join(vars_dict.keys())} -selyear,{start_year}/{end_year} -merge """
        + infiles,
        options=f"-f nc4 -z zip_2 -P {max_workers} --reduce_dim",
        returnXDataset=True,
    )
    ds["rlat"].attrs.update({"standard_name": "grid_latitude"})
    ds["rlon"].attrs.update({"standard_name": "grid_longitude"})
    for v in ds.data_vars:
        if "cell_methods" in ds[v].attrs:
            del ds[v].attrs["cell_methods"]
        if "_FillValue" in ds[v].attrs:
            del ds[v].attrs["_FillValue"]

    ds.attrs["Conventions"] = "CF-1.8"

    encoding = {var: {"_FillValue": False} for var in ["rlat", "rlon"]}
    comp = {"zlib": True, "complevel": 2}

    encoding_compression = {
        var: comp
        for var in ds.data_vars
        if var not in ("time", "time_bounds", "time_bnds")
    }
    encoding.update(encoding_compression)
    print(f"Writing to {output_file}")
    with ProgressBar():
        ds.to_netcdf(output_file, encoding=encoding)

    end = time.time()
    time_elapsed = end - start
    print(f"Time elapsed {time_elapsed:.0f}s")


def process_racmo(
    data_dir: Union[str, Path],
    output_file: Union[str, Path],
    vars_dict: Dict,
    max_workers: int = 4,
) -> None:
    """
    Process RACMO data and save the output to a NetCDF file.

    Parameters
    ----------
    data_dir : str
        Directory containing the input data.
    output_file : str
        Path to the output NetCDF file.

    Returns
    -------
    None
    """
    print("Processing RACMO")

    racmo_dir = data_dir / Path("racmo")
    racmo_dir.mkdir(parents=True, exist_ok=True)
    responses = download_racmo(output_dir=racmo_dir, max_workers=max_workers)

    ds = (
        xr.open_mfdataset(
            responses,
            chunks={"time": -1},
            parallel=True,
            compat="override",
            engine="h5netcdf",
        )
        .squeeze()
        .drop_vars(
            [
                "block1",
                "block2",
                "assigned",
                "date_bnds",
                "dir",
                "dtg",
                "hms_bnds",
                "height",
                "time_bnds",
            ]
        )
    )

    start_date = "1939-09-01"
    time = xr.date_range(start_date, freq="MS", periods=ds.time.size + 1)
    time_centered = time[:-1] + (time[1:] - time[:-1]) / 2
    ds = ds.assign_coords(time=time_centered)
    ds["time"].encoding = {
        "units": f"hours since {start_date}",
    }
    ds = ds.cf.add_bounds("time")
    ds["time_bounds"].values = np.array([time[:-1], time[1:]]).T

    del ds["smb"].attrs["standard_name"]

    for k, v in vars_dict.items():
        ds[k].attrs["units"] = v["units"]

    ds = ds.rename_vars({k: v["pism_name"] for k, v in vars_dict.items()})

    for v in ds.data_vars:
        if "cell_methods" in ds[v].attrs:
            del ds[v].attrs["cell_methods"]
        if "_FillValue" in ds[v].attrs:
            del ds[v].attrs["_FillValue"]

    ds["time"].attrs.update(
        {
            "axis": "T",
            "long_name": "time",
        }
    )
    ds.attrs["Conventions"] = "CF-1.8"

    encoding = {var: {"_FillValue": False} for var in ["rlat", "rlon", "lon", "lat"]}
    comp = {"zlib": True, "complevel": 2}

    encoding_compression = {
        var: comp
        for var in ds.data_vars
        if var not in ("time", "time_bounds", "time_bnds")
    }
    encoding.update(encoding_compression)

    print(f"Writing to {output_file}")

    with ProgressBar():
        ds.to_netcdf(output_file, encoding=encoding)


def process_mar_cdo(
    data_dir: str,
    output_file: str,
    vars_dict: Dict,
    start_year: int = 1975,
    end_year: int = 2023,
    max_workers: int = 4,
) -> None:
    """
    Process MAR data and save the output to a NetCDF file.

    Parameters
    ----------
    data_dir : str
        Directory containing the input data.
    output_file : str
        Path to the output NetCDF file.

    Returns
    -------
    None
    """
    print("Processing MAR")

    mar_dir = data_dir / Path("mar")
    mar_dir.mkdir(parents=True, exist_ok=True)
    responses = download_mar(
        mar_url, start_year, end_year, output_dir=mar_dir, max_workers=max_workers
    )
    infiles = [str(p.absolute()) for p in responses]
    infiles = " ".join(infiles)
    outfile = str(output_file)

    # Initialize an empty list to store the parts of the string
    chname_parts = []

    # Iterate over the dictionary items
    for key, value in vars_dict.items():
        chname_parts.append(key)
        chname_parts.append(value["pism_name"])
    chname = ",".join(chname_parts)

    # Initialize an empty list to store the parts of the string
    setattribute_parts = []

    # Iterate over the dictionary items
    for key, value in vars_dict.items():
        setattribute_parts.append(f"""{key}@units='{value["units"]}'""")
    setattribute = ",".join(setattribute_parts)

    start = time.time()
    cdo.setmisstodis(
        input=f"""-settbounds,1mon -settaxis,1975-01-01,,1mon -chname,{chname} -setattribute,{setattribute} -setgrid,grids/grid_mar_v3.14.txt -selvar,{",".join(vars_dict.keys())} -mergetime """
        + infiles,
        output=outfile,
        options=f"-f nc4 -z zip_2 -P {max_workers}",
    )
    end = time.time()
    time_elapsed = end - start
    print(f"Time elapsed {time_elapsed:.0f}s")


def process_mar(
    data_dir: str,
    output_file: str,
    vars_dict: Dict,
    start_year: int = 1975,
    end_year: int = 2023,
    max_workers: int = 4,
) -> None:
    """
    Process MAR data and save the output to a NetCDF file.

    Parameters
    ----------
    data_dir : str
        Directory containing the input data.
    output_file : str
        Path to the output NetCDF file.

    Returns
    -------
    None
    """
    print("Processing MAR")

    mar_dir = data_dir / Path("mar")
    mar_dir.mkdir(parents=True, exist_ok=True)
    responses = download_mar(
        mar_url, start_year, end_year, output_dir=mar_dir, max_workers=max_workers
    )

    ds = xr.open_mfdataset(
        responses,
        preprocess=preprocess_time,
        chunks={"time": -1},
        parallel=True,
        engine="h5netcdf",
        decode_cf=False,
    ).squeeze()

    for k, v in ds.attrs.items():
        ds.attrs[k] = v.encode("ASCII", "surrogateescape").decode("UTF-8")

    start_date = "1975-01-01"

    # Fix global attribute encoding, xr.open_mfdataset seems to cause issues with some characters
    ds = ds[list(vars_dict.keys()) + ["time_bounds"]]

    for k, v in vars_dict.items():
        ds[k].attrs["units"] = v["units"]
        del ds[k].attrs["standard_name"]

    ds = ds.rename_vars({k: v["pism_name"] for k, v in vars_dict.items()})

    ds["time"].encoding = {
        "units": f"hours since {start_date}",
    }

    ds.attrs["Conventions"] = "CF-1.8"

    ds.coords["x"].attrs.update(
        {
            "axis": "X",
            "long_name": "X-coordinate in Cartesian system",
            "standard_name": "projection_x_coordinate",
        }
    )

    ds.coords["y"].attrs.update(
        {
            "axis": "Y",
            "long_name": "Y-coordinate in Cartesian system",
            "standard_name": "projection_y_coordinate",
        }
    )

    ds.coords["time"].attrs.update(
        {
            "axis": "T",
            "long_name": "time",
        }
    )

    encoding = {var: {"_FillValue": False} for var in ["x", "y"]}
    comp = {"zlib": True, "complevel": 2}

    encoding_compression = {
        var: comp for var in ds.data_vars if var not in ("time", "time_bounds")
    }

    encoding.update(encoding_compression)

    tmp_file = mar_dir / Path("tmp.nc")
    with ProgressBar():
        ds.to_netcdf(tmp_file, encoding=encoding)
    print(f"Filling missing values and writing to {output_file}")
    infile = str(tmp_file.absolute())
    outfile = str(output_file.absolute())
    cdo.setmisstodis(
        input="-setgrid,grids/grid_mar_v3.14.txt " + infile,
        output=outfile,
        options="-f nc4 -z zip_2",
    )
    tmp_file.unlink()


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
    with (
        open(output_path, "wb") as file,
        tqdm(total=total_size, unit="iB", unit_scale=True) as progress_bar,
    ):
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

    print("Downloading RACMO")
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

    print(f"Downloading MAR from {base_url}")
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


def download_hirham(
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
    print(f"Downloading HIRHAM5 from {base_url}")
    responses = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for year in range(start_year, end_year + 1):
            year_file = f"{year}.zip"
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


def unzip_files(
    files=List[Union[str, Path]],
    output_dir: Union[str, Path] = ".",
    overwrite: bool = False,
    max_workers: int = 4,
) -> List[Path]:
    """
    Unzip files in parallel.

    Parameters
    ----------
    max_workers : int, optional
        The maximum number of threads to use for downloading, by default 4.
    """
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for f in files:
            futures.append(
                executor.submit(unzip_file, f, output_dir, overwrite=overwrite)
            )
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"An error occurred: {e}", unzip_file)

    responses = list(output_dir.rglob("*.nc"))
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


hirham_url = "http://ensemblesrt3.dmi.dk/data/prudence/temp/nichan/Daily2D_GrIS/"
mar_url = "http://ftp.climato.be/fettweis/MARv3.14/Greenland/ERA5-1km-monthly/"
xr.set_options(keep_attrs=True)


if __name__ == "__main__":
    __spec__ = None

    # set up the option parser
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.description = "Prepare climate forcing."
    parser.add_argument(
        "--n_jobs", help="""Number of parallel jobs.""", type=int, default=8
    )
    options = parser.parse_args()
    max_workers = options.n_jobs
    overwrite = False

    result_dir = Path("climate")
    result_dir.mkdir(parents=True, exist_ok=True)

    mar_vars_dict: Dict[str, Dict[str, str]] = {
        "STcorr": {"pism_name": "ice_surface_temp", "units": "degC"},
        "T2Mcorr": {"pism_name": "air_temp", "units": "degC"},
        "RUcorr": {"pism_name": "water_input_rate", "units": "kg m^-2 month^-1"},
        "SMBcorr": {"pism_name": "climatic_mass_balance", "units": "kg m^-2 month^-1"},
    }
    racmo_vars_dict: Dict[str, Dict[str, str]] = {
        "t2m": {"pism_name": "ice_surface_temp", "units": "kelvin"},
        "runoff": {"pism_name": "water_input_rate", "units": "kg m^-2 month^-1"},
        "precip": {"pism_name": "precipitation", "units": "kg m^-2 month^-1"},
        "smb": {"pism_name": "climatic_mass_balance", "units": "kg m^-2 month^-1"},
    }
    hirham_vars_dict: Dict[str, Dict[str, str]] = {
        "tas": {"pism_name": "ice_surface_temp", "units": "kelvin"},
        "rogl": {"pism_name": "water_input_rate", "units": "kg m^-2 day^-1"},
        "gld": {"pism_name": "climatic_mass_balance", "units": "kg m^-2 day^-1"},
    }

    output_file = result_dir / Path("RACMO2.3p2_ERA5_FGRN055_1975_2023.nc")
    process_racmo_cdo(
        data_dir=result_dir,
        start_year=1975,
        end_year=2023,
        output_file=output_file,
        vars_dict=racmo_vars_dict,
        max_workers=max_workers,
    )

    output_file = result_dir / Path("MARv3.14-monthly-ERA5_1975_2023.nc")
    process_mar_cdo(
        data_dir=result_dir,
        vars_dict=mar_vars_dict,
        start_year=1975,
        end_year=2023,
        output_file=output_file,
        max_workers=max_workers,
    )

    output_file = result_dir / Path("HIRHAM5-monthly-ERA5_1975_2021.nc")
    process_hirham_cdo(
        data_dir=result_dir,
        vars_dict=hirham_vars_dict,
        start_year=1980,
        end_year=2021,
        output_file=output_file,
        base_url=hirham_url,
        max_workers=max_workers,
    )