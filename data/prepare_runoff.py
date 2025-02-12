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
# pylint: disable=unused-import,broad-exception-caught,too-many-positional-arguments
# mypy: ignore-errors

import time
import zipfile
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Union

import cf_xarray  # pylint: disable=unused-import
import geopandas as gp
import numpy as np
import pint_xarray  # pylint: disable=unused-import
import requests
import rioxarray as rxr
import xarray as xr
from cdo import Cdo
from dask.diagnostics import ProgressBar
from dask.distributed import Client, progress
from pyproj import CRS
from tqdm import tqdm

cdo = Cdo()
cdo.debug = True


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


def process_year(
    year: int, output_dir: Path, vars_dict: Dict, start_date: str = "1975-01-01"
) -> Path:
    """
    Prepare and process NetCDF files for a given year and generate a monthly averaged dataset.

    Parameters
    ----------
    year : int
        The year to process.
    output_dir : Path
        The directory where the output file will be saved.
    vars_dict : Dict
        Dictionary of variables to process with their attributes.
    start_date : str, optional
        The reference start date for the time encoding, by default "1975-01-01".

    Returns
    -------
    Path
        The path to the generated NetCDF file.
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


def process_hirham_cdo_daily(
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
    Prepare and process HIRHAM data and save the output to a NetCDF file.

    Parameters
    ----------
    data_dir : Union[str, Path]
        Directory containing the input data.
    output_file : Union[str, Path]
        Path to the output NetCDF file.
    base_url : str
        Base URL for downloading HIRHAM data.
    vars_dict : Dict
        Dictionary of variables to process with their attributes.
    overwrite : bool, optional
        Whether to overwrite existing files, by default False.
    max_workers : int, optional
        Maximum number of parallel workers, by default 4.
    start_year : int, optional
        Starting year for processing, by default 1980.
    end_year : int, optional
        Ending year for processing, by default 2021.
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

    print("Merging daily files.")
    for year in range(start_year, end_year + 1):
        print(f"..processing {year}")
        p = hirham_nc_dir / Path(str(year))
        responses = sorted(p.glob("*.nc"))
        infiles = [str(p.absolute()) for p in responses]
        infiles = " ".join(infiles)
        ofile = hirham_nc_dir / Path(f"daily_{year}.nc")
        outfile = str(ofile.absolute())

        start = time.time()
        cdo.setreftime(
            f"""{start_year}-01-01""",
            input=f"""-settbounds,day -settaxis,{start_year}-01-01 -chname,{chname} -setattribute,{setattribute} -setgrid,grids/grid_hirham.txt -selvar,{",".join(vars_dict.keys())} -mergetime """
            + infiles,
            output=outfile,
            options=f"-f nc4 -z zip_2 -P {max_workers}",
        )
        end = time.time()
        time_elapsed = end - start
        print(f"Time elapsed {time_elapsed:.0f}s")

    start = time.time()
    infiles = [
        str((hirham_nc_dir / Path(f"daily_{year}.nc")).absolute())
        for year in range(start_year, end_year + 1)
    ]
    infiles = " ".join(infiles)
    merged_ofile = cdo.mergetime(
        input=infiles,
        options=f"-f nc4 -z zip_2 -P {max_workers}",
    )
    ds = cdo.settunits(
        "days",
        input=f"-settbounds,1day -settaxis,{start_year}-01-01,,1day " + merged_ofile,
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
    comp = {"zlib": True, "complevel": 2, "_FillValue": None}

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


def download_file(url: str, output_path: Path) -> None:
    """
    Download a file from a URL with a progress bar.

    Parameters
    ----------
    url : str
        The URL of the file to download.
    output_path : Path
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


def download_hirham(
    base_url: str,
    start_year: int,
    end_year: int,
    output_dir: Union[str, Path] = ".",
    max_workers: int = 4,
) -> List[Path]:
    """
    Download HIRHAM files in parallel.

    Parameters
    ----------
    base_url : str
        The base URL for downloading HIRHAM data.
    start_year : int
        The starting year of the files to download.
    end_year : int
        The ending year of the files to download.
    output_dir : Union[str, Path], optional
        The directory where the downloaded files will be saved, by default ".".
    max_workers : int, optional
        The maximum number of threads to use for downloading, by default 4.

    Returns
    -------
    List[Path]
        List of paths to the downloaded files.
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


def compute_basin(
    ds: xr.Dataset, name: str = "basin", dim: list = ["x", "y"]
) -> xr.Dataset:
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
    ds = ds.sum(dim=dim).expand_dims("basin", axis=-1)
    ds["basin"] = [name]
    return ds.compute()


def extract_basins(ds: xr.Dataset, basins: gp.GeoDataFrame) -> None:
    """
    Extract the runoff for each basin and compute the total runoff for each basin.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset containing the runoff data.
    basins : gp.GeoDataFrame
        A GeoDataFrame containing the basin geometries.
    """
    # Extract the attributes
    grid_north_pole_longitude = ds["rotated_pole"].attrs["grid_north_pole_longitude"]
    grid_north_pole_latitude = ds["rotated_pole"].attrs["grid_north_pole_latitude"]
    north_pole_grid_longitude = ds["rotated_pole"].attrs["north_pole_grid_longitude"]

    # Define the CRS using pyproj
    rotated_crs = CRS.from_proj4(
        f"+proj=ob_tran +o_proj=longlat +o_lon_p={north_pole_grid_longitude} +o_lat_p={grid_north_pole_latitude}  +lon_0={grid_north_pole_longitude + 180} +datum=WGS84"
    )
    polar_stereo_crs = "EPSG:3413"

    del ds["time"].attrs["bounds"]
    ds = ds.drop_vars(["time_bounds", "time_bnds", "timestamp"], errors="ignore")
    ds = ds[["water_input_rate"]]
    ds.rio.set_spatial_dims(x_dim="rlon", y_dim="rlat")
    ds.rio.write_crs(rotated_crs, inplace=True)

    basins = basins.to_crs(rotated_crs)
    basins_polar_stereo = basins.to_crs(polar_stereo_crs)

    client = Client()
    print(f"Open client in browser: {client.dashboard_link}")

    basins_ds_scattered = client.scatter(
        [ds.rio.clip([basin.geometry]) for _, basin in basins.iterrows()]
    )
    basin_names = [basin["SUBREGION1"] for _, basin in basins.iterrows()]

    futures = client.map(
        compute_basin, basins_ds_scattered, basin_names, dim=["rlon", "rlat"]
    )

    progress(futures)
    result = client.gather(futures)

    basin_sums = xr.concat(result, dim="basin")

    # Replace "SUBREGION1" with the attribute name that contains the basin names.
    # Calculate the area of each basin in km^2 in North Polar Stereographic projection
    # and then convert to Gt/year.
    basin_area = {v.SUBREGION1: v.area for k, v in basins_polar_stereo.iterrows()}
    basin_area_da = xr.DataArray(
        data=list(basin_area.values()),
        dims="basin",
        coords={"basin": basin_sums.coords["basin"]},
        attrs={"units": "km^2"},
    ).pint.quantify()

    for v in basin_sums.data_vars:
        basin_sums[v].attrs.update({"units": "kg m^-2 day^-1"})
        basin_sums[v] = basin_sums[v].pint.quantify()
        basin_sums[v] = (basin_sums[v] * basin_area_da).pint.to("Gt year^-1")

    basin_sums.to_netcdf("sums.nc")


hirham_url = "http://ensemblesrt3.dmi.dk/data/prudence/temp/nichan/Daily2D_GrIS/"
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

    hirham_vars_dict: Dict[str, Dict[str, str]] = {
        "tas": {"pism_name": "ice_surface_temp", "units": "kelvin"},
        "rogl": {"pism_name": "water_input_rate", "units": "kg m^-2 day^-1"},
    }

    start_year, end_year = 2011, 2011
    output_file = result_dir / Path(f"HIRHAM5-daily-runoff_{start_year}.nc")
    process_hirham_cdo_daily(
        data_dir=result_dir,
        vars_dict=hirham_vars_dict,
        start_year=start_year,
        end_year=end_year,
        output_file=output_file,
        base_url=hirham_url,
        max_workers=max_workers,
    )

    ds = xr.open_dataset(output_file)
    basins = gp.read_file(
        "/Users/andy/base/pism-ragis/data/basins/Greenland_Basins_PS_v1.4.2_clean.shp"
    )
    extract_basins(ds, basins)

hirham_url = "http://ensemblesrt3.dmi.dk/data/prudence/temp/nichan/Daily2D_GrIS/"
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

    hirham_vars_dict: Dict[str, Dict[str, str]] = {
        "tas": {"pism_name": "ice_surface_temp", "units": "kelvin"},
        "rogl": {"pism_name": "water_input_rate", "units": "kg m^-2 day^-1"},
    }

    start_year, end_year = 2011, 2011
    output_file = result_dir / Path(f"HIRHAM5-daily-runoff_{start_year}.nc")
    process_hirham_cdo_daily(
        data_dir=result_dir,
        vars_dict=hirham_vars_dict,
        start_year=start_year,
        end_year=end_year,
        output_file=output_file,
        base_url=hirham_url,
        max_workers=max_workers,
    )

    ds = xr.open_dataset(output_file)
    basins = gp.read_file(
        "/Users/andy/base/pism-ragis/data/basins/Greenland_Basins_PS_v1.4.2_clean.shp"
    )
    extract_basins(ds, basins)
hirham_url = "http://ensemblesrt3.dmi.dk/data/prudence/temp/nichan/Daily2D_GrIS/"
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

    hirham_vars_dict: Dict[str, Dict[str, str]] = {
        "tas": {"pism_name": "ice_surface_temp", "units": "kelvin"},
        "rogl": {"pism_name": "water_input_rate", "units": "kg m^-2 day^-1"},
    }

    start_year, end_year = 2011, 2011
    output_file = result_dir / Path(f"HIRHAM5-daily-runoff_{start_year}.nc")
    process_hirham_cdo_daily(
        data_dir=result_dir,
        vars_dict=hirham_vars_dict,
        start_year=start_year,
        end_year=end_year,
        output_file=output_file,
        base_url=hirham_url,
        max_workers=max_workers,
    )

    ds = xr.open_dataset(output_file)
    basins = gp.read_file(
        "/Users/andy/base/pism-ragis/data/basins/Greenland_Basins_PS_v1.4.2_clean.shp"
    )
    extract_basins(ds, basins)
