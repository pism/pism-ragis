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

# pylint: disable=consider-using-with
# import-untyped

"""
Prepare mass balance data from Mankoff et al. (2021).
"""
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from importlib.resources import files
from pathlib import Path

import cf_xarray.units  # pylint: disable=unused-import
import numpy as np
import pandas as pd
import pint_xarray  # pylint: disable=unused-import
import toml
import xarray as xr
from pyproj import Proj, Transformer
from shapely.geometry import Polygon
from shapely.ops import transform

from pism_ragis.datetools import decimal_year_to_datetime
from pism_ragis.download import download_earthaccess, download_netcdf, save_netcdf

xr.set_options(keep_attrs=True)


# Precompute the transformer
proj_wgs84 = Proj(proj="latlong", datum="WGS84")
proj_equal_area = Proj(proj="aea", lat_1=0, lat_2=90)
project = Transformer.from_crs(
    proj_wgs84.crs, proj_equal_area.crs, always_xy=True
).transform


def calculate_polygon_area(lat_0, lat_1, lon_0, lon_1):
    """
    Calculate the area of a polygon defined by latitude and longitude bounds.

    Parameters
    ----------
    lat_0 : float
        Latitude of the first bound.
    lat_1 : float
        Latitude of the second bound.
    lon_0 : float
        Longitude of the first bound.
    lon_1 : float
        Longitude of the second bound.

    Returns
    -------
    float
        Area of the polygon in square meters.
    """
    coords = (
        (lon_0, lat_0),
        (lon_1, lat_0),
        (lon_1, lat_1),
        (lon_0, lat_1),
        (lon_0, lat_0),
    )
    polygon = Polygon(coords)
    equal_area_polygon = transform(project, polygon)
    return equal_area_polygon.area


def _calculate_area(lat_0: float, lat_1: float, lon_0: float, lon_1: float) -> float:
    """
    Calculate the area of each grid cell in square meters.

    Parameters
    ----------
    lat_0 : float
        Latitude of the first bound.
    lat_1 : float
        Latitude of the second bound.
    lon_0 : float
        Longitude of the first bound.
    lon_1 : float
        Longitude of the second bound.

    Returns
    -------
    float
        Area of each grid cell in square meters.
    """
    return calculate_polygon_area(lat_0, lat_1, lon_0, lon_1)


def prepare_grace_goddard(result_dir: Path = Path("mass_balance")):
    """
    Prepare mass balance data from GRACE Goddard.

    Parameters
    ----------
    result_dir : Path, optional
        Directory to save the results, by default "mass_balance".
    """
    url = "https://earth.gsfc.nasa.gov/sites/default/files/geo/gsfc.glb_.200204_202410_rl06v2.0_obp-ice6gd_halfdegree.nc"
    fn = "grace_gsfc_greenland_mass_balance_clean.nc"
    p_fn = result_dir / fn
    ds = download_netcdf(url)
    save_netcdf(ds, p_fn)
    ds = xr.open_dataset(p_fn)
    ds = ds.sel({"lon": slice(360 - 75, 360 - 10), "lat": slice(59, 84)})
    ds["land_mask"].attrs.update({"units": ""})
    ds = ds.pint.quantify()
    lat_bounds, lon_bounds = ds["lat_bounds"], ds["lon_bounds"]
    lat_bounds_2d, lon_bounds_2d = xr.broadcast(lat_bounds, lon_bounds)
    lat_bounds_2d = lat_bounds_2d.transpose("lat", "lon", "bounds")
    lon_bounds_2d = lon_bounds_2d.transpose("lat", "lon", "bounds")

    # Assuming ds is your dataset
    lat_bounds, lon_bounds = ds["lat_bounds"], ds["lon_bounds"]
    lat_bounds_2d, lon_bounds_2d = xr.broadcast(lat_bounds, lon_bounds)
    lat_bounds_2d = lat_bounds_2d.transpose("lat", "lon", "bounds")
    lon_bounds_2d = lon_bounds_2d.transpose("lat", "lon", "bounds")

    lats_0, lats_1 = lat_bounds_2d.isel({"bounds": 0}), lat_bounds_2d.isel(
        {"bounds": 1}
    )
    lons_0, lons_1 = lon_bounds_2d.isel({"bounds": 0}), lon_bounds_2d.isel(
        {"bounds": 1}
    )

    area = xr.apply_ufunc(
        _calculate_area,
        lats_0,
        lats_1,
        lons_0,
        lons_1,
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
    )

    area.name = "area"
    area.attrs.update({"units": "m^2"})

    ds["lwe_thickness_err"] = xr.zeros_like(ds["lwe_thickness"]) + 4

    water_density = xr.DataArray(1000.0).pint.quantify("kg m^-3").pint.to("Gt m^-3")
    ds["cumulative_mass_balance"] = (
        ds["lwe_thickness"].where(ds["land_mask"]).pint.to("m")
        * area.pint.quantify()
        * water_density
    )
    days_in_interval = (
        (ds.time.diff(dim="time") / np.timedelta64(1, "s"))
        .pint.quantify("s")
        .pint.to("year")
    )
    ds["mass_balance"] = (
        ds["cumulative_mass_balance"].diff(dim="time") / days_in_interval
    )
    ds["mass_balance_err"] = (
        xr.zeros_like(ds["mass_balance"])
        + xr.DataArray(4).pint.quantify("cm yr^-1").pint.to("m yr^-1")
        * area.pint.quantify()
        * water_density
    )

    fn = "grace_gsfc_greenland_mass_balance.nc"

    p_fn = result_dir / fn
    grace_ds = ds.pint.dequantify()
    save_netcdf(grace_ds, p_fn)


def prepare_grace_tellus(result_dir: Path = Path("mass_balance")):
    """
    Prepare mass balance data from GRACE Tellus.

    Parameters
    ----------
    result_dir : Path, optional
        Directory to save the results, by default "mass_balance".
    """
    short_name = "GREENLAND_MASS_TELLUS_MASCON_CRI_TIME_SERIES_RL06.1_V3"
    results = download_earthaccess(result_dir=result_dir, short_name=short_name)
    grace_file = results[0]
    # Read the data into a pandas DataFrame
    df = pd.read_csv(
        grace_file,
        header=32,  # Skip the header lines
        sep=r"\s+",
        names=[
            "year",
            "cumulative_mass_balance",
            "mass_balance_uncertainty",
        ],
    )

    # Vectorize the function to apply it to the entire array
    vectorized_conversion = np.vectorize(decimal_year_to_datetime)

    # Print the result
    date = vectorized_conversion(df["year"])
    df["time"] = date

    ds = xr.Dataset.from_dataframe(df.set_index(df["time"]))
    ds["cumulative_mass_balance"].attrs.update({"units": "Gt"})
    ds["cumulative_mass_balance_uncertainty"] = np.sqrt(
        (ds["mass_balance_uncertainty"] ** 2).cumsum(dim="time")
    )
    ds["cumulative_mass_balance_uncertainty"].attrs.update({"units": "Gt"})

    ds = ds.pint.quantify()

    days_in_interval = (
        (ds.time.diff(dim="time") / np.timedelta64(1, "s"))
        .pint.quantify("s")
        .pint.to("year")
    )

    ds["mass_balance"] = (
        ds["cumulative_mass_balance"].diff(dim="time") / days_in_interval
    )
    fn = "grace_greenland_mass_balance.nc"
    p_fn = result_dir / fn
    grace_ds = ds.pint.dequantify()
    save_netcdf(grace_ds, p_fn)


def prepare_mankoff(
    url: str = "https://dataverse.geus.dk/api/access/datafile/:persistentId?persistentId=doi:10.22008/FK2/OHI23Z/MRSBQR",
    result_dir: Path = Path("mass_balance"),
):
    """
    Prepare mass balance data from Mankoff et al. (2021).

    Parameters
    ----------
    url : str, optional
        URL to download the mass balance data, by default "https://dataverse.geus.dk/api/access/datafile/:persistentId?persistentId=doi:10.22008/FK2/OHI23Z/MRSBQR".
    result_dir : Path, optional
        Directory to save the results, by default "mass_balance".
    """
    ragis_config_file = Path(
        str(files("pism_ragis.data").joinpath("ragis_config.toml"))
    )
    ragis_config = toml.load(ragis_config_file)

    basin_vars_dict = ragis_config["Mankoff"]["basin"]
    basin_vars = [v for v in basin_vars_dict.values() if not "uncertainty" in v]
    basin_uncertainty_vars = [v for v in basin_vars_dict.values() if "uncertainty" in v]

    gis_vars_dict = ragis_config["Mankoff"]["gis"]

    ds = download_netcdf(url)
    for v in ["MB_err", "BMB_err", "MB_ROI", "MB_ROI_err", "BMB_ROI_err"]:
        ds[v].attrs["units"] = "Gt day-1"
    ds = ds.pint.quantify()

    comp = {"zlib": True, "complevel": 2}
    encoding = {var: comp for var in ds.data_vars}

    fn = "mankoff_greenland_mass_balance_clean.nc"
    p_fn = result_dir / fn
    ds.pint.dequantify().to_netcdf(p_fn, encoding=encoding)

    gis = ds[list(gis_vars_dict.keys())]

    ds = ds.rename_vars(basin_vars_dict)[list(basin_vars_dict.values())]
    ds = ds.rename({"region": "basin"})

    gis = gis.rename_vars(gis_vars_dict)[list(gis_vars_dict.values())]
    gis = gis.expand_dims("basin")
    gis["basin"] = ["GIS"]

    ds = xr.concat([ds, gis], dim="basin")
    ds["basin"] = ds["basin"].astype("<U3")

    # The data is unevenly space in time. Until 1986, the interval is 1 year, and
    # from 1986 on, the interval is 1 day. We thus compute the amount of days in a given
    # interval.
    dt = (
        (ds.time.diff(dim="time", label="lower") / np.timedelta64(1, "s"))
        .pint.quantify("s")
        .pint.to("day")
    )
    for v in basin_vars + basin_uncertainty_vars:
        ds[f"cumulative_{v}"] = (ds[v].pint.to("Gt day^-1") * dt).cumsum(dim="time")
        ds[v] = ds[v].pint.to("Gt year^-1")
        ds[f"cumulative_{v}"] = ds[f"cumulative_{v}"].pint.to("Gt")

    # remove last time entry because it is NaN for cumulative uncertainties.
    ds = ds.isel(time=slice(0, -1))
    discharge_sign = xr.DataArray(-1).pint.quantify("1")
    ds["grounding_line_flux"] *= discharge_sign

    comp = {"zlib": True, "complevel": 2}
    encoding = {var: comp for var in ds.data_vars}

    fn = "mankoff_greenland_mass_balance.nc"
    p_fn = result_dir / fn
    mankoff_ds = ds.pint.dequantify()
    save_netcdf(mankoff_ds, p_fn)


if __name__ == "__main__":
    __spec__ = None  # type: ignore

    # set up the option parser
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.description = "Prepare Mass Balance data sets."
    options = parser.parse_args()
    p = Path("mass_balance")
    p.mkdir(parents=True, exist_ok=True)

    prepare_grace_goddard(result_dir=p)

    # prepare_mankoff(result_dir=p)

    # prepare_grace_tellus(result_dir=p)
