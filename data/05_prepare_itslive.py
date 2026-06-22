# Copyright (C) 2024-2026 Andy Aschwanden
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

Streams the per-year ITS_LIVE v2.1 COG mosaics for an RGI o1 region (default
``05`` — Greenland) directly from the public S3 bucket, materializes one
NetCDF per year, and produces an inverse-distance-weighted time-mean field.

The COG path mirrors the fast streaming pattern used by
``pism_terra.glacier.observations.get_itslive_velocities_by_region_code``
(no earthaccess / NSIDC auth required).
"""

# pylint: disable=unused-import

import warnings
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from pathlib import Path
from typing import Iterable

import dask.array as da
import numpy as np
import rioxarray as rxr
import xarray as xr
from dask.diagnostics import ProgressBar

from pism_ragis.download import save_netcdf

xr.set_options(keep_attrs=True)
warnings.filterwarnings("ignore", category=UserWarning)

# Per-component CF metadata, matching the static-mosaic helper in pism-terra.
_COMPONENT_ATTRS = {
    "v": {"units": "m year^-1", "long_name": "ice speed"},
    "vx": {"units": "m year^-1", "long_name": "x component of ice velocity"},
    "vy": {"units": "m year^-1", "long_name": "y component of ice velocity"},
    "v_error": {"units": "m year^-1", "long_name": "ice speed error"},
    "vx_error": {"units": "m year^-1", "long_name": "x component error"},
    "vy_error": {"units": "m year^-1", "long_name": "y component error"},
    "count": {"units": "1", "long_name": "image-pair count"},
    "landice": {"units": "1", "long_name": "land ice mask (1=ice)"},
}

_ANNUAL_URL = (
    "https://its-live-data.s3.amazonaws.com/velocity_mosaic/v2.1/annual/cog/"
    "ITS_LIVE_velocity_120m_RGI{region}A_{year:04d}_V02.1_{var}.tif"
)
_STATIC_URL = (
    "https://its-live-data.s3.amazonaws.com/velocity_mosaic/v2.1/static/cog/"
    "ITS_LIVE_velocity_120m_RGI{region}A_0000_V02.1_{var}.tif"
)


def _open_cog_stack(urls: Iterable[str], variables: Iterable[str]) -> xr.Dataset:
    """
    Stream a set of single-band ITS_LIVE COGs into one xarray Dataset.

    Parameters
    ----------
    urls : iterable of str
        HTTPS URLs of the COGs, one per variable.
    variables : iterable of str
        Variable names corresponding (in order) to ``urls``.

    Returns
    -------
    xarray.Dataset
        Merged dataset with one variable per URL, chunked for lazy access.
    """
    dss = []
    for var, url in zip(variables, urls):
        da_ = (
            rxr.open_rasterio(url, parse_coordinates=True, chunks={"x": 1024, "y": 1024}, masked=True)
            .isel(band=0)
            .drop_vars("band")
        )
        da_.name = var
        # Drop the junky per-band attrs rioxarray surfaces from the COG header.
        for k in ("scale_factor", "add_offset", "AREA_OR_POINT", "_FillValue"):
            da_.attrs.pop(k, None)
        da_.attrs.update(_COMPONENT_ATTRS.get(var, {}))
        dss.append(da_)
    return xr.merge(dss, compat="no_conflicts")


def get_annual_mosaic(year: int, region: str, components: Iterable[str]) -> xr.Dataset:
    """Return one year of ITS_LIVE annual COGs with a ``time`` dim attached."""
    urls = [_ANNUAL_URL.format(region=region, year=year, var=c) for c in components]
    ds = _open_cog_stack(urls, components)
    return ds.expand_dims(time=[np.datetime64(f"{year}-01-01", "ns")])


def get_static_mosaic(region: str, components: Iterable[str]) -> xr.Dataset:
    """Return the static (0000 composite) ITS_LIVE COGs."""
    urls = [_STATIC_URL.format(region=region, var=c) for c in components]
    return _open_cog_stack(urls, components)


def idw_weights(distance: xr.DataArray, power: float = 1.0) -> xr.DataArray:
    """
    Calculate inverse distance weighting (IDW) weights.

    Parameters
    ----------
    distance : xarray.DataArray
        The array of distances.
    power : float, optional
        The power parameter for IDW, by default 1.0.

    Returns
    -------
    xarray.DataArray
        The calculated IDW weights.
    """
    return 1.0 / (distance + 1e-12) ** power


if __name__ == "__main__":
    __spec__ = None  # type: ignore

    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.description = "Prepare ITS_LIVE v2.1 annual mosaics for Greenland (RGI 05)."
    parser.add_argument("--region", default="05", help="RGI o1 region code (two digits).")
    parser.add_argument("--start-year", type=int, default=1984)
    parser.add_argument("--end-year", type=int, default=2024)
    parser.add_argument("--power", type=float, default=1.0, help="IDW power.")
    parser.add_argument("--result-dir", type=Path, default=Path("itslive"))
    options = parser.parse_args()

    region = options.region
    result_dir = options.result_dir
    result_dir.mkdir(parents=True, exist_ok=True)

    print(f"Preparing ITS_LIVE v2.1 for RGI{region}A, {options.start_year}..{options.end_year}")

    # Annual COGs publish 7 layers; the landice mask only exists on the static
    # composite, so we pull it separately and use it to mask the time series.
    annual_vars = ["v", "vx", "vy", "v_error", "vx_error", "vy_error", "count"]
    static_vars = ["landice"]
    years = list(range(options.start_year, options.end_year + 1))

    output_files: list[Path] = []
    for year in years:
        ofile = result_dir / f"ITS_LIVE_RGI{region}A_{year}.nc"
        if ofile.exists():
            print(f"Skipping existing {ofile}")
        else:
            print(f"Streaming {year} -> {ofile}")
            ds = get_annual_mosaic(year, region, annual_vars)
            with ProgressBar():
                save_netcdf(ds, ofile)
            del ds
        output_files.append(ofile)

    landice_file = result_dir / f"ITS_LIVE_RGI{region}A_landice.nc"
    if not landice_file.exists():
        print(f"Streaming static landice mask -> {landice_file}")
        with ProgressBar():
            save_netcdf(get_static_mosaic(region, static_vars), landice_file)

    print("Combining annual files and applying inverse-distance weighting")
    ds = xr.open_mfdataset(output_files, parallel=False, chunks={"time": -1})
    landice = xr.open_dataset(landice_file)["landice"]
    ds = ds.where(landice > 0)

    nt = ds.time.size
    dt = xr.DataArray(da.arange(nt, chunks=-1), dims=("time",))
    speed = ds["v"]
    # Distance metric in the original script: the position along time, masked
    # to where speed is finite. Keeps the existing IDW behavior.
    distance = np.isfinite(speed) * dt.broadcast_like(speed)
    weights = idw_weights(distance, power=options.power)
    idw_ofile = result_dir / (
        f"ITS_LIVE_RGI{region}A_{options.start_year}_{options.end_year}_IDW_EXP_{options.power}.nc"
    )
    print(f"Inverse-distance weighting (power={options.power}) -> {idw_ofile}")
    weighted_mean = ds.weighted(weights).mean(dim="time")
    with ProgressBar():
        save_netcdf(weighted_mean, idw_ofile)
