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
Prepare GSFC GRACE time series.
"""
import time
from pathlib import Path
from typing import Union

import cf_xarray.units  # pylint: disable=unused-import
import earthaccess
import geopandas as gp
import numpy as np
import pint_xarray  # pylint: disable=unused-import
import xarray as xr
from dask.distributed import Client

from pism_ragis.processing import calculate_area, compute_basin

gp.options.io_engine = "pyogrio"
xr.set_options(keep_attrs=True)

# s3://podaac-ops-cumulus-protected/TELLUS_GRAC-GRFO_MASCON_CRI_GRID_RL06.1_V3/GRCTellus.JPL.200204_202402.GLO.RL06.1M.MSCNv03CRI.nc


def download_grace(
    short_name: str = "TELLUS_GRAC-GRFO_MASCON_CRI_GRID_RL06.1_V3",
    result_dir: Union[Path, str] = "grace",
):
    """
    Download a GRACE dataset from the specified URL and return it as an xarray Dataset.
    """
    p = Path(result_dir)
    p.mkdir(parents=True, exist_ok=True)

    earthaccess.login()
    result = earthaccess.search_data(short_name=short_name)
    f = earthaccess.download(result, p)
    return xr.open_dataset(f[0])


if __name__ == "__main__":
    __spec__ = None

    crs = "EPSG:4326"

    p = Path("grace")
    p.mkdir(parents=True, exist_ok=True)

    # grace = download_grace(result_dir=p)

    water_density = xr.DataArray(1000.0).pint.quantify("kg m-3")

    grace = xr.open_dataset(
        "grace/GRCTellus.JPL.200204_202402.GLO.RL06.1M.MSCNv03CRI.nc"
    )
    grace["land_mask"].attrs["units"] = "1"
    # grid_area = xr.open_dataset("grace/area.nc")
    # grace = xr.merge([grace, grid_area])

    grace = grace.pint.quantify()

    basins = gp.read_file("mouginot/Greenland_Basins_PS_v1.4.2_w_shelves.gpkg").to_crs(
        crs
    )

    print("Processing GRACE.")
    grace.coords["lon"] = (grace.coords["lon"] + 180) % 360 - 180
    grace = grace.sortby(grace.lon).drop_vars(
        ["lon_bounds", "lat_bounds", "time_bounds"], errors="ignore"
    )

    buffer = 2
    x_min, y_min, x_max, y_max = basins.dissolve().bounds.values[0]
    grace = grace.sel(
        lon=slice(x_min - buffer, x_max + buffer),
        lat=slice(y_min - buffer, y_max + buffer),
    )
    # Calculate the area of each grid cell
    area = calculate_area(grace["lat"].values, grace["lon"].values)

    # Create a DataArray for the area
    area_da = xr.DataArray(
        area,
        coords=[grace["lat"][:-1], grace["lon"][:-1]],
        dims=["lat", "lon"],
        attrs={"units": "m2"},
        name="my_cell_area",
    ).pint.quantify()

    grace[["lwe_thickness", "uncertainty"]] = grace[
        ["lwe_thickness", "uncertainty"]
    ].where(grace["land_mask"])

    # area: m2, lwe in cm
    grace["cumulative_mass_balance"] = (
        grace["lwe_thickness"] * area_da * water_density
    ).pint.to("Gt")
    grace["cumulative_mass_balance_uncertainty"] = (
        grace["uncertainty"] * area_da * water_density
    ).pint.to("Gt")

    grace = xr.merge([grace, area_da])

    grace["mass_balance"] = (
        grace["cumulative_mass_balance"].diff(dim="time")
        / (grace["time"].diff(dim="time") / np.timedelta64(1, "s")).pint.quantify("s")
    ).pint.to("Gt month-1")

    grace = grace.pint.dequantify()
    grace.to_netcdf("grace/grace.nc")

    grace["cumulative_mass_balance_uncertainty"] = (
        grace["cumulative_mass_balance_uncertainty"] ** 2
    )

    grace.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)
    grace.rio.write_crs(crs, inplace=True)

    basin_file = p / Path("jpl_grace_mouginot_basins.nc")
    if basin_file.exists():
        basin_file.unlink()

    print("Extracting basins.")
    with Client() as client:
        print(f"Open client in browser: {client.dashboard_link}")

        start = time.time()

        basins_ds_scattered = client.scatter(
            [grace.rio.clip([basin.geometry]) for _, basin in basins.iterrows()]
        )
        basin_names = [basin["SUBREGION1"] for _, basin in basins.iterrows()]
        print(basin_names)
        futures = client.map(
            compute_basin, basins_ds_scattered, basin_names, dim=["lat", "lon"]
        )
        basin_sums = xr.concat(client.gather(futures), dim="basin")
        if "time_bounds" in grace.data_vars:
            basin_sums["time_bounds"] = grace["time_bounds"]
        basin_sums["cumulative_mass_balance"] -= basin_sums[
            "cumulative_mass_balance"
        ].isel(time=0)
        basin_sums["cumulative_mass_balance_uncertainty"] = np.sqrt(
            basin_sums["cumulative_mass_balance_uncertainty"].cumsum(dim="time")
        )

        basin_sums.to_netcdf(basin_file)

        end = time.time()
        time_elapsed = end - start
        print(f"Time elapsed {time_elapsed:.0f}s")
