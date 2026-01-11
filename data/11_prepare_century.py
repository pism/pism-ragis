# Copyright (C) 2025 Andy Aschwanden
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
Prepare shelfmassflux (Aschwanden et al, 2019).
"""
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from pathlib import Path

import cf_xarray.units  # pylint: disable=unused-import
import numpy as np
import pint_xarray  # pylint: disable=unused-import
import xarray as xr
from pyproj import Proj, Transformer

xr.set_options(keep_attrs=False)


if __name__ == "__main__":
    __spec__ = None  # type: ignore

    # set up the option parser
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.description = "Prepare constant ocean data set."

    options = parser.parse_args()
    p = Path("ocean")
    p.mkdir(parents=True, exist_ok=True)

    bm = xr.open_dataset("dem//BedMachineGreenland-v5.nc").thin({"x": 4, "y": 4})
    shelfbmassflux = xr.zeros_like(bm["bed"])

    bmelt_0 = 228.0
    bmelt_1 = 10.0
    lat_0 = 69.0
    lat_1 = 80.0

    ice_density = 910.0

    bmelt_0 *= ice_density
    bmelt_1 *= ice_density

    # bmelt = a*y + b
    a = (bmelt_1 - bmelt_0) / (lat_1 - lat_0)
    b = bmelt_0 - a * lat_0

    X, Y = np.meshgrid(bm.x, bm.y)
    crs = "EPSG:3413"
    proj_map = Proj(crs)
    wgs84 = Proj("EPSG:4326")
    tf = Transformer.from_proj(proj_map, wgs84, always_xy=True)

    Lon, Lat = tf.transform(X, Y)

    bmelt = a * Lat + b
    bmelt[Lat < lat_0] = a * lat_0 + b
    bmelt[Lat > lat_1] = a * lat_1 + b
    shelfbmassflux[:] = bmelt
    shelfbmassflux.attrs = {"units": "kg m^-2 yr^-1"}
    shelfbmassflux.name = "shelfbmassflux"
    shelfbtemp = xr.zeros_like(bm["bed"])
    shelfbtemp.attrs = {"units": "celsisus"}
    shelfbtemp.name = "shelfbtemp"

    lat_0 = 74.0
    lat_1 = 76.0
    tct_0 = 400.0
    tct_1 = 50.0
    a_tct = (tct_1 - tct_0) / (lat_1 - lat_0)
    b_tct = tct_0 - a_tct * lat_0
    tct = a_tct * Lat + b_tct
    tct[Lat < lat_0] = a_tct * lat_0 + b_tct
    tct[Lat > lat_1] = a_tct * lat_1 + b_tct
    thickness_calving_threshold = xr.zeros_like(bm["bed"])
    thickness_calving_threshold[:] = tct
    thickness_calving_threshold.name = "thickness_calving_threshold"
    thickness_calving_threshold.attrs = {"units": "m"}

    ds = xr.merge([shelfbmassflux, shelfbtemp, thickness_calving_threshold])
    t = xr.date_range("1900-01-01,", "2100-01-01", freq="YS")
    ds = ds.expand_dims({"time": [t[0]]})
    ds["time_bounds"] = [t[0], t[-1]]
    ds["time"].attrs["bounds"] = "time_bounds"
    ds["time"].encoding["units"] = "hours since 1900-01-01 00:00:00"
    ds["time"].encoding["calendar"] = "standard"
    ds = ds.rio.set_spatial_dims(x_dim="x", y_dim="y")
    ds.rio.write_crs(crs, inplace=True)
    ds.to_netcdf(p / Path("as19_latitudinal_forcing.nc"))
