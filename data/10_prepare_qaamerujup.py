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

# pylint: disable=unused-import

"""
Generate local domains.
"""

from pathlib import Path

import geopandas as gp
import numpy as np
import rioxarray
import xarray as xr

from pism_ragis.domain import create_domain, get_bounds

# 150, 300, 450, 600, 900, 1200, 1500, 1800, 2400, 3000, 3600, and 4500m
multipliers = [1, 2, 4, 6, 8, 10, 12, 20]
max_mult = multipliers[-1]

# buffer in m
buffer = 3e3

buffer = 3e3

crs = "EPSG:3413"
basins = gp.read_file("grids/domain_qaamerujup.gpkg").to_crs(crs)
ds = xr.open_dataset("dem/BedMachineGreenland-v5.nc")

ds = ds.rio.set_spatial_dims(x_dim="x", y_dim="y")
ds.rio.write_crs(crs, inplace=True)

for m_id, basin in basins.iterrows():
    name = basin["Name"]
    if name is not None:
        print(name)
        minx, miny, maxx, maxy = basin.geometry.buffer(buffer).bounds
        ll = ds.sel({"x": minx, "y": miny}, method="nearest")
        ur = ds.sel({"x": maxx, "y": maxy}, method="nearest")
        tmp_ds = ds.sel({"x": slice(ll["x"], ur["x"]), "y": slice(ur["y"], ll["y"])})
        x_bnds, y_bnds = get_bounds(tmp_ds)
        sub_ds = ds.sel({"x": slice(*x_bnds), "y": slice(*y_bnds[::-1])})
        grid = create_domain(x_bnds, y_bnds)
        grid.attrs.update({"domain": name})
        name_str = name.replace(" ", "_")
        grid.to_netcdf(Path("grids") / Path(f"{m_id}_{name_str}.nc"))
