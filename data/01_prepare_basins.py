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

"""
Prepare Greenland basins.
"""


from pathlib import Path
from typing import Dict

import geopandas as gp
import pandas as pd

from pism_ragis.download import download_archive

imbie = {
    "name": "GRE_Basins_IMBIE2_v1.3",
    "url": "http://imbie.org/wp-content/uploads/2016/09/",
    "suffix": "zip",
    "path": "imbie",
}
mouginot = {
    "name": "Greenland_Basins_PS_v1.4.2",
    "url": "http://www.cpom.ucl.ac.uk/imbie/basins/mouginot_2019/",
    "suffix": "tar.gz",
    "path": "mouginot",
}
crs = "EPSG:3413"


def prepare_basin(basin_dict: Dict, col: str = "SUBREGION1"):
    """
    Prepare basin.

    Parameters
    ----------
    basin_dict : Dict
        Dictionary containing basin information such as name, url, suffix, and path.
    col : str, optional
        Column name to use, default is "SUBREGION1".
    """
    name = basin_dict["name"]
    print(f"Preparing {name}")
    url = basin_dict["url"] + basin_dict["name"] + "." + basin_dict["suffix"]

    archive = download_archive(url)

    path = Path(basin_dict["path"])
    path.mkdir(parents=True, exist_ok=True)
    archive.extractall(path=path)
    p = path / f"{name}.shp"
    basin_gp = gp.read_file(p).to_crs(crs)
    basin_gp["geometry"] = basin_gp["geometry"].apply(lambda x: x.exterior)
    shelves = gp.read_file("basins/GRE_Basins_shelf_extensions.gpkg").to_crs(crs)
    basin_dissolved_by_basin = basin_gp.dissolve(col)
    shelves_dissolved_by_basin = shelves.dissolve(col)
    basin_plus_shelves_geom = basin_dissolved_by_basin.union(shelves_dissolved_by_basin, align=True)
    basin_plus_shelves = gp.GeoDataFrame(basin_dissolved_by_basin, geometry=basin_plus_shelves_geom)
    m = pd.concat([basin_dissolved_by_basin, basin_plus_shelves]).dissolve(col).reset_index()
    m_no_periphery = m[m[col] != "ICE_CAP"]
    gis = gp.GeoDataFrame(m_no_periphery.dissolve())

    gis["SUBREGION1"] = "GIS"
    m = pd.concat([m, gis])
    m.to_file(path / f"{name}_w_shelves.gpkg")
    print("Done.\n")


if __name__ == "__main__":
    for basin in [mouginot, imbie]:
        prepare_basin(basin)
