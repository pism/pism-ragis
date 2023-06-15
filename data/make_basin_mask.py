#!/usr/bin/env python3
# Copyright (C) 2023 Andy Aschwanden
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
Module to create basin masks
"""


from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from pathlib import Path
from typing import Union

from osgeo import gdal, ogr

default_basin_file = "basins/GRE_Basins_IMBIE2_v1.3_epsg3413.shp"
default_layer = "GRE_Basins_IMBIE2_v1.3_epsg3413"

e0 = -678650
e1 = 905350
n0 = -3371600
n1 = -635600

default_extend = [e0, n0, e1, n1]


def get_info(vector_file: Union[Path, str]):
    """
    Return info
    """
    ds_v = ogr.Open(str(vector_file))
    for layer in ds_v:
        print("Layer Name:", layer.GetName())
        print("Layer Feature Count:", len(layer))
        print("Layer Schema")
        layer_defn = layer.GetLayerDefn()
        for i in range(layer_defn.GetFieldCount()):
            print(layer_defn.GetFieldDefn(i).GetName())
        # some layers have multiple geometric feature types
        # most of the time, it should only have one though
        for i in range(layer_defn.GetGeomFieldCount()):
            # some times the name doesn't appear
            # but the type codes are well defined
            print(
                layer_defn.GetGeomFieldDefn(i).GetName(),
                layer_defn.GetGeomFieldDefn(i).GetType(),
            )
        # get a feature with GetFeature(featureindex)
        # this is the one where featureindex may not start at 0
        layer.ResetReading()
        for feature in layer:
            print("Feature ID:", feature.GetFID())
            # get a metadata field with GetField('fieldname'/fieldindex)
            print("Feature Metadata Dict:", feature.items())


if __name__ == "__main__":
    # set up the option parser

    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.description = "Extract sub-regions from large-scale files."
    parser.add_argument("OUTFILE", nargs=1)
    parser.add_argument(
        "--attribute", help="attribute used for selection", default="SUBREGION1"
    )
    parser.add_argument(
        "--attribute_value", help="attribute used for selection", default="NW"
    )
    parser.add_argument(
        "--basin_file",
        dest="basin_file",
        help="Path to shape file with basins",
        default=default_basin_file,
    )
    parser.add_argument(
        "-n",
        "--n_procs",
        dest="n_procs",
        type=int,
        help="""number of cores/processors. default=4. Only used if --ugid all""",
        default=4,
    )
    parser.add_argument(
        "-r",
        "--resolution",
        type=float,
        help="""Resolution of output grid""",
        default=1200.0,
    )
    parser.add_argument(
        "-l",
        "--layers",
        type=str,
        help="""Layer""",
        default=default_layer,
    )

    options = parser.parse_args()
    attribute = options.attribute
    attribute_value = options.attribute_value
    layers = options.layers
    n_procs = options.n_procs
    ofile = options.OUTFILE[-1]
    resolution = options.resolution
    basin_file = Path(options.basin_file)

    select_cmd = f"""'{attribute}={attribute_value}'"""

    get_info(basin_file)

    gdal_options = gdal.RasterizeOptions(
        format="netCDF",
        where=select_cmd,
        layers=layers,
        xRes=resolution,
        yRes=resolution,
        noData=-9999,
        burnValues=1,
        outputBounds=default_extend,
    )
    ds = gdal.Rasterize(ofile, str(basin_file), options=gdal_options)
    ds = None
