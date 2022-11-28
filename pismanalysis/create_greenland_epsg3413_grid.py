#!/usr/bin/env python
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import numpy as np
from netCDF4 import Dataset as CDF
from pyproj import Proj

domains = {
    "large": {"e0": -1254650, "e1": 1013350, "n0": -3479600, "n1": -527600},
    "small": {"e0": -678650, "e1": 905350, "n0": -3371600, "n1": -635600},
}


def create_grid_file(nc_outfile, domains):

    e0 = domains[domain]["e0"]
    e1 = domains[domain]["e1"]
    n0 = domains[domain]["n0"]
    n1 = domains[domain]["n1"]

    # Shift to cell centers
    e0 += grid_spacing / 2
    n0 += grid_spacing / 2
    e1 -= grid_spacing / 2
    n1 -= grid_spacing / 2

    de = dn = grid_spacing  # m
    M = int((e1 - e0) / de) + 1
    N = int((n1 - n0) / dn) + 1

    easting = np.linspace(e0, e1, M)
    northing = np.linspace(n0, n1, N)
    ee, nn = np.meshgrid(easting, northing)

    # Set up EPSG 3413 (NSIDC north polar stereo) projection
    projection = "epsg:3413"
    proj = Proj(projection)

    lon, lat = proj(ee, nn, inverse=True)

    nc = CDF(nc_outfile, "w", format=fileformat)

    nc.createDimension("x", size=easting.shape[0])
    nc.createDimension("y", size=northing.shape[0])

    var = "x"
    var_out = nc.createVariable(var, "d", dimensions=("x"))
    var_out.axis = "X"
    var_out.long_name = "X-coordinate in Cartesian system"
    var_out.standard_name = "projection_x_coordinate"
    var_out.units = "meters"
    var_out[:] = easting

    var = "y"
    var_out = nc.createVariable(var, "d", dimensions=("y"))
    var_out.axis = "Y"
    var_out.long_name = "Y-coordinate in Cartesian system"
    var_out.standard_name = "projection_y_coordinate"
    var_out.units = "meters"
    var_out[:] = northing

    var = "lon"
    var_out = nc.createVariable(var, "d", dimensions=("y", "x"))
    var_out.units = "degrees_east"
    var_out.valid_range = -180.0, 180.0
    var_out.standard_name = "longitude"
    var_out[:] = lon

    var = "lat"
    var_out = nc.createVariable(var, "d", dimensions=("y", "x"))
    var_out.units = "degrees_north"
    var_out.valid_range = -90.0, 90.0
    var_out.standard_name = "latitude"
    var_out[:] = lat

    var = "dummy"
    var_out = nc.createVariable(var, "f", dimensions=("y", "x"), fill_value=-9999)
    var_out.units = "meters"
    var_out.long_name = "Just A Dummy"
    var_out.comment = "This is just a dummy variable for CDO."
    var_out.grid_mapping = "mapping"
    var_out.coordinates = "lon lat"
    var_out[:] = 0.0

    mapping = nc.createVariable("mapping", "c")
    mapping.ellipsoid = "WGS84"
    mapping.false_easting = 0.0
    mapping.false_northing = 0.0
    mapping.grid_mapping_name = "polar_stereographic"
    mapping.latitude_of_projection_origin = 90.0
    mapping.standard_parallel = 70.0
    mapping.straight_vertical_longitude_from_pole = -45.0

    from time import asctime

    historystr = f"Created {asctime()} \n"
    nc.history = historystr
    nc.proj = projection
    nc.Conventions = "CF-1.5"
    nc.close()


if __name__ == "__main__":
    # set up the argument parser
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.description = "Create CDO-compliant grid description"
    parser.add_argument("FILE", nargs="*")
    parser.add_argument(
        "-g",
        "--grid_spacing",
        dest="grid_spacing",
        type=float,
        help="use X m grid spacing",
        default=1800,
    )
    parser.add_argument(
        "-d",
        "--domain",
        dest="domain",
        choices=domains.keys(),
        help="Choose domain size",
        default="small",
    )
    parser.add_argument(
        "-f",
        "--format",
        dest="fileformat",
        type=str.upper,
        choices=["NETCDF4", "NETCDF4_CLASSIC", "NETCDF3_CLASSIC", "NETCDF3_64BIT"],
        help="file format out output file",
        default="netcdf4",
    )

    options = parser.parse_args()
    args = options.FILE
    grid_spacing = options.grid_spacing  # convert
    domain = options.domain

    fileformat = options.fileformat.upper()

    if len(args) == 0:
        nc_outfile = "grn" + str(grid_spacing) + "m.nc"
    elif len(args) == 1:
        nc_outfile = args[0]
    else:
        print("wrong number arguments, 0 or 1 arguments accepted")
        parser.print_help()
        import sys

        sys.exit(0)

    create_grid_file(nc_outfile, domains)
