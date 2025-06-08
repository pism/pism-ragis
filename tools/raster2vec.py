#!/usr/bin/env python
# Copyright (C) 2015-17, 2025 Bob McNabb, Andy Aschwanden

"""
Convert raster velocity fields to vector line data.

This script reads raster files containing (U,V) components of a velocity field,
optionally prunes and scales the vectors, and outputs a vector file (e.g., GeoJSON, Shapefile)
with lines representing the velocity vectors.
"""

import logging
import logging.handlers
import re
import sys
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import cftime
import geopandas as gpd
import numpy as np
import pandas as pd
import rioxarray as rxr
import xarray as xr
from shapely.geometry import LineString, mapping
from tqdm.auto import tqdm

# create logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# create file handler which logs even debug messages
fh = logging.handlers.RotatingFileHandler("extract.log")
fh.setLevel(logging.DEBUG)
# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.ERROR)
# create formatter
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s"
)

# add formatter to ch and fh
ch.setFormatter(formatter)
fh.setFormatter(formatter)

# add ch to logger
logger.addHandler(ch)
logger.addHandler(fh)


parser = ArgumentParser(
    formatter_class=ArgumentDefaultsHelpFormatter,
    description="Convert rasters containing (U,V) components of velocity field to vector line data.",
)
parser.add_argument("INFILE", nargs=1)
parser.add_argument("OUTFILE", nargs=1)
parser.add_argument(
    "--vx",
    help="variable containing x components of velocity",
    default="vx",
)
parser.add_argument(
    "--vy",
    help="variable containing y components of velocity",
    default="vy",
)
parser.add_argument(
    "--vx_err",
    help="variable x components of error",
    default="vx_err",
)
parser.add_argument(
    "--vy_err",
    help="variable containing y components of error",
    default="vy_err",
)
parser.add_argument(
    "--crs",
    help="CRS code of project. Overrides input projection",
    default="epsg:3413",
)
parser.add_argument(
    "-s",
    type=float,
    dest="scale_factor",
    help="Scales length of line. Default=1.",
    default=1.0,
)
parser.add_argument(
    "-p",
    "--prune_factor",
    type=int,
    dest="prune_factor",
    help="Pruning. Only use every x-th value. Default=1",
    default=1,
)
parser.add_argument(
    "-t",
    "--threshold",
    type=float,
    dest="threshold",
    help="Magnitude values smaller or equal than threshold will be masked. Default=None",
    default=0.0,
)


args = parser.parse_args()
prune_factor = args.prune_factor
scale_factor = args.scale_factor
threshold = args.threshold
infile = args.INFILE[0]
outfile = args.OUTFILE[0]
vx_var = args.vx
vy_var = args.vy
crs = args.crs

ds = xr.open_dataset(infile, decode_timedelta=True).thin(
    {"x": prune_factor, "y": prune_factor}
)
ds = ds.rio.set_spatial_dims(x_dim="x", y_dim="y")
ds.rio.write_crs(crs, inplace=True)

try:
    ds = ds.where(ds.ice)
except:
    pass

x = ds["x"]
y = ds["y"]
X, Y = np.meshgrid(x, y)
log = True


if "time" in ds.coords:
    # Loop over each time step
    gdfs = []
    for t in tqdm(ds.time.values):
        timestamp = (
            np.datetime64(t).astype("datetime64[ns]").item()
        )  # convert to Python datetime

        # Select single time slice (still lazy until .compute())
        vx = ds[vx_var].sel(time=t)
        vy = ds[vy_var].sel(time=t)

        # Compute needed arrays just for this time
        vx_np = vx.compute().values
        vy_np = vy.compute().values
        speed = np.sqrt(vx_np**2 + vy_np**2)

        # Filter valid values
        mask = np.isfinite(vx_np) & np.isfinite(vy_np)

        x_a = X - scale_factor * vx_np / 2
        y_a = Y - scale_factor * vy_np / 2
        x_e = X + scale_factor * vx_np / 2
        y_e = Y + scale_factor * vy_np / 2

        # Flatten arrays
        x_a_flat = x_a.ravel()
        y_a_flat = y_a.ravel()
        x_e_flat = x_e.ravel()
        y_e_flat = y_e.ravel()
        vx_flat = vx_np.ravel()
        vy_flat = vy_np.ravel()
        speed_flat = speed.ravel()
        mask_flat = mask.ravel()

        # Efficient filtering
        valid_idx = np.flatnonzero(mask_flat)

        lines = [
            LineString([(x_a_flat[i], y_a_flat[i]), (x_e_flat[i], y_e_flat[i])])
            for i in valid_idx
        ]

        gdf = gpd.GeoDataFrame(
            {
                "time": [timestamp] * len(lines),
                "vx": vx_flat[valid_idx],
                "vy": vy_flat[valid_idx],
                "speed": speed_flat[valid_idx],
                "geometry": lines,
            },
            crs=ds.rio.crs,
        )

        gdfs.append(gdf)

    # Concatenate all time steps
    gdf = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True), crs=ds.rio.crs)
else:
    # Extract 2D slices of variables at this time step
    vx = ds[vx_var]
    vy = ds[vy_var]

    # Compute needed arrays just for this time
    vx_np = vx.compute().values
    vy_np = vy.compute().values
    speed = np.sqrt(vx_np**2 + vy_np**2)

    # Filter valid values
    mask = np.isfinite(vx_np) & np.isfinite(vy_np)

    x_a = X - scale_factor * np.log10(speed) / speed * vx_np / 2
    y_a = Y - scale_factor * np.log10(speed) / speed * vy_np / 2
    x_e = X + scale_factor * np.log10(speed) / speed * vx_np / 2
    y_e = Y + scale_factor * np.log10(speed) / speed * vy_np / 2

    # Flatten arrays
    x_a_flat = x_a.ravel()
    y_a_flat = y_a.ravel()
    x_e_flat = x_e.ravel()
    y_e_flat = y_e.ravel()
    vx_flat = vx_np.ravel()
    vy_flat = vy_np.ravel()
    speed_flat = speed.ravel()
    mask_flat = mask.ravel()

    # Efficient filtering
    valid_idx = np.flatnonzero(mask_flat)

    lines = [
        LineString([(x_a_flat[i], y_a_flat[i]), (x_e_flat[i], y_e_flat[i])])
        for i in valid_idx
    ]

    gdf = gpd.GeoDataFrame(
        {
            "vx": vx_flat[valid_idx],
            "vy": vy_flat[valid_idx],
            "speed": speed_flat[valid_idx],
            "geometry": lines,
        },
        crs=ds.rio.crs,
    )

gdf.to_file(outfile)
