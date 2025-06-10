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
Prepare ISMIP6 Ocean Forcing.

This script processes ocean forcing data for ISMIP6 experiments, including
reading, masking, and extracting relevant oceanographic variables, and saving
the results as NetCDF files.

Examples
--------
$ python 09_prepare_ocean.py
"""

import time
import warnings
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

# pylint: disable=unused-import,assignment-from-none,unexpected-keyword-arg
from itertools import repeat
from pathlib import Path
from typing import Hashable, Iterable

import cf_xarray
import geopandas as gpd
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import rioxarray
import scipy
import xarray as xr
from dask.diagnostics import ProgressBar
from scipy.ndimage import label
from shapely.geometry import Polygon
from tqdm.auto import tqdm

xr.set_options(keep_attrs=True)


# ...existing code...


def save_netcdf(
    ds: xr.Dataset,
    output_filename: str | Path = "output.nc",
    comp={"zlib": True, "complevel": 2},
    **kwargs,
):
    """
    Save the xarray dataset to a NetCDF file with specified compression.

    Parameters
    ----------
    ds : xarray.Dataset
        The dataset to be saved.
    output_filename : str or Path, optional
        The output filename for the NetCDF file.
    comp : dict, optional
        Compression settings for numerical variables.
    **kwargs
        Additional keyword arguments passed to xarray.Dataset.to_netcdf.
    """
    encoding = {}

    for var in ds.data_vars:
        if np.issubdtype(ds[var].dtype, np.number):
            # Copy existing encoding and update with compression settings
            enc = ds[var].encoding.copy()
            enc.update(comp)
            encoding[var] = enc

    with ProgressBar():
        ds.to_netcdf(output_filename, encoding=encoding, **kwargs)


def compute_label(da: xr.DataArray, seed: tuple | None = None, connectivity: int = 2):
    """
    Compute a connectivity mask from a binary mask using a seed point.

    Parameters
    ----------
    da : xarray.DataArray
        Binary mask (True/False or 1/0) with spatial dimensions.
    seed : tuple of int, optional
        (y, x) indices of the seed point.
    connectivity : int, optional
        Connectivity for labeling (2 for 8-connectivity), by default 2.

    Returns
    -------
    np.ndarray
        Boolean mask of the connected region containing the seed point.
    """
    structure = np.ones((3, 3)) if connectivity == 2 else None
    labeled_array, _ = label(da.data, structure=structure)
    seed_label = labeled_array[seed]  # Note: (y,x)
    conn_mask = labeled_array == seed_label
    return conn_mask


def compute_label_xr(
    da: xr.DataArray,
    seed: dict | None = None,
    connectivity: int = 2,
    dim: str | Iterable[Hashable] = ["y", "x"],
):
    """
    Compute a connectivity mask as an xarray.DataArray using a seed point.

    Parameters
    ----------
    da : xarray.DataArray
        Binary mask (True/False or 1/0) with spatial dimensions.
    seed : dict, optional
        Dictionary with 'x' and 'y' keys for the seed point.
    connectivity : int, optional
        Connectivity for labeling (2 for 8-connectivity), by default 2.
    dim : str or Iterable, optional
        Dimensions to use for connectivity, by default ["y", "x"].

    Returns
    -------
    xarray.DataArray
        Boolean mask of the connected region containing the seed point.
    """
    nearest = da.sel(seed, method="nearest")
    seed_ij = tuple(da.get_index(d).get_loc(nearest[d].item()) for d in dim)

    da_ = xr.apply_ufunc(
        compute_label,
        da,
        input_core_dims=[dim],
        output_core_dims=[dim],
        kwargs={"seed": seed_ij, "connectivity": connectivity},
        vectorize=True,
        dask="parallelized",
    )
    da_.name = "connectivity_mask"
    return da_


def extract_forcing(p: Path | str, crs: str = "EPSG:3413"):
    """
    Extract ocean forcing fields and basin polygons from a .mat file.

    Parameters
    ----------
    p : Path or str
        Path to the .mat file.
    crs : str, optional
        Coordinate reference system to assign to the basin polygons.
        Default is "EPSG:3413".

    Returns
    -------
    forcing : xarray.Dataset
        Dataset containing salinity and temperature for each basin.
    basins_df : geopandas.GeoDataFrame
        GeoDataFrame with basin polygons.
    z : np.ndarray
        Depth levels.
    """
    ocean = scipy.io.loadmat(p)
    z = np.array(ocean["z"].ravel())
    n_basins = len(ocean["basins"][0])

    dfs = []
    dss = []
    for b in range(n_basins):
        x = (ocean["basins"][0][b][0]).ravel()
        y = (ocean["basins"][0][b][1]).ravel()

        years = ocean["year"].ravel()
        T = ocean["T"][b]
        S = ocean["S"][b]
        temperature = T.reshape(*T.shape, 1)
        salinity = S.reshape(*S.shape, 1)
        date = xr.date_range(start=str(years[0]), end=str(years[-1]), freq="YS")

        coords = {
            "depth": (
                ["depth"],
                z,
                {"units": "m", "axis": "Z", "positive": "down"},
            ),
            "basin": (
                ["basin_id"],
                [b + 1],
            ),
        }

        ds = xr.Dataset(
            {
                "salinity_ocean": xr.DataArray(
                    data=salinity,
                    dims=["time", "depth", "basin_id"],
                    coords={
                        "time": date,
                        "depth": coords["depth"],
                        "basin_id": coords["basin"],
                    },
                    attrs={
                        "units": "g/kg",
                    },
                ),
                "theta_ocean": xr.DataArray(
                    data=temperature,
                    dims=["time", "depth", "basin_id"],
                    coords={
                        "time": date,
                        "depth": coords["depth"],
                        "basin_id": coords["basin"],
                    },
                    attrs={
                        "units": "degree_Celsius",
                    },
                ),
            },
            attrs={"Conventions": "CF-1.8"},
        )

        polygon_coords = list(zip(x, y))
        if polygon_coords[0] != polygon_coords[-1]:
            polygon_coords.append(polygon_coords[0])

        polygon = Polygon(polygon_coords)
        df = gpd.GeoDataFrame([{"basin_id": b + 1, "geometry": polygon}], crs=crs)
        dfs.append(df)
        dss.append(ds)
    basins_df = pd.concat(dfs).reset_index(drop=True)
    forcing = xr.concat(dss, dim="basin_id")
    return forcing, basins_df, z


def compute_deepest_index(
    mask: xr.DataArray,
    basins_df: gpd.GeoDataFrame,
    seeds_gp: gpd.GeoDataFrame,
    crs: str = "EPSG:3413",
):
    """
    Compute the deepest index mask for each seed point and basin.

    Parameters
    ----------
    mask : xarray.DataArray
        Boolean mask (True/False or 1/0) with dimensions (depth, y, x), indicating where bed < depth.
    basins_df : geopandas.GeoDataFrame
        GeoDataFrame with basin polygons.
    seeds_gp : geopandas.GeoDataFrame
        GeoDataFrame with seed points.
    crs : str, optional
        Coordinate reference system to use for clipping and writing CRS. Default is "EPSG:3413".

    Returns
    -------
    xarray.DataArray
        DataArray with the deepest index for each (y, x) location.
    """
    level_masks = []
    for s, seed in tqdm(seeds_gp.iterrows(), total=len(seeds_gp)):
        basin_geometry = basins_df.iloc[s].geometry
        seed_point = {
            "x": seed.geometry.coords.xy[0],
            "y": seed.geometry.coords.xy[1],
        }

        deepest_index_ = compute_label_xr(mask, seed_point).astype("float")

        level_mask = (
            deepest_index_.sum(dim="depth", skipna=False)
            .rio.write_crs(crs, inplace=True)
            .rio.clip([basin_geometry], drop=False)
        ).drop_vars(["spatial_ref"], errors="ignore")
        level_masks.append(level_mask.expand_dims({"seed": [s]}))

    # levels are 0-indexed
    deepest_index = xr.concat(level_masks, dim="seed").sum(dim="seed").astype(int)
    return deepest_index


# Suppress specific warning from loky
warnings.filterwarnings("ignore", category=UserWarning)

if __name__ == "__main__":
    __spec__ = None  # type: ignore

    # set up the option parser
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.description = "Prepare ISMIP6 Ocean Forcing"
    parser.add_argument(
        "--crs",
        help="""Coordinate reference system. Default is EPSG:3413.""",
        type=str,
        default="EPSG:3413",
    )
    parser.add_argument(
        "--engine",
        help="""Engine for xarray. Default="netcdf4".""",
        type=str,
        default="h5netcdf",
    )
    parser.add_argument(
        "--thin",
        help="""Thinnig BedMachine. 1=150m, 4=600m. Default=4.""",
        type=int,
        default=40,
    )
    options = parser.parse_args()
    crs = options.crs
    engine = options.engine
    thin = options.thin

    start = time.time()
    dem_ds = xr.open_dataset("dem/BedMachineGreenland-v5.nc").thin(
        {"x": thin, "y": thin}
    )
    dem_ds = dem_ds.rio.set_spatial_dims(x_dim="x", y_dim="y")
    dem_ds.rio.write_crs(crs, inplace=True)
    bed = dem_ds["bed"]

    for gcm in ["MIROC5_RCP85"]:

        p = f"ocean/ocean_extrap_{gcm}.mat"
        forcing, basins_df, z = extract_forcing(p, crs=crs)
        levels = z[:-1] + np.diff(z) / 2

        seeds_gp = gpd.read_file("ocean/seed_points.gpkg")
        n_seeds = len(seeds_gp)

        masks = []
        for d in levels:
            m = bed < d
            m = m.expand_dims({"depth": [d]})
            masks.append(m)
        mask = xr.concat(masks, dim="depth")

        deepest_index = compute_deepest_index(mask, basins_df, seeds_gp)
        deepest_index = xr.where(bed < 0, deepest_index, 0)
        deepest_index.name = "deepest_index"
        deepest_index.to_netcdf("ocean/deepest_index.nc")

        # Use vectorized indexing
        deepest_level = xr.apply_ufunc(
            lambda idx, z=z: z[idx],
            deepest_index,
            vectorize=True,
            dask="parallelized",
            output_dtypes=[float],
        )

        n_basins = len(basins_df)
        print(f"Generating {gcm} forcing")
        print("-" * 80)

        depths = forcing.depth.values  # (D,) float64
        target_depth = deepest_level.data

        # Compute the nearest depth index for each point
        depth_indices = np.abs(depths[:, None, None] - target_depth[None, :, :]).argmin(
            axis=0
        )  # (y, x)
        target_depth_da = xr.DataArray(
            depth_indices,
            coords=deepest_level.coords,
            dims=deepest_level.dims,
        )

        mask = xr.concat(
            [
                bed.rio.write_crs(crs)
                .rio.clip([basin.geometry], drop=False)
                .drop_vars(["spatial_ref"], errors="ignore")
                .expand_dims({"basin_id": [basin.basin_id]})
                for _, basin in basins_df.iterrows()
            ],
            dim="basin_id",
        )
        mask.name = "deepest_index_mask"

        forcing_3d = forcing.isel({"depth": target_depth_da}).where(mask.notnull())
        basin_mask = xr.zeros_like(mask)

        for b in mask["basin_id"].values:
            cond = mask.sel({"basin_id": b}).notnull()
            basin_mask.loc[{"basin_id": b}] = xr.where(cond, b, 0)
            basin_mask = basin_mask.astype(int)
            basin_mask.name = "basin"

        ds = xr.merge([forcing_3d, basin_mask]).sum(dim="basin_id")
        ds = ds.rio.set_spatial_dims(x_dim="x", y_dim="y")
        ds.rio.write_crs(crs, inplace=True)
        ds = ds.drop_vars(["mapping", "depth"], errors="ignore").cf.add_bounds("time")

        save_netcdf(ds, f"ocean/{gcm}.nc", engine=engine)
        end = time.time()
        time_elapsed = end - start
        print(f"...time elapsed {time_elapsed:.0f}s")
