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


def compute_forcing(
    forcing: xr.Dataset,
    mask: xr.DataArray,
    target_depth: xr.DataArray,
    basin_id: int = 1,
):
    ds = (
        forcing.isel({"depth": target_depth})
        .where(mask.notnull())
        .drop_dims("basin", errors="ignore")
        .drop_vars("basin", errors="ignore")
    )
    basin_mask.name = "basin"
    ds = xr.merge([ds, basin_mask])
    return ds.expand_dims({"basin_id": [basin_id]})


def save_netcdf(
    ds: xr.Dataset,
    output_filename: str | Path = "output.nc",
    comp={"zlib": True, "complevel": 2},
    **kwargs,
):
    """
    Save the xarray dataset to a NetCDF file with specified compression,
    preserving existing encodings like grid_mapping.

    Parameters
    ----------
    ds : xarray.Dataset
        The dataset to be saved.
    output_filename : str or Path, optional
        The output filename for the NetCDF file.
    comp : dict, optional
        Compression settings for numerical variables.
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


def extract_profile(gcm_var, depth_index_da):
    """
    Quickly extract a GCM variable profile at specified depth indices.

    Parameters
    ----------
    gcm_var : xarray.DataArray
        The GCM variable with dimensions (time, depth, ...).
    depth_index_da : xarray.DataArray
        DataArray of integer indices specifying which depth to extract at each (y, x).

    Returns
    -------
    xarray.DataArray
        The extracted variable, transposed to (time, y, x).
    """
    # gcm_var: (time, depth)
    return gcm_var.isel(depth=depth_index_da)


def extract_gcm_profile(gcm_var, depth_field):
    """
    Extract a GCM variable profile at the nearest depth for each grid point.

    Parameters
    ----------
    gcm_var : xarray.DataArray
        The GCM variable with dimensions (time, depth, ...).
    depth_field : xarray.DataArray
        DataArray of depths to extract at each (y, x).

    Returns
    -------
    xarray.DataArray
        The extracted variable, with time as a dimension.
    """
    return xr.apply_ufunc(
        lambda d: gcm_var.sel(depth=d, method="nearest"),
        depth_field,
        input_core_dims=[[]],
        output_core_dims=[["time"]],
        dask_gufunc_kwargs={"output_sizes": {"time": gcm_var.sizes["time"]}},
        vectorize=True,
        dask="parallelized",
        output_dtypes=[gcm_var.dtype],
    )


def compute_label(da: xr.DataArray, seed: tuple = None, connectivity: int = 2):

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
    # FIXME: do not use "dim" get order from da
    nearest = da.sel(seed, method="nearest")
    seed_ij = tuple([da.get_index(d).get_loc(nearest[d].item()) for d in dim])

    da_ = xr.apply_ufunc(
        compute_label,
        da,
        input_core_dims=[dim],
        output_core_dims=[dim],
        kwargs={"seed": seed_ij},
        vectorize=True,
        dask="parallelized",
    )
    da_.name = "connectivity_mask"
    return da_


# Suppress specific warning from loky
warnings.filterwarnings("ignore", category=UserWarning)

if __name__ == "__main__":
    __spec__ = None  # type: ignore

    # set up the option parser
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.description = "Prepare GrIMP and BedMachine."
    options = parser.parse_args()

    thin = 12
    crs = "EPSG:3413"
    dem_ds = xr.open_dataset("dem/BedMachineGreenland-v5.nc").thin(
        {"x": thin, "y": thin}
    )
    dem_ds = dem_ds.rio.set_spatial_dims(x_dim="x", y_dim="y")
    dem_ds.rio.write_crs(crs, inplace=True)

    for gcm in ["MIROC5_RCP85"]:

        ocean = scipy.io.loadmat(f"ocean/ocean_extrap_{gcm}.mat")
        z = np.array(ocean["z"].ravel())
        levels = z[:-1] + np.diff(z) / 2
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

        seeds_gp = gpd.read_file("ocean/seed_points.gpkg")
        n_seeds = len(seeds_gp)
        level_masks = []

        bed = dem_ds["bed"]
        masks = []
        for d in levels:
            m = bed < d
            m = m.expand_dims({"depth": [d]})
            masks.append(m)
        mask = xr.concat(masks, dim="depth")

        for s, seed in tqdm(seeds_gp.iterrows(), total=n_seeds, position=1):
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
        deepest_index.to_netcdf("deepest_index.nc")

        # Use vectorized indexing
        deepest_level = xr.apply_ufunc(
            lambda idx: z[idx],
            deepest_index,
            vectorize=True,
            dask="parallelized",
            output_dtypes=[float],
        )

        n_basins = len(basins_df)
        start = time.time()
        print(f"Generating {gcm} forcing")
        print("-" * 80)

        bed_below_sea_level = (bed.where(bed < 0.0)).rio.write_crs(crs, inplace=True)

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
        save_netcdf(ds, f"{gcm}.nc", engine="h5netcdf")
        end = time.time()
        time_elapsed = end - start
        print(f"...time elapsed {time_elapsed:.0f}s")
