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

# pylint: disable=unused-import,assignment-from-none,unexpected-keyword-arg

import time
import warnings
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
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
from dask.distributed import Client, progress
from scipy.ndimage import label
from shapely.geometry import Polygon
from tqdm.auto import tqdm


def save_netcdf(
    ds: xr.Dataset,
    output_filename: str | Path = "GRE_G0240_1985_2018_IDW_EXP_1.nc",
    comp={"zlib": True, "complevel": 2},
):
    """
    Save the xarray dataset to a NetCDF file with specified compression.

    Parameters
    ----------
    ds : xarray.Dataset
        The dataset to be saved.
    output_filename : str or Path, optional
        The output filename for the NetCDF file, by default "GRE_G0240_1985_2018_IDW_EXP_1.nc".
    comp : dict, optional
        Compression settings for the NetCDF file, by default {"zlib": True, "complevel": 2}.
    """
    encoding = {
        var: comp for var in ds.data_vars if np.issubdtype(ds[var].dtype, np.number)
    }
    with ProgressBar():
        ds.to_netcdf(output_filename, encoding=encoding)


def fast_extract_gcm_profile(gcm_var, depth_index_da):
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
    return gcm_var.isel(depth=depth_index_da).transpose("time", "y", "x")


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


def compute_label(da: xr.DataArray, connectivity: int = 2, seed: tuple = None):
    structure = np.ones((3, 3)) if connectivity == 2 else None
    labeled_array, _ = label(da.data, structure=structure)
    seed_label = labeled_array[seed]  # Note: (y,x)
    conn_mask = labeled_array == seed_label
    return conn_mask


def compute_label_xr(
    da: xr.DataArray,
    connectivity: int = 2,
    seed: tuple = None,
    dim: str | Iterable[Hashable] | None = ["y", "x"],
):
    return xr.apply_ufunc(
        compute_label,
        da,
        input_core_dims=[dim],
        output_core_dims=[dim],
        kwargs={"seed": seed},
        vectorize=True,
        dask="parallelized",
    )


def compute_connectivity_mask(
    mask: xr.DataArray, seed_ij: tuple[int, int], connectivity=2
):
    """
    Compute a connectivity mask from a binary mask using a seed point.

    Parameters
    ----------
    mask : xarray.DataArray
        Binary mask (True/False or 1/0) with spatial dimensions.
    seed_ij : tuple of int
        (x, y) indices of the seed point.
    connectivity : int, optional
        Connectivity for labeling (2 for 8-connectivity), by default 2.

    Returns
    -------
    xarray.DataArray
        Boolean mask of the connected region containing the seed point.
    """
    structure = np.ones((3, 3)) if connectivity == 2 else None
    labeled_array, num_features = label(mask.data, structure=structure)
    seed_label = labeled_array[seed_ij[1], seed_ij[0]]  # Note: (y,x)
    conn_mask = labeled_array == seed_label
    return xr.DataArray(
        conn_mask, coords=mask.coords, dims=mask.dims, name="connectivity_mask"
    )


def get_ij(ds: xr.Dataset, xp, yp):
    """
    Get the (i, j) indices in the dataset closest to the given (x, y) coordinates.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset with 'x' and 'y' coordinates.
    xp : float
        X coordinate.
    yp : float
        Y coordinate.

    Returns
    -------
    tuple of int
        (i, j) indices corresponding to the nearest grid point.
    """
    nearest = ds.sel(x=xp, y=yp, method="nearest")
    i = ds.get_index("x").get_loc(nearest.x.item())
    j = ds.get_index("y").get_loc(nearest.y.item())
    return (i, j)


def process_basin(basin, basin_wet):
    """
    Prepare a single ocean basin, extracting and masking salinity and temperature.

    Parameters
    ----------
    basin : object
        Basin geometry or identifier.
    basin_wet : xarray.DataArray
        Masked bed data for the basin.
    """
    z_salinity = fast_extract_gcm_profile(gcm_basin["salinity_ocean"], depth_index_da)
    z_theta = fast_extract_gcm_profile(gcm_basin["theta_ocean"], depth_index_da)

    # Mask using basin_bed (if it's NaN, set output to NaN)
    mask = basin_bed.notnull()

    z_salinity = xr.where(mask, z_salinity, np.nan)
    z_theta = xr.where(mask, z_theta, np.nan)

    # Assign time coordinate and reorder dims
    z_salinity = z_salinity.assign_coords(time=gcm_ds.time).transpose("time", "y", "x")
    z_theta = z_theta.assign_coords(time=gcm_ds.time).transpose("time", "y", "x")

    # Combine into Dataset
    ds = xr.Dataset({"salinity_ocean": z_salinity, "theta_ocean": z_theta})
    ds = ds.expand_dims(basin=[b])
    gcm_basins.append(ds)


# Suppress specific warning from loky
warnings.filterwarnings("ignore", category=UserWarning)

if __name__ == "__main__":
    __spec__ = None  # type: ignore

    # set up the option parser
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.description = "Prepare GrIMP and BedMachine."
    options = parser.parse_args()

    client = Client()
    print(f"Open client in browser: {client.dashboard_link}")

    thin = 3
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
            date = xr.date_range(
                start=str(years[0]), end=str(years[-1]), freq="YS", use_cftime=True
            )

            coords = {
                "depth": (
                    ["depth"],
                    z,
                    {"units": "m", "axis": "Z", "positive": "down"},
                ),
                "basin": (
                    ["basin"],
                    [b],
                ),
            }

            ds = xr.Dataset(
                {
                    "salinity_ocean": xr.DataArray(
                        data=salinity,
                        dims=["time", "depth", "basin"],
                        coords={
                            "time": date,
                            "depth": coords["depth"],
                            "basin": coords["basin"],
                        },
                        attrs={
                            "units": "g/kg",
                        },
                    ),
                    "theta_ocean": xr.DataArray(
                        data=temperature,
                        dims=["time", "depth", "basin"],
                        coords={
                            "time": date,
                            "depth": coords["depth"],
                            "basin": coords["basin"],
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
            dfs.append(gpd.GeoDataFrame([{"id": b, "geometry": polygon}], crs=crs))
            dss.append(ds)
        basins_df = pd.concat(dfs)
        gcm_ds = (
            xr.concat(dss, dim="basin").sel(time=slice("1980", "1990")).chunk("auto")
        )

        start = time.time()
        print(f"Generating deepest index mask")
        seeds_gp = gpd.read_file("ocean/seed_points.gpkg")
        n_seeds = len(seeds_gp)
        level_masks = []

        bed = dem_ds["bed"]

        for s, seed in tqdm(seeds_gp.iterrows(), total=n_seeds, position=1):
            basin_geometry = basins_df.iloc[s].geometry
            seed_point = seed.geometry.coords.xy
            seed_ij = get_ij(dem_ds, *seed_point)  # (x_index, y_index)
            masks = []

            for d in tqdm(levels, position=2, leave=False):
                mask = bed < d
                connectivity_mask = compute_connectivity_mask(
                    mask, seed_ij, connectivity=2
                )
                connectivity_mask = connectivity_mask.astype("float")
                connectivity_mask = connectivity_mask.expand_dims({"depth": [d]})
                masks.append(connectivity_mask)

            level_mask = (
                xr.concat(masks, dim="depth")
                .sum(dim="depth", skipna=False)
                .rio.write_crs(crs, inplace=True)
                .rio.clip([basin_geometry], drop=False)
            )
            level_masks.append(level_mask.expand_dims({"seed": [s]}))

        # levels are 0-indexed
        seeded_level_mask = (
            xr.concat(level_masks, dim="seed").sum(dim="seed").astype(int)
        )
        seeded_level_mask.to_netcdf("deepest_index.nc")

        # Use vectorized indexing
        seeded_depth_mask = xr.apply_ufunc(
            lambda idx: z[idx],
            seeded_level_mask,
            vectorize=True,
            dask="parallelized",
            output_dtypes=[float],
        )
        print("Finished calculating deepest index mask\n")
        end = time.time()
        time_elapsed = end - start
        print(f"...time elapsed {time_elapsed:.0f}s")

        gcm_basins = []
        n_basins = len(basins_df)

        print(f"Generating {gcm} foring")
        print("-" * 80)
        for b, basin in tqdm(basins_df.iterrows(), total=n_basins):
            """
            Iterate over each basin, extract and mask ocean variables, and
            append the processed dataset to the list.

            Parameters
            ----------
            b : int
                Basin index.
            basin : GeoSeries
                Basin geometry.
            """
            gcm_basin = gcm_ds.sel(basin=b)
            bed = dem_ds["bed"]
            wet_bed = bed.where(bed < 0.0)
            wet_bed.rio.write_crs(crs, inplace=True)

            basin_bed = wet_bed.rio.clip([basin.geometry], drop=False)

            depths = gcm_basin.depth.values  # (D,) float64
            target_depth = seeded_depth_mask.data  # (y, x) float32

            # Compute the nearest depth index for each point
            depth_indices = np.abs(
                depths[:, None, None] - target_depth[None, :, :]
            ).argmin(
                axis=0
            )  # (y, x)
            depth_index_da = xr.DataArray(
                depth_indices,
                coords=seeded_depth_mask.coords,
                dims=seeded_depth_mask.dims,
            )

            z_salinity = fast_extract_gcm_profile(
                gcm_basin["salinity_ocean"], depth_index_da
            )
            z_theta = fast_extract_gcm_profile(gcm_basin["theta_ocean"], depth_index_da)

            # Mask using basin_bed (if it's NaN, set output to NaN)
            mask = basin_bed.notnull()

            basin_mask = xr.zeros_like(basin_bed) + b

            z_salinity = xr.where(mask, z_salinity, np.nan)
            z_theta = xr.where(mask, z_theta, np.nan)

            # Assign time coordinate and reorder dims
            z_salinity = z_salinity.assign_coords(time=gcm_ds.time).transpose(
                "time", "y", "x"
            )
            z_salinity.attrs.update({"units": "g/kg"})
            z_theta = z_theta.assign_coords(time=gcm_ds.time).transpose(
                "time", "y", "x"
            )
            z_theta.attrs.update({"units": "degree_Celsius"})

            # Combine into Dataset
            ds = xr.Dataset(
                {
                    "salinity_ocean": z_salinity,
                    "theta_ocean": z_theta,
                    "basin": basin_mask,
                }
            )
            ds = ds.expand_dims(basin_id=[b])
            gcm_basins.append(ds)

        ds = xr.concat(gcm_basins, dim="basin_id").sum(dim="basin_id")
        ds = ds.rio.set_spatial_dims(x_dim="x", y_dim="y")
        ds.rio.write_crs(crs, inplace=True)
        # ds = ds.cf.add_bounds("time")

        save_netcdf(ds, f"{gcm}.nc")
