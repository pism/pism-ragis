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
Prepare CALFIN front retreat.
"""
# pylint: disable=unused-import

import time
import warnings
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from pathlib import Path
from typing import Dict, Union

import cf_xarray as cfxr
import dask
import geopandas as gp
import numpy as np
import pandas as pd
import rioxarray as rxr
import xarray as xr
from geocube.api.core import make_geocube
from joblib import Parallel, delayed
from tqdm.auto import tqdm

from pism_ragis.processing import download_earthaccess_dataset, tqdm_joblib

gp.options.io_engine = "pyogrio"
xr.set_options(keep_attrs=True)
# Suppress specific warning from loky
warnings.filterwarnings("ignore", category=UserWarning)

x_min = -653000
x_max = 879700
y_min = -632750
y_max = -3384350
bbox = [x_min, y_min, x_max, y_max]
geom = {
    "type": "Polygon",
    "crs": {"properties": {"name": "EPSG:3413"}},
    "bbox": bbox,
    "coordinates": [
        [
            (x_min, y_min),
            (x_max, y_min),
            (x_max, y_max),
            (x_min, y_max),
            (x_min, y_min),  # Close the loop by repeating the first point
        ]
    ],
}


def dissolve(ds, date, crs: str = "EPSG:3413"):
    """
    Dissolve geometries.
    """
    ds = gp.GeoDataFrame(ds, crs=crs)
    geom_valid = ds.geometry.make_valid()
    ds.geometry = geom_valid
    ds = ds.dissolve()
    ds["Date"] = date
    ds = ds.set_index("Date")
    return ds


def aggregate(n, df):
    """
    Aggregate geometries.
    """
    if n == 0:
        return df.iloc[[n]]
    else:
        geom = df.iloc[range(n)].unary_union
        merged_df = df.iloc[[n]]
        merged_df.iloc[0].geometry = geom
        return merged_df


def create_ds(
    date: pd.Timestamp,
    ds1: gp.GeoDataFrame,
    ds2: gp.GeoDataFrame,
    geom: Dict,
    resolution: float = 450,
    crs: str = "EPSG:3413",
    result_dir: Union[Path, str] = "front_retreat",
    encoding_time: Dict = {
        "time": {"units": "hours since 1972-01-01 00:00:00", "calendar": "standard"}
    },
) -> Path:
    """
    Create a dataset representing land ice area fraction retreat and save it to a NetCDF file.

    Parameters
    ----------
    date : pd.Timestamp
        The date for which the dataset is created.
    ds1 : gp.GeoDataFrame
        The first GeoDataFrame containing the initial geometries.
    ds2 : gp.GeoDataFrame
        The second GeoDataFrame containing the geometries to be compared.
    geom : Dict
        The geometry dictionary for the geocube.
    resolution : float, optional
        The resolution of the geocube, by default 450.
    crs : str, optional
        The coordinate reference system, by default "EPSG:3413".
    result_dir : Union[Path, str], optional
        The directory where the result NetCDF file will be saved, by default "front_retreat".
    encoding_time : Dict, optional
        The encoding settings for the time variable, by default {"time": {"units": "hours since 1972-01-01 00:00:00", "calendar": "standard"}}.

    Returns
    -------
    Path
        The path to the saved NetCDF file.

    Examples
    --------
    >>> import geopandas as gp
    >>> import pandas as pd
    >>> from pathlib import Path
    >>> date = pd.Timestamp("2023-01-01")
    >>> ds1 = gp.read_file("path_to_ds1.shp")
    >>> ds2 = gp.read_file("path_to_ds2.shp")
    >>> geom = {"type": "Polygon", "coordinates": [[[...]]]}
    >>> result_path = create_ds(date, ds1, ds2, geom)
    >>> print(result_path)
    """
    ds = gp.GeoDataFrame(ds1, crs=crs)
    geom_valid = ds.geometry.make_valid()
    ds.geometry = geom_valid
    ds_dissolved = ds.dissolve()
    diff = ds2.difference(ds_dissolved.buffer(5))
    n = len(diff)
    diff_df = {"land_ice_area_fraction_retreat": np.ones(n)}
    diff_gp = gp.GeoDataFrame(data=diff_df, geometry=diff, crs=crs)
    ds = make_geocube(
        vector_data=diff_gp, geom=geom, resolution=(resolution, resolution)
    )
    ds = ds.fillna(0)
    ds["land_ice_area_fraction_retreat"].attrs["units"] = "1"

    start = date.replace(day=1)

    ds = ds.expand_dims(time=[start])
    p = Path(result_dir)
    p.mkdir(parents=True, exist_ok=True)

    comp = {"zlib": True, "complevel": 2}
    encoding = {var: comp for var in ds.data_vars}
    encoding.update(encoding_time)

    fn = p / Path(
        f"frontretreat_g{resolution}m_{start.year}-{start.month}-{start.day}.nc"
    )
    ds.to_netcdf(fn, encoding=encoding)
    return fn


if __name__ == "__main__":
    __spec__ = None

    # set up the option parser
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.description = "Prepare retreat mask from CALFIN."
    parser.add_argument(
        "--crs",
        help="""Coordinate reference system. Default is EPSG:3413.""",
        type=str,
        default="EPSG:3413",
    )
    parser.add_argument(
        "--resolution",
        help="""Raster resolution. Default is 450.""",
        type=int,
        default=450,
    )
    parser.add_argument(
        "--n_jobs", help="""Number of parallel jobs.""", type=int, default=8
    )
    options = parser.parse_args()

    doi = "10.5067/7FILV218JZA2"
    filter_str = "Greenland_polygons"
    result_dir = "calfin"
    download_earthaccess_dataset(doi=doi, filter_str=filter_str, result_dir=result_dir)

    crs = options.crs
    encoding = {
        "time": {"units": "hours since 1972-01-01 00:00:00", "calendar": "standard"}
    }
    resolution = options.resolution
    n_jobs = options.n_jobs

    print("Preparing datasets")
    imbie = gp.read_file("imbie/GRE_Basins_IMBIE2_v1.3_w_shelves.gpkg").to_crs(crs)
    calfin = gp.read_file(
        "calfin/termini_1972-2019_Greenland_polygons_v1.0.shp"
    ).to_crs(crs)

    date = pd.DatetimeIndex(calfin["Date"])
    calfin["Date"] = date
    calfin = calfin.set_index(date).sort_index()

    geom_valid = calfin.geometry.make_valid()
    calfin.geometry = geom_valid
    calfin_dissolved = calfin.dissolve()

    imbie_dissolved = imbie.dissolve()
    imbie_union = imbie_dissolved.union(calfin_dissolved)

    def dissolve_range(df, k):
        """
        Dissolve over a given range.
        """
        return df.reset_index().iloc[range(k)].dissolve(aggfunc="last")

    n_calfin_grouped = len(
        [date for date, df in calfin.groupby(by=pd.Grouper(freq="ME")) if len(df) > 0]
    )
    with tqdm_joblib(
        tqdm(desc="Grouping geometries", total=n_calfin_grouped)
    ) as progress_bar:
        result = Parallel(n_jobs=n_jobs)(
            delayed(dissolve)(ds, date)
            for date, ds in calfin.groupby(by=pd.Grouper(freq="ME"))
            if len(ds) > 0
        )
    calfin_grouped = pd.concat(result)

    with tqdm_joblib(
        tqdm(desc="Dissolving geometries", total=n_calfin_grouped)
    ) as progress_bar:
        result = Parallel(n_jobs=n_jobs)(
            delayed(dissolve_range)(calfin_grouped, k)
            for k in range(len(calfin_grouped))
        )

    calfin_aggregated = pd.concat(result[1::]).set_index("Date")
    n_calfin_aggregated = len(calfin_aggregated)
    with tqdm_joblib(
        tqdm(desc="Aggregating geometries", total=n_calfin_aggregated)
    ) as progress_bar:
        result = Parallel(n_jobs=n_jobs)(
            delayed(create_ds)(date, ds, imbie_union, geom=geom, resolution=resolution)
            for date, ds in calfin_aggregated.groupby(by=pd.Grouper(freq="ME"))
            if len(ds) > 0
        )

    p = Path("front_retreat")
    fn = Path(f"pism_g{resolution}m_frontretreat_calfin_1972_2019.nc")
    p_fn = p / fn

    print(f"Merging datasets and saving to {str(p_fn.absolute())}")

    result_filtered = [element for element in result if element is not None]

    start = time.time()
    ds = xr.open_mfdataset(result_filtered).load()
    ds = ds.cf.add_bounds("time")

    comp = {"zlib": True, "complevel": 2}
    encoding_compression = {var: comp for var in ds.data_vars}
    encoding.update(encoding_compression)

    ds.to_netcdf(p_fn, encoding=encoding)

    end = time.time()
    time_elapsed = end - start
    print(f"Time elapsed {time_elapsed:.0f}s")
