# Copyright (C) 2023 Andy Aschwanden, Constantine Khroulev
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
Module provides functions for calculating trajectories
"""


from pathlib import Path
from typing import Tuple, Union

import geopandas as gp
import numpy as np
import pandas as pd
from geopandas import GeoDataFrame
from numpy import ndarray
from osgeo import ogr, osr
from shapely import Point
from tqdm.auto import tqdm
from xarray import DataArray

from pismragis.interpolation import interpolate_rkf, velocity_at_point


def compute_trajectory(
    point: Point,
    Vx: Union[ndarray, DataArray],
    Vy: Union[ndarray, DataArray],
    x: Union[ndarray, DataArray],
    y: Union[ndarray, DataArray],
    dt: float = 0.1,
    total_time: float = 1000,
    reverse: bool = False,
) -> Tuple[list[Point], list]:
    """
    Compute trajectory
    """
    if reverse:
        Vx = -Vx
        Vy = -Vy
    pts = [point]
    pts_error_estim = [0.0]
    time = 0.0
    while abs(time) <= (total_time):
        point, point_error_estim = interpolate_rkf(Vx, Vy, x, y, point, delta_time=dt)
        if (point is None) or (point_error_estim is None):
            break
        pts.append(point)
        pts_error_estim.append(point_error_estim)
        time += dt
    return pts, pts_error_estim


def compute_perturbation(
    url: Union[str, Path],
    VX_min: Union[ndarray, DataArray],
    VX_max: Union[ndarray, DataArray],
    VY_min: Union[ndarray, DataArray],
    VY_max: Union[ndarray, DataArray],
    x: Union[ndarray, DataArray],
    y: Union[ndarray, DataArray],
    perturbation: int = 0,
    sample: Union[list, ndarray] = [0.5, 0.5],
    total_time: float = 10_000,
    dt: float = 1,
    reverse: bool = False,
) -> GeoDataFrame:
    """
    Compute a perturbed trajectory.

    It appears OGR objects cannot be pickled by joblib hence we load it here.

    """
    Vx = VX_min + sample[0] * (VX_max - VX_min)
    Vy = VY_min + sample[1] * (VY_max - VY_min)

    ogr.UseExceptions()
    if isinstance(url, Path):
        url = str(url.absolute())
    in_ds = ogr.Open(url)

    layer = in_ds.GetLayer(0)
    layer_type = ogr.GeometryTypeToName(layer.GetGeomType())
    srs = layer.GetSpatialRef()
    srs_geo = osr.SpatialReference()
    srs_geo.ImportFromEPSG(3413)

    all_glaciers = []
    progress = tqdm(enumerate(layer), total=len(layer), leave=False)
    for ft, feature in progress:
        geometry = feature.GetGeometryRef()
        geometry.TransformTo(srs_geo)
        points = geometry.GetPoints()
        points = [Point(p) for p in points]
        attrs = feature.items()
        attrs["perturbation"] = perturbation
        glacier_name = attrs["name"]
        progress.set_description(f"""Processing {glacier_name}""")
        trajs = []
        for p in points:
            traj, _ = compute_trajectory(
                p, Vx, Vy, x, y, total_time=total_time, dt=dt, reverse=reverse
            )
            trajs.append(traj)
        df = trajectories_to_geopandas(trajs, Vx, Vy, x, y, attrs=attrs)
        all_glaciers.append(df)
    return pd.concat(all_glaciers)


def trajectories_to_geopandas(
    trajs: list,
    Vx: np.ndarray,
    Vy: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    attrs: dict = {},
) -> gp.GeoDataFrame:
    """Convert trajectory to GeoDataFrame"""
    dfs = []
    for traj_id, traj in enumerate(trajs):
        vx, vy = velocity_at_point(Vx, Vy, x, y, traj)
        v = np.sqrt(vx**2 + vy**2)
        d = [0] + [traj[k].distance(traj[k - 1]) for k in range(1, len(traj))]
        traj_data = {
            "vx": vx,
            "vy": vy,
            "v": v,
            "trai_id": traj_id,
            "traj_pt": range(len(traj)),
            "distance": d,
            "distance_from_origin": np.cumsum(d),
        }
        for k, v in attrs.items():
            traj_data[k] = v
        df = gp.GeoDataFrame.from_dict(traj_data, geometry=traj, crs="EPSG:3413")
        dfs.append(df)
    return pd.concat(dfs).reset_index(drop=True)
