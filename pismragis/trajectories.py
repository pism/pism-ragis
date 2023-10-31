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


from typing import Tuple

import geopandas as gp
import numpy as np
import pandas as pd
from shapely import Point

from pismragis.interpolation import interpolate_rkf, velocity_at_point


def compute_trajectory(
    point: Point,
    Vx: np.ndarray,
    Vy: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
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
