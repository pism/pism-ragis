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


import geopandas as gp
import numpy as np
import pandas as pd
from shapely import Point
from tqdm import tqdm

from pismragis.interpolation import interpolate_rkf, point_velocity


def compute_trajectory(
    p: Point,
    Vx: np.ndarray,
    Vy: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    dt: float = 0.1,
    total_time: float = 1000,
    reverse: bool = False,
) -> list[Point]:
    """
    Compute trajectory
    """
    if reverse:
        Vx = -Vx
        Vy = -Vy
    pts = [p]
    time = 0.0
    while abs(time) <= (total_time):
        p, _ = interpolate_rkf(Vx, Vy, x, y, p, delta_time=dt)
        pts.append(p)
        time += dt
    return pts


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
    progress = tqdm(enumerate(trajs), total=len(trajs), leave=False, position=3)
    for traj_id, traj in progress:
        progress.set_description(f"Converting trajectory {traj_id} to GeoDataFrame")
        vx, vy = point_velocity(Vx, Vy, x, y, traj)
        v = np.sqrt(vx**2 + vy**2)
        d = [0] + [traj[k].distance(traj[k - 1]) for k in range(1, len(traj))]
        traj_data = {
            "vx": vx,
            "vy": vy,
            "v": v,
            "trai_id": traj_id,
            "distance": d,
            "distance_from_origin": np.cumsum(d),
        }
        for k, v in attrs.items():
            traj_data[k] = v
        df = gp.GeoDataFrame.from_dict(traj_data, geometry=traj, crs="EPSG:3413")
        dfs.append(df)
    return pd.concat(dfs).reset_index(drop=True)
