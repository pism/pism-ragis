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

from typing import Union

import numpy as np
import pandas as pd
import xarray as xr


def run_stats(
    infiles: list,
    stats_vars: list = [
        "processor_hours",
        "wall_clock_hours",
        "model_years_per_processor_hour",
    ],
    experiment: Union[str, None] = None,
) -> pd.DataFrame:
    """
    Collect PISM run_stats for a list of files and returns a DataFrame
    """
    dfs = []
    for m_file in infiles:
        with xr.open_dataset(m_file) as ds:
            dfs.append(
                pd.DataFrame(
                    data=np.array(
                        [ds["run_stats"].attrs[stats_var] for stats_var in stats_vars]
                    ).reshape(1, -1),
                    columns=stats_vars,
                )
            )
    df = pd.concat(dfs)
    if experiment:
        df["Experiment"] = experiment
    return df.reset_index(drop=True)