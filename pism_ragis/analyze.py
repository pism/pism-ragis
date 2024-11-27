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

"""
Module for sensitivity analysis.
"""

from typing import Any, Dict

import numpy as np
import pandas as pd
import xarray as xr
from SALib.analyze import delta


def delta_analyze(
    response: np.ndarray,
    problem: Dict[str, Any],
    ensemble_df: pd.DataFrame,
    dim: str = "pism_config_axis",
) -> xr.Dataset:
    """
    Perform SALib delta analysis.

    Parameters
    ----------
    response : np.ndarray
        The response variable to analyze.
    problem : dict
        A dictionary defining the problem for SALib analysis. It should contain keys like 'num_vars' and 'names'.
    ensemble_df : pd.DataFrame
        A DataFrame containing the ensemble data to be analyzed.
    dim : str, optional
        The name of the dimension to use for the configuration axis in the resulting Dataset (default is "pism_config_axis").

    Returns
    -------
    xr.Dataset
        An xarray Dataset containing the results of the delta analysis.
    """
    try:
        delta_moments = delta.analyze(
            problem,
            ensemble_df.values,
            response,
            num_resamples=10,
            seed=0,
            print_to_console=False,
        )
        df = delta_moments.to_df()
    except Exception:  # pylint: disable=broad-exception-caught
        delta_analysis = {
            key: np.empty(problem["num_vars"]) + np.nan
            for key in ["delta", "delta_conf", "S1", "S1_conf"]
        }
        df = pd.DataFrame.from_dict(delta_analysis)
        df[dim] = problem["names"]
        df.set_index(dim, inplace=True)
    return xr.Dataset.from_dataframe(df)
