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

# pylint: disable=not-callable

"""
Tests for analyze module.
"""

from typing import Tuple

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from SALib.analyze import delta

from pism_ragis.analyze import delta_analysis


@pytest.fixture(name="sensitivity_data")
def fixture_load_sensitivity_data() -> Tuple[pd.DataFrame, xr.Dataset]:
    """
    Load test data.

    Returns
    -------
    Tuple[pd.DataFrame, xr.Dataset]
        A tuple containing the input DataFrame and response Dataset.
    """
    input_df = pd.read_csv("tests/data/sensitivity_input.csv", index_col="param")

    response_ds = xr.open_dataset("tests/data/sensitivity_response.nc")

    return input_df, response_ds


def test_delta_analyze(sensitivity_data):
    """
    Test delta analysis.

    Parameters
    ----------
    sensitivity_data : Tuple[pd.DataFrame, xr.Dataset]
        The sensitivity data for testing.
    """
    input_df, response_ds = sensitivity_data

    gdim = "GIS"
    group_dim = "basin"

    for filter_var in ["grounding_line_flux", "mass_balance"]:
        df = input_df[input_df[group_dim] == gdim]
        df = df.drop(columns=[group_dim])
        problem = {
            "num_vars": len(df.columns),
            "names": df.columns,  # Parameter names
            "bounds": zip(
                df.min().values,
                df.max().values,
            ),  # Parameter bounds
        }

        response = response_ds.sel({"basin": gdim}).isel(time=0)[filter_var].load()

        result_true = pd.read_csv(
            f"tests/data/sensitivity_result_{filter_var}.csv",
            index_col="param",
        )

        result = delta.analyze(problem, df.values, response.to_numpy(), seed=42, method="sobol").to_df()[
            ["S1", "S1_conf"]
        ]
        result = delta_analysis(Y=response.to_numpy(), X=df.to_numpy(), problem=problem).to_dataframe()[
            ["S1", "S1_conf"]
        ]
        np.testing.assert_array_almost_equal(result_true.values, result.values)
