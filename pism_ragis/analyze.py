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

# pylint: disable=too-many-positional-arguments

"""
Module for sensitivity analysis.
"""

from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

# from dask.distributed import Client, progress
from SALib.analyze import delta, sobol

from pism_ragis.decorators import timeit


@timeit
# def run_sensitivity_analysis(
#     input_df: pd.DataFrame,
#     response_ds: xr.Dataset,
#     filter_vars: list[str],
#     group_dim: str = "basin",
#     iter_dim: str = "time",
#     notebook: bool = False,
#     client: Client | None = None,
# ) -> xr.Dataset:
#     """
#     Run delta sensitivity analysis on the given dataset.

#     This function calculates sensitivity indices for each basin in the dataset,
#     filtered by the specified variables. It uses Dask for parallel processing
#     to improve performance.

#     Parameters
#     ----------
#     input_df : pd.DataFrame
#         DataFrame containing ensemble information, with a 'basin' column to group by.
#     response_ds : xr.Dataset
#         The input dataset containing the data to be analyzed.
#     filter_vars : list[str]
#         List of variables to filter by for sensitivity analysis.
#     group_dim : str, optional
#         The dimension to group by, by default "basin".
#     iter_dim : str, optional
#         The dimension to iterate over, by default "time".
#     notebook : bool, optional
#         Whether to display a nicer progress bar when running in a notebook, by default False.

#     Returns
#     -------
#     xr.Dataset
#         A dataset containing the calculated sensitivity indices for each basin and filter variable.

#     Notes
#     -----
#     It is imperative to load the dataset before starting the Dask client,
#     to avoid each Dask worker loading the dataset separately, which would
#     significantly slow down the computation.
#     """
#     print("Calculating Sensitivity Indices")
#     print("===============================")

#     if client is None:
#         client = Client()
#         print(f"Open client in browser: {client.dashboard_link}")
#     sensitivity_indices_list = []
#     for gdim, df in input_df.groupby(by=group_dim):
#         df = df.drop(columns=[group_dim])
#         problem = {
#             "num_vars": len(df.columns),
#             "names": df.columns,  # Parameter names
#             "bounds": zip(
#                 df.min().values,
#                 df.max().values,
#             ),  # Parameter bounds
#         }
#         for filter_var in filter_vars:
#             print(
#                 f"  ...sensitivity indices for basin {gdim} filtered by {filter_var} ",
#             )

#             responses = response_ds.sel({group_dim: gdim})[filter_var].load()
#             responses_scattered = client.scatter(
#                 [responses.isel({"time": k}).to_numpy() for k in range(len(responses[iter_dim]))]
#             )

#             futures = client.map(
#                 delta_analysis,
#                 responses_scattered,
#                 X=df.to_numpy(),
#                 problem=problem,
#             )
#             progress(futures, notebook=notebook)
#             result = client.gather(futures)

#             sensitivity_indices = xr.concat([r.expand_dims(iter_dim) for r in result], dim=iter_dim)
#             sensitivity_indices[iter_dim] = responses[iter_dim]
#             sensitivity_indices = sensitivity_indices.expand_dims(group_dim, axis=1)
#             sensitivity_indices[group_dim] = [gdim]
#             sensitivity_indices = sensitivity_indices.expand_dims("filtered_by", axis=2)
#             sensitivity_indices["filtered_by"] = [filter_var]
#             sensitivity_indices_list.append(sensitivity_indices)

#     all_sensitivity_indices: xr.Dataset = xr.merge(sensitivity_indices_list)

#     return all_sensitivity_indices


def delta_analysis(
    Y: np.ndarray,
    X: np.ndarray,
    problem: dict[str, Any],
    dim: str = "pism_config_axis",
) -> xr.Dataset:
    """
    Perform SALib delta analysis.

    Parameters
    ----------
    Y : numpy.ndarray
        A NumPy array containing the model outputs.
    X : numpy.ndarray
        A NumPy matrix containing the model inputs.
    problem : dict
        A dictionary defining the problem for SALib analysis. It should contain keys like 'num_vars' and 'names'.
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
            X,
            Y,
            seed=42,
            method="sobol",
        )
        df = delta_moments.to_df()[["S1", "S1_conf"]]  # pylint: disable=not-callable
    except Exception:  # pylint: disable=broad-exception-caught
        delta_df = {key: np.empty(problem["num_vars"]) + np.nan for key in ["S1", "S1_conf"]}
        df = pd.DataFrame.from_dict(delta_df)
        df[dim] = problem["names"]
        df.set_index(dim, inplace=True)
    return xr.Dataset.from_dataframe(df)


def sobol_analysis(
    response: np.ndarray,
    problem: dict[str, Any],
    ensemble_df: pd.DataFrame,
    dim: str = "pism_config_axis",
) -> xr.Dataset:
    """
    Perform SALib Sobol analysis.

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
        An xarray Dataset containing the results of the Sobol analysis.
    """
    try:
        sobol_moments = sobol.analyze(
            problem,
            ensemble_df.values,
            response,
            num_resamples=10,
            seed=0,
            print_to_console=False,
        )
        df = sobol_moments.to_df()  # pylint: disable=not-callable
    except Exception:  # pylint: disable=broad-exception-caught
        sobol_df = {key: np.empty(problem["num_vars"]) + np.nan for key in ["S1", "S1_conf"]}
        df = pd.DataFrame.from_dict(sobol_df)
        df[dim] = problem["names"]
        df.set_index(dim, inplace=True)
    return xr.Dataset.from_dataframe(df)
