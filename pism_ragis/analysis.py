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
Module for data analysis
"""

import pathlib
import time
from typing import Union

import numpy as np
import pandas as pd
import xarray as xr
from joblib import Parallel, delayed
from pandas.api.types import is_string_dtype
from SALib.analyze import delta
from scipy.interpolate import interp1d
from tqdm.auto import tqdm

from .processing import tqdm_joblib


def prepare_df(url: str):
    """
     Helper function to read csv or parquet file and return pd.DataFrame.

     Parameters
    ----------
     url : str
         The file location

     Returns
     -------
     pd.DataFrame
         a Pandas DataFrame
    """

    suffix = pathlib.Path(url).suffix
    if suffix in (".csv", ".gz"):
        df = pd.read_csv(str, parse_dates=["time"])
    elif suffix in (".parquet"):
        df = pd.read_parquet(url)
    else:
        print(f"{suffix} not recognized")

    return df


def sensitivity_analysis(
    df: pd.DataFrame,
    ensemble_file: str,
    calc_variables: Union[str, list] = [
        "grounding_line_flux (Gt year-1)",
        "limnsw (kg)",
    ],
    n_jobs: int = 4,
    sensitivity_indices: Union[str, list] = ["delta", "S1"],
    seed: Union[None, int] = None,
) -> pd.DataFrame:
    """
    Calculate sensitivity indices using SALIB and return pd.DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        A DataFrame procduced with processing.convert_netcdf_to_dataframe
    ensemble_file: str
        A csv file that maps ensemble member id's to parameters
    calc_variables: list
        A list of variables for which sensitivity indices are calculated
    n_jobs: int
        Number of parallel workers
    sensitivity_indices: str or list
        A list of sensitivity indices

    Returns
    -------
    pd.DataFrame
        A Pandas DataFrame with sensitivity indices
    """

    if isinstance(calc_variables, str):
        calc_variables = [calc_variables]

    if isinstance(sensitivity_indices, str):
        sensitivity_indices = [sensitivity_indices]

        print("Running sensitivity analysis")
    print("-------------------------------------------\n")
    start_time = time.perf_counter()

    # remove True/False
    id_df = (pd.read_csv(ensemble_file) * 1).replace(np.nan, 0)

    param_names = id_df.drop(columns="id").columns.values.tolist()
    for k, col in id_df.items():
        if is_string_dtype(col):
            u = col.unique()
            u.sort()
            v = [k for k, v in enumerate(u)]
            col.replace(to_replace=dict(zip(u, v)), inplace=True)
    # Define a salib "problem"
    problem = {
        "num_vars": len(id_df.drop(columns="id").columns.values),
        "names": param_names,  # Parameter names
        "bounds": zip(
            id_df.drop(columns="id").min().values,
            id_df.drop(columns="id").max().values,
        ),  # Parameter bounds
    }

    df = pd.merge(id_df, df, on="id")
    # filter out dates with only 1 experiment, e.g., due to
    # CDO changing the mid-point time when running averaging
    df = pd.concat([x for _, x in df.groupby(by="time") if len(x) > 1])
    n_dates = len(df["time"].unique())
    if n_jobs == 1:
        sensitivity_dfs = []
        for m_date, s_df in df.groupby(by="time"):
            sensitivity_dfs.append(
                compute_sensitivity_indices(
                    m_date,
                    s_df,
                    id_df,
                    problem,
                    calc_variables,
                    seed=seed,
                    sensitivity_indices=sensitivity_indices,
                )
            )
    else:
        with tqdm_joblib(tqdm(desc="Processing date", total=n_dates)) as progress_bar:
            sensitivity_dfs = Parallel(n_jobs=n_jobs)(
                delayed(compute_sensitivity_indices)(
                    m_date,
                    s_df,
                    id_df,
                    problem,
                    calc_variables,
                    seed=seed,
                    sensitivity_indices=sensitivity_indices,
                )
                for m_date, s_df in df.groupby(by="time")
            )
            del progress_bar

    sensitivity_df = pd.concat(sensitivity_dfs)
    sensitivity_df.reset_index(inplace=True, drop=True)

    finish_time = time.perf_counter()
    time_elapsed = finish_time - start_time
    print(f"Program finished in {time_elapsed:.0f} seconds")

    return sensitivity_df


def compute_sensitivity_indices(
    m_date,
    s_df: pd.DataFrame,
    id_df: pd.DataFrame,
    problem: dict,
    calc_variables: list,
    sensitivity_indices: list,
    seed: Union[None, int],
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Calculate sensitivity indices using SALIB and return pd.DataFrame.

    Parameters
    ----------
    m_date: datetime64[ns]
        The date stamp of the time to be processed
    s_df : pd.DataFrame
        A DataFrame procduced with processing.convert_netcdf_to_dataframe
    id_df : pd.DataFrame
        A DataFrame with parameters for all ensemble members
    problem: dict
        A SALib-like problem description, see https://salib.readthedocs.io/en/latest/user_guide/basics.html#an-example
        e.g.:
        problem = {
                   'num_vars': 3,
                    'names': ['x1', 'x2', 'x3'],
                    'bounds': [[-3.14159265359, 3.14159265359],
                               [-3.14159265359, 3.14159265359],
                               [-3.14159265359, 3.14159265359]]
                   }
     calc_variables: list
        A list of variables for which sensitivity indices are calculated
     n_jobs: int
         Number of parallel workers
     sensitivity_indices: str or list
         A list of sensitivity indices

    Returns
    -------
    pd.DataFrame
        A Pandas DataFrame with sensitivity indices
    """

    missing_ids = list(set(id_df["id"]).difference(s_df["id"]))
    if missing_ids:
        if verbose:
            print(f"The following simulation ids are missing:\n   {missing_ids}")

        id_df_missing_removed = id_df[~id_df["id"].isin(missing_ids)]
        params = np.array(
            id_df_missing_removed.drop(columns="id").values, dtype=np.float32
        )
    else:
        params = np.array(id_df.drop(columns="id").values, dtype=np.float32)
    sensitivity_dfs = []
    for calc_variable in calc_variables:
        response_matrix = s_df[calc_variable].values
        Si = delta.analyze(
            problem,
            params,
            response_matrix,
            num_resamples=100,
            seed=seed,
            print_to_console=False,
        )
        Si_df = Si.to_df()

        s_dfs = []
        for s_index in sensitivity_indices:
            m_df = pd.DataFrame(
                data=Si_df[s_index].values.reshape(1, -1),
                columns=Si_df.transpose().columns,
            )
            m_df["Date"] = m_date
            m_df["Si"] = s_index
            m_df["Variable"] = calc_variable

            m_conf_df = pd.DataFrame(
                data=Si_df[s_index + "_conf"].values.reshape(1, -1),
                columns=Si_df.transpose().columns,
            )
            m_conf_df["Date"] = m_date
            m_conf_df["Si"] = s_index + "_conf"
            m_conf_df["Variable"] = calc_variable
            s_dfs.append(pd.concat([m_df, m_conf_df]))

        a_df = pd.concat(s_dfs)
        sensitivity_dfs.append(a_df)
    return pd.concat(sensitivity_dfs)


def resample_ensemble_by_data(
    observed: xr.Dataset,
    simulated: xr.Dataset,
    start_date: str = "1992-01-01",
    end_date: str = "2020-01-01",
    id_var: str = "exp_id",
    fudge_factor: float = 3.0,
    n_samples: int = 100,
    obs_mean_var: str = "mass_balance",
    obs_std_var: str = "mass_balance_uncertainty",
    sim_var: str = "ice_mass",
) -> np.ndarray:
    """
    Resample an ensemble of simulated data to match observed data using a likelihood-based approach.

    Parameters
    ----------
    observed : xr.Dataset
        An xarray Dataset containing the observed data.
    simulated : xr.Dataset
        An xarray Dataset containing the simulated data.
    start_date : str, optional
        The start date for the period over which to perform the resampling, by default "1992-01-01".
    end_date : str, optional
        The end date for the period over which to perform the resampling, by default "2020-01-01".
    id_var : str, optional
        The variable name in `simulated` that identifies each ensemble member, by default "exp_id".
    fudge_factor : float, optional
        A multiplicative factor applied to the observed standard deviation to widen the likelihood function,
        allowing for greater tolerance in the matching process, by default 3.0.
    n_samples : int, optional
        The number of samples to draw from the simulated ensemble, by default 100.
    obs_mean_var : str, optional
        The variable name in `observed` that represents the mean observed data, by default "mass_balance".
    obs_std_var : str, optional
        The variable name in `observed` that represents the observed data's standard deviation,
        by default "mass_balance_uncertainty".
    sim_var : str, optional
        The variable name in `simulated` that represents the simulated data to be resampled,
        by default "ice_mass".

    Returns
    -------
    np.ndarray
        An array of ensemble member IDs selected through the resampling process.

    Notes
    -----
    This function implements a resampling algorithm that uses a likelihood-based approach to select ensemble members
    from the simulated dataset that are most consistent with the observed data. The likelihood is computed based on
    the difference between the simulated and observed means, scaled by the observed standard deviation (adjusted by
    the fudge factor). This method allows for the incorporation of observational uncertainty into the ensemble
    selection process.
    """

    # Interpolate simulated data to match the observed data's calendar
    # Note: This needs to happen before selecting the time slice to avoid nans.
    simulated = simulated.interp_like(observed, method="linear")

    # Select the observed and simulated data within the specified date range
    observed = observed.sel(time=slice(start_date, end_date))
    simulated = simulated.sel(time=slice(start_date, end_date))

    observed = observed.sel(time=slice(start_date, end_date))
    simulated = simulated.sel(time=slice(start_date, end_date))

    simulated = simulated.interp_like(observed, method="nearest")

    # Calculate the observed mean and adjusted standard deviation
    obs_mean = observed[obs_mean_var]
    obs_std = fudge_factor * observed[obs_std_var]

    # Extract the simulated data
    sim = simulated[sim_var].dropna(dim="exp_id")
    # Compute the log-likelihood of each simulated data point
    n = len(obs_mean["time"])
    log_likes = -0.5 * ((sim - obs_mean) / obs_std) ** 2 - n * 0.5 * np.log(
        2 * np.pi * obs_std**2
    )
    log_likes_sum = log_likes.sum(dim="time")
    log_likes_scaled = log_likes_sum - log_likes_sum.mean()

    # Convert log-likelihoods to weights
    weights = np.exp(log_likes_scaled)
    weights /= weights.sum()

    # Draw samples based on the computed weights
    sampled_ids = np.random.choice(sim[id_var].values, size=n_samples, p=weights.values)

    return sampled_ids


def resample_ensemble_by_data_df(
    observed: pd.DataFrame,
    simulated: pd.DataFrame,
    id_var: str = "id",
    calibration_start: float = 1992.0,
    calibration_end: float = 2017.0,
    fudge_factor: float = 3,
    n_samples: int = 100,
    verbose: bool = False,
    m_var: str = "Mass (Gt)",
    m_var_std: str = "Mass uncertainty (Gt)",
    return_weights: bool = False,
) -> pd.DataFrame:
    """
    Resampling algorithm by Douglas C. Brinkerhoff


    Parameters
    ----------
    observed : pandas.DataFrame
        A dataframe with observations
    simulated : pandas.DataFrame
        A dataframe with simulations
    calibration_start : float
        Start year for calibration
    calibration_end : float
        End year for calibration
    fudge_factor : float
        Tolerance for simulations. Calculated as fudge_factor * standard deviation of observed
    n_samples : int
        Number of samples to draw.

    """

    observed_calib_time = (observed["Year"] >= calibration_start) & (
        observed["Year"] <= calibration_end
    )
    observed_calib_period = observed[observed_calib_time]
    simulated_calib_time = (simulated["Year"] >= calibration_start) & (
        simulated["Year"] <= calibration_end
    )
    simulated_calib_period = simulated[simulated_calib_time]

    resampled_list = []
    log_likes = []
    experiments = sorted(simulated_calib_period[id_var].unique())
    evals = []
    for i in experiments:
        exp_ = simulated_calib_period[(simulated_calib_period[id_var] == i)]
        exp_interp = interp1d(exp_["Year"], exp_[m_var])
        log_like = 0.0
        for year, observed_mean, observed_std in zip(
            observed_calib_period["Year"],
            observed_calib_period[m_var],
            observed_calib_period[m_var_std],
        ):
            try:
                observed_std *= fudge_factor
                exp = exp_interp(year)

                log_like -= 0.5 * (
                    (exp - observed_mean) / observed_std
                ) ** 2 + 0.5 * np.log(2 * np.pi * observed_std**2)
            except ValueError:
                pass
        if log_like != 0:
            evals.append(i)
            log_likes.append(log_like)
            if verbose:
                print(f"Experiment {i:.0f}: {log_like:.2f}")
    experiments_array = np.asarray(evals)
    w = np.array(log_likes)
    w -= w.mean()
    weights = np.exp(w)
    weights /= weights.sum()
    resampled_experiments = np.random.choice(experiments_array, n_samples, p=weights)
    new_frame = []
    for i in resampled_experiments:
        new_frame.append(simulated[(simulated[id_var] == i)])
    simulated_resampled = pd.concat(new_frame)
    resampled_list.append(simulated_resampled)

    simulated_resampled = pd.concat(resampled_list)

    if return_weights:
        return simulated_resampled, weights
    else:
        return simulated_resampled
