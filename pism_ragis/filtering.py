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
Module for filtering (calibration).
"""

import logging
import warnings
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
import xarray as xr
from dask.diagnostics import ProgressBar

import pism_ragis.processing as prp
from pism_ragis.decorators import timeit
from pism_ragis.likelihood import log_normal
from pism_ragis.logger import get_logger
from pism_ragis.processing import config_to_dataframe, filter_config

logger: logging.Logger = get_logger("pism_ragis")


def sample_with_replacement(
    weights: np.ndarray, exp_id: np.ndarray, n_samples: int, seed: int
) -> np.ndarray:
    """
    Sample with replacement from exp_id based on the given weights.

    Parameters
    ----------
    weights : np.ndarray
        The probabilities associated with each entry in exp_id.
    exp_id : np.ndarray
        The array of experiment IDs to sample from.
    n_samples : int
        The number of samples to draw.
    seed : int
        The random seed for reproducibility.

    Returns
    -------
    np.ndarray
        An array of sampled experiment IDs.
    """
    rng = np.random.default_rng(seed)
    try:
        ids = rng.choice(exp_id, size=n_samples, p=weights)
    except:
        ids = exp_id
    return ids


def sample_with_replacement_xr(
    weights, n_samples: int = 100, seed: int = 0, dim="exp_id"
) -> xr.DataArray:
    """
    Sample with replacement from a DataArray along a specified dimension.

    Parameters
    ----------
    weights : xr.DataArray
        The DataArray containing the weights for sampling.
    n_samples : int, optional
        The number of samples to draw, by default 100.
    seed : int, optional
        The random seed for reproducibility, by default 0.
    dim : str, optional
        The dimension along which to sample, by default "exp_id".

    Returns
    -------
    xr.DataArray
        A DataArray with the sampled values along the specified dimension.
    """
    da = xr.apply_ufunc(
        sample_with_replacement,
        weights.chunk({dim: -1}),
        input_core_dims=[[dim]],
        output_core_dims=[["sample"]],
        vectorize=True,
        dask="parallelized",
        kwargs={
            dim: weights[dim].to_numpy(),
            "n_samples": n_samples,
            "seed": seed,
        },
        dask_gufunc_kwargs={"output_sizes": {"sample": n_samples}},
    )
    da.name = dim + "_sampled"
    return da.rename({"sample": dim})


@timeit
def importance_sampling(
    simulated: xr.Dataset,
    observed: xr.Dataset,
    log_likelihood: Callable,
    likelihood_kwargs={},
    dim: str = "exp_id",
    sum_dim=["time"],
    fudge_factor: float = 3.0,
    n_samples: int = 100,
    obs_mean_var: str = "mass_balance",
    obs_std_var: str = "mass_balance_uncertainty",
    sim_var: str = "mass_balance",
    seed: int = 0,
) -> np.ndarray:
    """
    Filter an ensemble of simulated data to match observed data using a likelihood-based approach.

    Parameters
    ----------
    simulated : xr.Dataset
        An xarray Dataset containing the simulated data.
    observed : xr.Dataset
        An xarray Dataset containing the observed data.
    log_likelihood : Callable
        The log-likelihood function to use for filtering.
    likelihood_kwargs : dict, optional
        Additional keyword arguments to pass to the log-likelihood function, by default {}.
    dim : str, optional
        The variable name in `simulated` that identifies each ensemble member, by default "exp_id".
    sum_dim : list, optional
        The dimensions to sum over when computing the log-likelihood, by default ["time"].
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
        by default "mass_balance".
    seed : int, optional
        The random seed for reproducibility, by default 0.

    Returns
    -------
    np.ndarray
        An array of ensemble member IDs selected through the filtering process.

    Notes
    -----
    This function implements a filtering algorithm that uses a likelihood-based approach to select ensemble members
    from the simulated dataset that are most consistent with the observed data. The likelihood is computed based on
    the difference between the simulated and observed means, scaled by the observed standard deviation (adjusted by
    the fudge factor). This method allows for the incorporation of observational uncertainty into the ensemble
    selection process.
    """

    # Interpolate simulated data to match the observed data's calendar
    simulated = simulated.interp_like(observed, method="linear")

    # Calculate the observed mean and adjusted standard deviation
    obs_mean = observed[obs_mean_var]
    obs_std = fudge_factor * observed[obs_std_var]

    # Extract the simulated data
    sim = simulated[sim_var]

    # Compute the log-likelihood of each simulated data point
    n = np.prod([observed.sizes[d] for d in sum_dim])
    log_likes = log_likelihood(sim, obs_mean, obs_std, n=n, **likelihood_kwargs)
    log_likes = log_likes.where(log_likes != 0, np.nan)
    log_likes.name = "log_likes"
    log_likes_sum = log_likes.sum(dim=sum_dim)
    log_likes_scaled = log_likes_sum - log_likes_sum.mean(dim=dim)
    # Convert log-likelihoods to weights
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", r"overflow encountered")
        weights = np.exp(log_likes_scaled)
    weights /= weights.sum(dim=dim)
    weights.name = "weights"

    samples = sample_with_replacement_xr(weights, n_samples=n_samples, seed=seed)
    return xr.merge([log_likes, weights, samples])


@timeit
def filter_outliers(
    ds: xr.Dataset,
    valid_range: List[float],
    outlier_variable: str,
    freq: str = "YS",
    subset: Dict[str, str | int] = {"basin": "GIS", "ensemble_id": "RAGIS"},
) -> Tuple[xr.Dataset, xr.Dataset]:
    """
    Filter outliers from a dataset based on a specified variable and range.

    This function filters out ensemble members from the dataset `ds` where the values of
    `outlier_variable` fall outside the specified `valid_range`. The filtering is done
    for the specified subset of the dataset.

    Parameters
    ----------
    ds : xr.Dataset
        The input dataset containing the data to be filtered.
    valid_range : List[float]
        A list containing the lower and upper bounds for the valid range.
    outlier_variable : str
        The variable in the dataset to be used for outlier detection.
    freq : str, optional
        The frequency for resampling the data, by default "YS".
    subset : Dict[str, Union[str, int]], optional
        A dictionary specifying the subset of the dataset to apply the filter on, by default {"basin": "GIS", "ensemble_id": "RAGIS"}.

    Returns
    -------
    Tuple[xr.Dataset, xr.Dataset]
        A tuple containing two xarray.Dataset objects:
        - The valid dataset without outliers.
        - The dataset containing only the outliers.
    """
    lower_bound, upper_bound = valid_range
    if hasattr(ds[outlier_variable], "units"):
        outlier_variable_units = ds[outlier_variable].attrs["units"]
    else:
        outlier_variable_units = ""
    print(
        f"Filtering outliers [{lower_bound}, {upper_bound}] {outlier_variable_units} for {outlier_variable}"
    )

    # Select the subset and drop non-numeric variables once
    subset_ds = (
        ds.sel(subset)
        .drop_vars([var for var in ds.data_vars if not ds[var].dtype.kind in "iufc"])
        .drop_vars(subset.keys(), errors="ignore")
    )

    outlier_filter = (
        subset_ds[outlier_variable].resample({"time": freq}).mean(dim="time")
    )

    mask = (outlier_filter <= lower_bound) | (outlier_filter >= upper_bound)
    mask = mask.any(dim="time").compute()  # Compute the mask

    # Filter the dataset based on the mask
    filtered_exp_ids = ds.exp_id.where(~mask, drop=True)
    outlier_exp_ids = ds.exp_id.where(mask, drop=True)

    n_members = len(ds.exp_id)
    n_members_filtered = len(filtered_exp_ids)
    valid_ds = ds.sel(exp_id=filtered_exp_ids)
    outliers_ds = ds.sel(exp_id=outlier_exp_ids)
    print(f"Ensemble size: {n_members}, outlier-filtered size: {n_members_filtered}\n")
    return valid_ds, outliers_ds


@timeit
def run_importance_sampling(
    observed: xr.Dataset,
    simulated: xr.Dataset,
    obs_mean_vars: List[str] = ["grounding_line_flux", "mass_balance"],
    obs_std_vars: List[str] = [
        "grounding_line_flux_uncertainty",
        "mass_balance_uncertainty",
    ],
    sim_vars: List[str] = ["grounding_line_flux", "mass_balance"],
    filter_range: List[int] = [1990, 2019],
    fudge_factor: float = 3.0,
    params: List[str] = [],
) -> Tuple[pd.DataFrame, xr.Dataset, xr.Dataset]:
    """
    Run sampling to process observed and simulated datasets.

    This function performs importance sampling using the specified observed and simulated datasets,
    processes the results, and returns a DataFrame with the prior and posterior configurations.

    Parameters
    ----------
    observed : xr.Dataset
        The observed dataset.
    simulated : xr.Dataset
        The simulated dataset.
    obs_mean_vars : List[str], optional
        A list of variable names for the observed mean values, by default ["grounding_line_flux", "mass_balance"].
    obs_std_vars : List[str], optional
        A list of variable names for the observed standard deviation values, by default ["grounding_line_flux_uncertainty", "mass_balance_uncertainty"].
    sim_vars : List[str], optional
        A list of variable names for the simulated values, by default ["grounding_line_flux", "mass_balance"].
    filter_range : List[int], optional
        A list containing the start and end years for filtering, by default [1990, 2019].
    fudge_factor : float, optional
        A fudge factor for the importance sampling, by default 3.0.
    params : List[str], optional
        A list of parameter names to be used for filtering configurations, by default [].

    Returns
    -------
    Tuple[pd.DataFrame, xr.Dataset, xr.Dataset]
        A tuple containing:
        - A DataFrame with the prior and posterior configurations.
        - The prior simulated dataset.
        - The posterior simulated dataset.
    """
    filter_start_year, filter_end_year = filter_range

    simulated_prior = simulated
    simulated_prior["ensemble"] = "Prior"

    prior_config = filter_config(simulated.isel({"time": 0}), params)
    prior_df = config_to_dataframe(prior_config, ensemble="Prior")

    prior_posterior_list = []
    posterior_list = []
    for obs_mean_var, obs_std_var, sim_var in zip(
        obs_mean_vars, obs_std_vars, sim_vars
    ):
        print(f"Importance sampling using {obs_mean_var}")
        f = importance_sampling(
            simulated=simulated.sel(
                time=slice(str(filter_start_year), str(filter_end_year))
            ),
            observed=observed.sel(
                time=slice(str(filter_start_year), str(filter_end_year))
            ),
            log_likelihood=log_normal,
            fudge_factor=fudge_factor,
            n_samples=len(simulated.exp_id),
            obs_mean_var=obs_mean_var,
            obs_std_var=obs_std_var,
            sim_var=sim_var,
        )

        with ProgressBar() as pbar:
            result = f.compute()
            logger.info(
                "Importance Sampling: Finished in %2.2f seconds", pbar.last_duration
            )

        importance_sampled_ids = result["exp_id_sampled"]
        importance_sampled_ids["basin"] = importance_sampled_ids["basin"].astype(str)

        simulated_posterior = simulated.sel(exp_id=importance_sampled_ids)
        simulated_posterior["ensemble"] = "Posterior"
        simulated_posterior = simulated_posterior.expand_dims(
            {"filtered_by": [obs_mean_var]}
        )

        posterior_config = filter_config(simulated_posterior.isel({"time": 0}), params)
        posterior_df = config_to_dataframe(posterior_config, ensemble="Posterior")

        prior_posterior_f = pd.concat([prior_df, posterior_df]).reset_index(drop=True)
        prior_posterior_f["filtered_by"] = obs_mean_var
        prior_posterior_list.append(prior_posterior_f)

        posterior_list.append(simulated_posterior)

    prior_posterior = pd.concat(prior_posterior_list).reset_index(drop=True)
    prior_posterior = prior_posterior.apply(prp.convert_column_to_numeric)
    prior = simulated_prior
    posterior = xr.concat(posterior_list, dim="filtered_by")
    return prior_posterior, prior, posterior
