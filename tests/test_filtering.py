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
Tests for filtering module
"""

# from typing import Tuple

import numpy as np
import pytest
import xarray as xr

from pism_ragis.filtering import sample_with_replacement, sample_with_replacement_xr

# @pytest.fixture(name="sampling_data")
# def fixture_create_sampling_data() -> Tuple[xr.Dataset, xr.Dataset]:
#     n = 25
#     m = 3
#     r = 1
#     x = np.linspace(-10, 10, n)
#     X = np.tile(x, (r, m, 1)).T
#     a = 2
#     b = 4
#     sigma_true = 2
#     p = 2 / n


# def Y_true(x, a, b):
#     return a * x + b


# def noise(sigma, size):
#     return np.random.normal(0, sigma, size=size)


@pytest.fixture(name="weights_da")
def fixture_create_weights_da() -> xr.DataArray:
    """
    Fixture to create a sample weights DataArray for testing.

    Returns
    -------
    xr.DataArray
        A DataArray with dimensions ('basin', 'exp_id', 'ensemble_id') and sample weights data.
    """
    return xr.DataArray.from_dict(
        {
            "dims": ("basin", "exp_id", "ensemble_id"),
            "attrs": {},
            "data": [
                [
                    [6.060561375673975e-05],
                    [0.696964558202507],
                    [0.29823485321531285],
                    [0.004739861264801057],
                    [1.2170362236786714e-07],
                ],
                [
                    [0.15397218134232646],
                    [0.056671155910062845],
                    [0.13327299064213569],
                    [0.09575498529499678],
                    [0.5603286868104781],
                ],
            ],
            "coords": {
                "ensemble_id": {
                    "dims": ("ensemble_id",),
                    "attrs": {},
                    "data": ["RAGIS"],
                },
                "basin": {"dims": ("basin",), "attrs": {}, "data": ["CE", "NO"]},
                "exp_id": {"dims": ("exp_id",), "attrs": {}, "data": [0, 1, 2, 3, 5]},
            },
            "name": "weights",
        }
    )


def test_sample_with_replacement_xr(weights_da: xr.DataArray) -> None:
    """
    Test the sample_with_replacement_xr function.

    Parameters
    ----------
    weights_da : xr.DataArray
        The DataArray containing the weights for sampling.

    Returns
    -------
    None
    """
    n_samples = 1_000
    seed = 42
    sample_with_replacement_xr(weights_da, n_samples=n_samples, seed=seed)


def test_sample_with_replacement():
    """
    Test the sample_with_replacement function.

    Returns
    -------
    None
    """
    weights = np.array([0.1, 0.2, 0.3, 0.4])
    exp_id = np.array([0, 1, 2, 3])
    n_samples = 1_000
    seed = 42

    result = sample_with_replacement(weights, exp_id, n_samples, seed)

    assert (
        len(result) == n_samples
    ), "The number of samples should be equal to n_samples"
    assert set(result).issubset(set(exp_id)), "All sampled IDs should be from exp_id"
    assert np.allclose(
        np.bincount(result, minlength=len(exp_id)) / n_samples, weights, atol=0.1
    ), "Sampled distribution should be close to weights"
