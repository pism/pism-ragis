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

# mypy: disable-error-code="no-redef"
# pylint: disable=no-name-in-module
"""
Tests for likelihood module.
"""

import numpy as np
import pytest
import xarray as xr
from scipy.special import pseudo_huber
from sklearn.metrics import jaccard_score

from pism_ragis.likelihood import (
    log_jaccard_score,
    log_normal,
    log_pseudo_huber,
)


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


def test_log_normal_ndarray() -> None:
    """
    Test log_normal likelihood with np.ndarray inputs.

    This test verifies that the log_normal function correctly computes the log-likelihood
    when the inputs are numpy arrays.
    """
    x: np.ndarray = np.array([1.0, 2.0, 3.0])
    mu: np.ndarray = np.array([0.0, 0.0, 0.0])
    std: np.ndarray = np.array([1.0, 1.0, 1.0])

    expected: np.ndarray = -0.5 * ((x - mu) / std) ** 2 - 0.5 * np.log(2 * np.pi * std**2)
    result: np.ndarray = log_normal(x, mu, std)
    assert np.allclose(result, expected), f"Expected {expected}, got {result}"


def test_log_pseudo_huber_ndarray() -> None:
    """
    Test log_pseudo_huber likelihood with np.ndarray inputs.

    This test verifies that the log_pseudo_huber function correctly computes the log-likelihood
    when the inputs are numpy arrays.
    """
    x: np.ndarray = np.array([1.0, 2.0, 3.0])
    mu: np.ndarray = np.array([0.0, 0.0, 0.0])
    std: np.ndarray = np.array([1.0, 1.0, 1.0])

    delta: float = 2
    expected: np.ndarray = -pseudo_huber(delta, (x - mu) / std) - 1
    result: np.ndarray = log_pseudo_huber(x, mu, std, delta=delta)
    assert np.allclose(result, expected), f"Expected {expected}, got {result}"

    delta: float = 0
    expected: np.ndarray = -pseudo_huber(delta, (x - mu) / std) - 1
    result: np.ndarray = log_pseudo_huber(x, mu, std, delta=delta)
    assert np.allclose(result, expected), f"Expected {expected}, got {result}"


def test_log_normal_xarray() -> None:
    """
    Test log_normal likelihood with xr.DataArray inputs.

    This test verifies that the log_normal function correctly computes the log-likelihood
    when the inputs are xarray DataArrays.
    """
    x: xr.DataArray = xr.DataArray(
        ([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]),
        coords={"x": [0, 1], "y": [0, 1, 2]},
        name="X",
    )
    mu: xr.DataArray = xr.DataArray(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        coords={"x": [0, 1], "y": [0, 1, 2]},
        name="mu",
    )
    std: xr.DataArray = xr.DataArray(
        [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
        coords={"x": [0, 1], "y": [0, 1, 2]},
        name="std",
    )
    expected: xr.DataArray = -0.5 * ((x - mu) / std) ** 2 - 0.5 * np.log(2 * np.pi * std**2)
    result: xr.DataArray = log_normal(x, mu, std)
    assert np.allclose(result, expected), f"Expected {expected}, got {result}"


def test_log_pseudo_huber_xarray() -> None:
    """
    Test log_pseudo_huber likelihood with xr.DataArray inputs.

    This test verifies that the log_pseudo_huber function correctly computes the log-likelihood
    when the inputs are xarray DataArrays.
    """
    x: xr.DataArray = xr.DataArray(
        ([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]),
        coords={"x": [0, 1], "y": [0, 1, 2]},
        name="X",
    )
    mu: xr.DataArray = xr.DataArray(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        coords={"x": [0, 1], "y": [0, 1, 2]},
        name="mu",
    )
    std: xr.DataArray = xr.DataArray(
        [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
        coords={"x": [0, 1], "y": [0, 1, 2]},
        name="std",
    )
    delta: float = 2.0
    expected = -pseudo_huber(delta, (x - mu) / std) - 1
    result: xr.DataArray = log_pseudo_huber(x, mu, std, delta=delta)
    assert np.allclose(result, expected), f"Expected {expected}, got {result}"


def test_log_jaccard_score_ndarray() -> None:
    """
    Test log_jaccard_score likelihood with np.ndarray inputs.

    This test verifies that the log_jaccard_score function correctly computes the log-likelihood
    when the inputs are numpy arrays.
    """
    y_true = np.array([[0, 1, 1], [1, 1, 0]])
    y_pred = np.array([[1, 1, 1], [1, 0, 0]])

    expected = -jaccard_score(y_true[0], y_pred[0])
    result = log_jaccard_score(y_true[0], y_pred[0])
    assert np.allclose(result, expected), f"Expected {expected}, got {result}"


def test_log_jaccard_score_xarray() -> None:
    """
    Test log_jaccard_score likelihood with xr.DataArray inputs.

    This test verifies that the log_jaccard_score function correctly computes the log-likelihood
    when the inputs are xarray DataArrays.
    """
    y_true: xr.DataArray = xr.DataArray([0, 1, 1], coords={"x": [-1, 0, 1]}, name="y_true")
    y_pred: xr.DataArray = xr.DataArray([1, 1, 1], coords={"x": [-1, 0, 1]}, name="y_pred")

    expected = -jaccard_score(y_true, y_pred)
    result = log_jaccard_score(y_true, y_pred)
    assert np.allclose(result, expected), f"Expected {expected}, got {result}"
