# Copyright (C) 2023-24 Andy Aschwanden
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
Tests for domain module.
"""

import numpy as np
import xarray as xr

from pism_ragis.domain import create_domain, new_range


def test_new_range():
    """
    Test new_range function.
    """
    x = np.array([0, 1, 2, 3, 4])
    dx = 1.0
    center, Lx, N = new_range(x, dx)

    assert center == 2.0, "Center should be 2.0"
    assert Lx == 2.5, "Half-width should be 2.5"
    assert N == 5, "Number of grid points should be 5"

    x = np.array([0, 2, 4, 6, 8])
    dx = 2.0
    center, Lx, N = new_range(x, dx)

    assert center == 4.0, "Center should be 4.0"
    assert Lx == 5.0, "Half-width should be 5.0"
    assert N == 5, "Number of grid points should be 5"


def test_create_domain():
    """
    Test domain creation.
    """
    x_bnds = [0, 10]
    y_bnds = [0, 20]
    ds = create_domain(x_bnds, y_bnds)
    assert isinstance(ds, xr.Dataset), "The result should be an xarray.Dataset"
    assert "x" in ds.coords, "Dataset should have 'x' coordinate"
    assert "y" in ds.coords, "Dataset should have 'y' coordinate"
    assert "x_bnds" in ds.data_vars, "Dataset should have 'x_bnds' data variable"
    assert "y_bnds" in ds.data_vars, "Dataset should have 'y_bnds' data variable"
    assert "domain" in ds.data_vars, "Dataset should have 'domain' data variable"

    _ = (
        np.testing.assert_array_equal(ds["x_bnds"].values, [[0, 10]]),
        "x_bnds values should match the input",
    )
    _ = (
        np.testing.assert_array_equal(ds["y_bnds"].values, [[0, 20]]),
        "y_bnds values should match the input",
    )
