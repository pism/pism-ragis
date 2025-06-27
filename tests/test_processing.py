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
Tests for procesing module.
"""

import numpy as np
import xarray as xr

from pism_ragis.processing import calculate_area


def test_drop_nonnumeric_vars():
    """
    Test the drop_nonnumeric_vars method of the UtilsMethods class.

    This test creates a sample xarray Dataset with both numeric and non-numeric variables,
    applies the drop_nonnumeric_vars method, and checks that non-numeric variables are dropped
    and that the remaining variables are numeric.
    """
    data = xr.Dataset(
        {
            "temperature": (("x", "y"), [[15.5, 16.2], [14.8, 15.1]]),
            "humidity": (("x", "y"), [[80, 85], [78, 82]]),
            "location": (("x", "y"), [["A", "B"], ["C", "D"]]),
        }
    )

    # Apply the drop_nonnumeric_vars method
    numeric_data = data.utils.drop_nonnumeric_vars()

    # Check that non-numeric variables are dropped
    assert "location" not in numeric_data.data_vars
    assert "temperature" in numeric_data.data_vars
    assert "humidity" in numeric_data.data_vars

    # Check the data types of the remaining variables
    assert np.issubdtype(numeric_data["temperature"].dtype, np.number)
    assert np.issubdtype(numeric_data["humidity"].dtype, np.number)


def test_calculate_area():
    """
    Test the calculate_area function.

    This test checks if the calculate_area function correctly calculates the area of each grid cell
    given arrays of latitudes and longitudes.

    The test uses predefined latitude and longitude arrays and compares the function output with
    the expected output calculated manually.
    """
    # Define test inputs
    lat = np.array([0, 1, 2])
    lon = np.array([0, 1, 2])

    # Expected output (calculated manually or using a trusted method)
    R = 6371000  # Radius of the Earth in meters
    lat_rad = np.deg2rad(lat)
    lon_rad = np.deg2rad(lon)
    dlon = np.diff(lon_rad)
    expected_area = np.zeros((len(lat) - 1, len(lon) - 1))

    for i in range(len(lat) - 1):
        for j in range(len(lon) - 1):
            expected_area[i, j] = (R**2) * np.abs(np.sin(lat_rad[i + 1]) - np.sin(lat_rad[i])) * np.abs(dlon[j])

    # Call the function
    result = calculate_area(lat, lon)

    # Assert the result is as expected
    np.testing.assert_array_almost_equal(result, expected_area, decimal=5)
