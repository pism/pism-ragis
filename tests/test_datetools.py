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
Tests for date tools module.
"""


from pism_ragis.datetools import days_in_year


def test_days_in_year():
    """
    Test the days_in_year function.

    This function tests the days_in_year function for various types of years:
    - Common year
    - Leap year
    - Century year that is not a leap year
    - Century year that is a leap year

    The days_in_year function is expected to return:
    - 365 days for a common year
    - 366 days for a leap year
    - 365 days for a century year that is not a leap year
    - 366 days for a century year that is a leap year

    Examples
    --------
    >>> test_days_in_year()
    """
    # Test for a common year
    assert days_in_year(2021) == 365

    # Test for a leap year
    assert days_in_year(2020) == 366

    # Test for a century year that is not a leap year
    assert days_in_year(1900) == 365

    # Test for a century year that is a leap year
    assert days_in_year(2000) == 366
