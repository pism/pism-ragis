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

# pylint: disable=too-many-positional-arguments

"""
Module for date processing.
"""

import datetime
from calendar import isleap


def decimal_year_to_datetime(decimal_year: float) -> datetime.datetime:
    """
    Convert a decimal year to a datetime object.

    Parameters
    ----------
    decimal_year : float
        The decimal year to be converted.

    Returns
    -------
    datetime.datetime
        The corresponding datetime object.

    Notes
    -----
    The function calculates the date by determining the start of the year and adding
    the fractional part of the year as days. If the resulting date has an hour value
    of 12 or more, it rounds up to the next day and sets the time to midnight.
    """
    year = int(decimal_year)
    remainder = decimal_year - year
    start_of_year = datetime.datetime(year, 1, 1)
    days_in_year = (datetime.datetime(year + 1, 1, 1) - start_of_year).days
    date = start_of_year + datetime.timedelta(days=remainder * days_in_year)
    if date.hour >= 12:
        date = date + datetime.timedelta(days=1)
    return date.replace(hour=0, minute=0, second=0, microsecond=0)


def days_in_year(year: int) -> int:
    """
    Calculate the number of days in a given year.

    Parameters
    ----------
    year : int
        The year for which to calculate the number of days.

    Returns
    -------
    int
        The number of days in the specified year. Returns 366 if the year is a leap year, otherwise returns 365.
    """
    if isleap(year):
        return 366
    else:
        return 365
