#!/usr/bin/env python
# Copyright (C) 2020-23 Andy Aschwanden

# pylint: disable=no-name-in-module,chained-comparison
# mypy: ignore-errors

"""
Seasonal calving.
"""

from argparse import ArgumentParser
from calendar import isleap
from datetime import datetime
from pathlib import Path

import cftime
import numpy as np
import pylab as plt
from dateutil import rrule
from netCDF4 import Dataset as NC
from typin import Union

# set up the option parser
parser = ArgumentParser()
parser.add_argument("FILE", nargs="*")
parser.add_argument(
    "--year_high",
    type=float,
    help="Start when high values are applied.",
    default=2001,
)
parser.add_argument(
    "--calving_low",
    type=float,
    help="Low value.",
    default=1.0,
)
parser.add_argument(
    "--calving_high",
    type=float,
    help="High value.",
    default=1.5,
)

options = parser.parse_args()
args = options.FILE
amplification_factors = [-1.0, 1.0, 1.05, 1.10, 1.20, 1.50, 2.00]
start_year = 1975
end_year = 2025
year_high = 2000

if len(args) == 0:
    nc_outfile = "seasonal_calving.nc"
elif len(args) == 1:
    nc_outfile = args[0]
else:
    print("wrong number arguments, 0 or 1 arguments accepted")
    parser.print_help()
    import sys

    sys.exit(0)


def create_file(
    amplification_factor,
    filename: Union[str, Path],
    start_year: int = 1975,
    end_year: int = 2025,
    year_high: int = 2000,
):
    """
    Create netCDF.
    """

    def annual_calving(year_length, frac_calving_rate_max):
        """
        Create calving.
        """
        frac_calving_rate = np.zeros(year_length)
        for t in range(year_length):
            if (t <= winter_e) and (t >= winter_a):
                frac_calving_rate[
                    t
                ] = frac_calving_rate_max - frac_calving_rate_max / np.sqrt(
                    winter_e
                ) * np.sqrt(
                    np.mod(t, year_length)
                )
            elif (t > winter_e) and (t < spring_e):
                frac_calving_rate[t] = (
                    frac_calving_rate_max / np.sqrt(spring_e - winter_e)
                ) * np.sqrt(np.mod(t - winter_e, year_length))
            else:
                frac_calving_rate[t] = frac_calving_rate_max
        return frac_calving_rate

    start_date = datetime(start_year, 1, 1)
    end_date = datetime(end_year, 1, 1)

    calendar = "standard"
    units = "days since 1980-1-1"

    sampling_interval = "daily"
    rd = {
        "daily": rrule.DAILY,
        "weekly": rrule.WEEKLY,
        "monthly": rrule.MONTHLY,
        "yearly": rrule.YEARLY,
    }

    bnds_datelist = list(
        rrule.rrule(rd[sampling_interval], dtstart=start_date, until=end_date)
    )
    # calculate the days since refdate, including refdate, with time being the
    bnds_interval_since_refdate = cftime.date2num(
        bnds_datelist, units, calendar=calendar
    )
    time_interval_since_refdate = (
        bnds_interval_since_refdate[0:-1] + np.diff(bnds_interval_since_refdate) / 2
    )

    # mid-point value:
    # time[n] = (bnds[n] + bnds[n+1]) / 2
    time_interval_since_refdate = (
        bnds_interval_since_refdate[0:-1] + np.diff(bnds_interval_since_refdate) / 2
    )

    # Create netCDF file
    nc = NC(filename, "w", format="NETCDF4", compression_level=2)

    nc.createDimension("time")
    nc.createDimension("nb", size=2)

    var = "time"
    var_out = nc.createVariable(var, "d", dimensions="time")
    var_out.axis = "T"
    var_out.units = "days since 1980-1-1"
    var_out.long_name = "time"
    var_out.bounds = "time_bounds"
    var_out[:] = time_interval_since_refdate

    var = "time_bounds"
    var_out = nc.createVariable(var, "d", dimensions=("time", "nb"))
    var_out.bounds = "time_bounds"
    var_out[:, 0] = bnds_interval_since_refdate[0:-1]
    var_out[:, 1] = bnds_interval_since_refdate[1::]

    var = "frac_calving_rate"
    var_out = nc.createVariable(var, "f", dimensions="time")
    var_out.units = "1"

    winter_a = 300
    winter_e = 90
    spring_e = 105

    winter_a = 0
    winter_e = 150
    spring_e = 170

    idx = 0
    for year in range(start_year, end_year):
        print(f"Preparing Year {year}")
        if isleap(year):
            year_length = 366
        else:
            year_length = 365

        if year <= year_high:
            frac_calving_rate = annual_calving(year_length, 1)
        else:
            frac_calving_rate = annual_calving(year_length, amplification_factor)

        frac_calving_rate = np.roll(frac_calving_rate, -90)
        if amplification_factor > 0:
            var_out[idx::] = frac_calving_rate
        else:
            var_out[idx::] = 1.0
        idx += year_length

    calving = var_out[-year_length:-1]
    nc.amplification_factor = amplification_factor
    nc.close()
    return calving


result_dir = Path("calving")
result_dir.mkdir(parents=True, exist_ok=True)
c = []
for k, amplification_factor in enumerate(amplification_factors):
    filename = result_dir / Path(f"seasonal_calving_id_{k}_{start_year}_{end_year}.nc")
    calving = create_file(
        amplification_factor, filename, start_year, end_year, year_high
    )
    c.append(calving)

fig, ax = plt.subplots(1, 1, figsize=[3.2, 2.4])
for calving in c:
    calving = np.roll(calving, 0)
    ax.plot(calving)
fig.savefig("seasonal_calving.pdf")
