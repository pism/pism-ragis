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

# pylint: disable=unused-import

"""
Seasonal calving.
"""

from argparse import ArgumentParser
import xarray as xr
import numpy as np
import pandas as pd
import cf_xarray
from typing import Union
from pathlib import Path

# Create a time coordinate that spans the years 2000 to 2020
# Define the smoothed step function
def smoothed_step_function(time, amplification_factor: float = 1.0):
    # Convert time to day of year and year
    day_of_year = time.dayofyear
    years = time.year
    # Initialize the function values to zero
    values = np.zeros_like(day_of_year, dtype=float)
    for year in np.unique(years):
        # Define the transition periods for the current year
        start_ramp_up = pd.Timestamp(f"{year}-04-01").dayofyear
        end_ramp_up = pd.Timestamp(f"{year}-05-01").dayofyear
        start_ramp_down = pd.Timestamp(f"{year}-09-01").dayofyear
        end_ramp_down = pd.Timestamp(f"{year}-10-01").dayofyear
        # Ramp up from 0 to 1
        ramp_up_mask = (years == year) & (day_of_year >= start_ramp_up) & (day_of_year <= end_ramp_up)
        values[ramp_up_mask] = (day_of_year[ramp_up_mask] - start_ramp_up) / (end_ramp_up - start_ramp_up)
        # Stay at 1
        stay_at_one_mask = (years == year) & (day_of_year > end_ramp_up) & (day_of_year < start_ramp_down)
        values[stay_at_one_mask] = amplification_factor
        # Ramp down from 1 to 0
        ramp_down_mask = (years == year) & (day_of_year >= start_ramp_down) & (day_of_year <= end_ramp_down)
        values[ramp_down_mask] = 1 - (day_of_year[ramp_down_mask] - start_ramp_down) / (end_ramp_down - start_ramp_down)
    return values

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
amplification_factors = [-1.0, 1.0, 1.05, 1.10, 1.20, 1.50, 2.00]
start_year = 1975
end_year = 2025
year_high = 2000

time = xr.date_range(str(start_year), str(end_year), freq="D")
time_centered = time[:-1] + (time[1:] - time[:-1]) / 2
zeros = np.zeros(len(time_centered))
# Create an xarray.DataArray with the time coordinate and the array of zeros
da = xr.DataArray(zeros, coords=[time_centered], dims=["time"], name="frac_calving_rate")

result_dir = Path("calving")
result_dir.mkdir(parents=True, exist_ok=True)
c = []
for k, amplification_factor in enumerate(amplification_factors):
    filename = result_dir / Path(f"seasonal_calving_id_{k}_{start_year}_{end_year}.nc")
    print(f"Processing {filename}")
    # Apply the smoothed step function to the time coordinate
    ds = da.to_dataset().copy()
    # if amplification_factor == -1.0:
    #     ds["frac_calving_rate"] = np.ones_like(ds.time.values)
    # else:
    data = smoothed_step_function(time_centered)
    if amplification_factor == -1.0:
        data *= 0
    ds["frac_calving_rate"].values = data
    ds = ds.cf.add_bounds("time")
    ds["time"].encoding = {
        "units": f"hours since {start_year}-01-01",
    }

    ds["time"].attrs.update(
        {
            "axis": "T",
            "long_name": "time",
        }
    )
    ds["frac_calving_rate"].attrs.update({"units": "1"})
    ds.attrs["Conventions"] = "CF-1.8"
    ds.to_netcdf(filename)

