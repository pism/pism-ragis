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


from pathlib import Path
from typing import Union

import cf_xarray
import matplotlib as mpl
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import xarray as xr

from pism_ragis.processing import preprocess_nc


def smoothed_step_function(
    time: pd.DatetimeIndex, amplification_factor: float = 1.0, step_year: int = 2000
) -> np.ndarray:
    """
    Generate a smoothed step function based on the day of the year.

    This function creates a smoothed step function that ramps up from 0 to 1,
    stays at 1, and then ramps down from 1 to 0 over specified periods of the year.

    Parameters
    ----------
    time : pd.DatetimeIndex
        A pandas DatetimeIndex representing the time series.
    amplification_factor : float, optional
        The factor by which the function value is amplified during the stay-at-one period, by default 1.0.
    step_year : int, optional
        The year at which the amplification factor is applied, by default 2000.

    Returns
    -------
    np.ndarray
        An array of the same length as `time` containing the smoothed step function values.
    """
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
        ramp_up_mask = (
            (years == year)
            & (day_of_year >= start_ramp_up)
            & (day_of_year <= end_ramp_up)
        )
        values[ramp_up_mask] = (day_of_year[ramp_up_mask] - start_ramp_up) / (
            end_ramp_up - start_ramp_up
        )
        # Stay at 1
        stay_at_one_mask = (
            (years == year)
            & (day_of_year > end_ramp_up)
            & (day_of_year < start_ramp_down)
        )
        if year >= step_year:
            values[stay_at_one_mask] = amplification_factor
        else:
            values[stay_at_one_mask] = 1

        # Ramp down from 1 to 0
        ramp_down_mask = (
            (years == year)
            & (day_of_year >= start_ramp_down)
            & (day_of_year <= end_ramp_down)
        )
        values[ramp_down_mask] = 1 - (day_of_year[ramp_down_mask] - start_ramp_down) / (
            end_ramp_down - start_ramp_down
        )
    return values


amplification_factors = [-1.0, 1.0, 1.05, 1.10, 1.20]
start_year = 1900
end_year = 2025
step_year = 2000

time = xr.date_range(str(start_year), str(end_year), freq="D")
time_centered = time[:-1] + (time[1:] - time[:-1]) / 2
zeros = np.zeros(len(time_centered))
# Create an xarray.DataArray with the time coordinate and the array of zeros
da = xr.DataArray(
    zeros, coords=[time_centered], dims=["time"], name="frac_calving_rate"
)

result_dir = Path("calving")
result_dir.mkdir(parents=True, exist_ok=True)
for k, amplification_factor in enumerate(amplification_factors):
    filename = result_dir / Path(f"seasonal_calving_id_{k}_{start_year}_{end_year}.nc")
    print(f"Processing {filename}")
    # Apply the smoothed step function to the time coordinate
    ds = da.to_dataset().copy()
    data = smoothed_step_function(
        time_centered, amplification_factor=amplification_factor, step_year=step_year
    )
    if amplification_factor == -1.0:
        data *= 0
        data += 1
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

infiles = result_dir.glob(f"seasonal_calving_id_*_{start_year}_{end_year}.nc")
ds = xr.open_mfdataset(infiles, preprocess=preprocess_nc)

rcparams = {
    "axes.linewidth": 0.25,
    "xtick.direction": "in",
    "xtick.major.size": 2.5,
    "xtick.major.width": 0.25,
    "ytick.direction": "in",
    "ytick.major.size": 2.5,
    "ytick.major.width": 0.25,
    "hatch.linewidth": 0.25,
}

mpl.rcParams.update(rcparams)
rc_params = {
    "font.size": 6,
    # Add other rcParams settings if needed
}

with mpl.rc_context(rc=rc_params):

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=(6.2, 1.8))
    fig.subplots_adjust(bottom=0.2, wspace=0.05)  # adjust space between Axes

    ds["frac_calving_rate"].plot(ax=ax1, hue="exp_id", lw=0.5)
    ds["frac_calving_rate"].plot(ax=ax2, hue="exp_id", lw=0.5)
    ds["frac_calving_rate"].plot(ax=ax3, hue="exp_id", lw=0.5)

    # zoom-in / limit the view to different portions of the data
    ax1.set_xlim(np.datetime64("1980-01-01"), np.datetime64("1984-01-01"))
    ax2.set_xlim(np.datetime64("1998-01-01"), np.datetime64("2002-01-01"))
    ax3.set_xlim(np.datetime64("2016-01-01"), np.datetime64("2020-01-01"))

    ax2.get_legend().set_visible(False)
    ax3.get_legend().set_visible(False)

    # hide the spines between ax and ax2
    ax1.set_xlabel(None)
    ax3.set_xlabel(None)
    ax2.set_ylabel(None)
    ax3.set_ylabel(None)
    # ax1.spines.bottom.set_visible(False)
    # ax2.spines.top.set_visible(False)
    # ax1.xaxis.tick_top()
    # ax1.tick_params(labeltop=False)  # don't put tick labels at the top
    # ax2.xaxis.tick_bottom()

    d = 0.75  # proportion of vertical to horizontal extent of the slanted line
    kwargs = {
        "marker": [(-1, -d), (1, d)],
        "markersize": 6,
        "linestyle": "none",
        "color": "k",
        "mec": "k",
        "mew": 1,
        "clip_on": False,
    }
    ax1.plot([1, 1], [0, 1], transform=ax1.transAxes, **kwargs)
    ax2.plot([1, 1], [0, 1], transform=ax2.transAxes, **kwargs)
    ax2.plot([0, 0], [0, 1], transform=ax2.transAxes, **kwargs)
    ax3.plot([0, 0], [0, 1], transform=ax3.transAxes, **kwargs)
    fig.savefig("calving/seasonal_calving.pdf")
