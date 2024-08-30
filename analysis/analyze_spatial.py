# Copyright (C) 2024 Andy Aschwanden, Constantine Khroulev
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
Analyze RAGIS ensemble.
"""

import time
import warnings
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from importlib.resources import files
from pathlib import Path
from typing import Any, Dict, Hashable, List, Mapping, Union

import dask
import numpy as np
import pandas as pd
import pylab as plt
import rioxarray as rxr
import seaborn as sns
import toml
import xarray as xr
from dask.diagnostics import ProgressBar
from dask.distributed import Client, LocalCluster, progress
from joblib import Parallel, delayed
from tqdm.auto import tqdm

from pism_ragis.filtering import importance_sampling
from pism_ragis.likelihood import log_normal, log_pseudo_huber
from pism_ragis.processing import preprocess_nc

xr.set_options(keep_attrs=True)

plt.style.use("tableau-colorblind10")

sim_alpha = 0.5
sim_cmap = sns.color_palette("crest", n_colors=4).as_hex()[0:3:2]
sim_cmap = ["#a6cee3", "#1f78b4"]
sim_cmap = ["#CC6677", "#882255"]
obs_alpha = 1.0
obs_cmap = ["0.8", "0.7"]
# obs_cmap = ["#88CCEE", "#44AA99"]
hist_cmap = ["#a6cee3", "#1f78b4"]


if __name__ == "__main__":
    __spec__ = None
    # set up the option parser
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.description = "Compute ensemble statistics."
    parser.add_argument(
        "--result_dir",
        help="""Result directory.""",
        type=str,
        default="./results",
    )
    parser.add_argument(
        "--obs_url",
        help="""Path to "observed" mass balance.""",
        type=str,
        default="data/itslive/GRE_G0240_1985.nc",
    )
    parser.add_argument(
        "--filter_range",
        help="""Time slice used for Importance Sampling. Default="1990 2019". """,
        type=str,
        nargs=2,
        default="1986 2019",
    )
    parser.add_argument(
        "--outlier_range",
        help="""Ensemble members outside this range are removed. Default="-1_250 250". """,
        type=str,
        nargs=2,
        default="-1250 -250",
    )
    parser.add_argument(
        "--outlier_variable",
        help="""Quantity to filter outliers. Default="grounding_line_flux".""",
        type=str,
        default="grounding_line_flux",
    )
    parser.add_argument(
        "--fudge_factor",
        help="""Observational uncertainty multiplier. Default=3""",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--n_jobs", help="""Number of parallel jobs.""", type=int, default=4
    )
    parser.add_argument(
        "--notebook",
        help="""Use when running in a notebook to display a nicer progress bar. Default=False.""",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--parallel",
        help="""Open dataset in parallel. Default=False.""",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--resampling_frequency",
        help="""Resampling data to resampling_frequency for importance sampling. Default is "MS".""",
        type=str,
        default="MS",
    )
    parser.add_argument(
        "--reference_year",
        help="""Reference year.""",
        type=int,
        default=1986,
    )
    parser.add_argument(
        "--temporal_range",
        help="""Time slice to extract.""",
        type=str,
        nargs=2,
        default=None,
    )
    parser.add_argument(
        "FILES",
        help="""Ensemble netCDF files.""",
        nargs="*",
    )

    options = parser.parse_args()
    spatial_files = options.FILES
    filter_start_year, filter_end_year = options.filter_range.split(" ")
    fudge_factor = options.fudge_factor
    notebook = options.notebook
    parallel = options.parallel
    reference_year = options.reference_year
    resampling_frequency = options.resampling_frequency
    outlier_variable = options.outlier_variable
    outlier_range = [float(v) for v in options.outlier_range.split(" ")]
    ragis_config_file = Path(
        str(files("pism_ragis.data").joinpath("ragis_config.toml"))
    )
    ragis_config = toml.load(ragis_config_file)

    crs = "EPSG:3413"
    ds = (
        xr.open_mfdataset(
            spatial_files,
            parallel=True,
            chunks="auto",
            preprocess=preprocess_nc,
            combine="nested",
            concat_dim="exp_id",
        )
        .squeeze()
        .sortby("exp_id")
    )
    ds.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=True)
    ds.rio.write_crs(crs, inplace=True)

    observed = xr.open_dataset(options.obs_url)
    observed.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=True)
    observed.rio.write_crs(crs, inplace=True)

    time = pd.date_range("1985-01-01", periods=1, freq="YS")
    observed = observed[["v", "v_err", "ice"]].expand_dims({"time": time})
    observed = observed.where(observed["ice"])
    observed_resampled = observed.interp_like(ds)

    print("Importance sampling using v")
    f = importance_sampling(
        observed=observed_resampled,
        simulated=ds,
        log_likelihood=log_pseudo_huber,
        n_samples=len(ds.exp_id),
        fudge_factor=5,
        obs_mean_var="v",
        obs_std_var="v_err",
        sim_var="velsurf_mag",
        sum_dim=["x", "y"],
        likelihood_kwargs={"delta": 1.35},
    )
    with ProgressBar():
        filtered_ids = f.compute()
