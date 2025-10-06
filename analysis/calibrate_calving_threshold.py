# Copyright (C) 2024-25 Andy Aschwanden, Constantine Khroulev
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

# pylint: disable=unused-import,too-many-positional-arguments,unused-argument
"""
Analyze RAGIS ensemble.
"""
import json
import time
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from importlib.resources import files
from itertools import chain
from pathlib import Path
from typing import Callable

import cartopy.crs as ccrs
import cf_xarray.units  # pylint: disable=unused-import
import geopandas as gp
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pint_xarray  # pylint: disable=unused-import
import rioxarray as rxr
import seaborn as sns
import toml
import xarray as xr
from dask.distributed import Client, progress
from shapely.geometry import mapping
from tqdm.auto import tqdm

from pism_ragis.filtering import importance_sampling
from pism_ragis.likelihood import log_jaccard_score_xr, log_normal_xr
from pism_ragis.logger import get_logger
from pism_ragis.processing import (
    config_to_dataframe,
    filter_config,
    prepare_liafr,
    preprocess_config,
    preprocess_nc,
)

xr.set_options(
    keep_attrs=True,
    warn_for_unclosed_files=False,
    use_flox=True,
    use_bottleneck=True,
    use_opt_einsum=True,
)


plt.style.use("tableau-colorblind10")
logger = get_logger("pism_ragis")

sim_alpha = 0.5
sim_cmap = sns.color_palette("crest", n_colors=4).as_hex()[0:3:2]
sim_cmap = ["#a6cee3", "#1f78b4"]
sim_cmap = ["#CC6677", "#882255"]
obs_alpha = 1.0
obs_cmap = ["0.8", "0.7"]
# obs_cmap = ["#88CCEE", "#44AA99"]
hist_cmap = ["#a6cee3", "#1f78b4"]
cartopy_crs = ccrs.NorthPolarStereo(central_longitude=-45, true_scale_latitude=70, globe=None)


if __name__ == "__main__":
    __spec__ = None  # type: ignore

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
        default="data/itslive/ITS_LIVE_GRE_G0240_2018.nc",
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
        "--filter_var",
        help="""Filter variable. Default='retreat'""",
        type=str,
        choices=["land_ice_area_fraction_retreat", "dhdt", "speed", "grace"],
        default="land_ice_area_fraction_retreat",
    )
    parser.add_argument(
        "--data_dir",
        help="""Observational uncertainty multiplier. Default=3""",
        type=str,
        default="land_ice_are_fraction_retreat",
    )
    parser.add_argument("--n_jobs", help="""Number of parallel jobs.""", type=int, default=4)
    parser.add_argument(
        "--parallel",
        help="""Open dataset in parallel. Default=False.""",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--resampling_frequency",
        help="""Resampling data to resampling_frequency for importance sampling. Default is "YS".""",
        type=str,
        default="YS",
    )
    parser.add_argument(
        "--engine",
        help="""Engine for xarray. Default="h5netcdf".""",
        type=str,
        default="h5netcdf",
    )
    parser.add_argument(
        "FILES",
        help="""Ensemble netCDF files.""",
        nargs="*",
    )

    options = parser.parse_args()
    engine = options.engine
    spatial_files = sorted(options.FILES)
    filter_var = options.filter_var
    fudge_factor = options.fudge_factor
    parallel = options.parallel
    input_data_dir = options.data_dir
    resampling_frequency = options.resampling_frequency
    outlier_variable = options.outlier_variable
    ragis_config_file = Path(str(files("pism_ragis.data").joinpath("ragis_config.toml")))
    ragis_config = toml.load(ragis_config_file)
    config = json.loads(json.dumps(ragis_config))
    params_short_dict = config["Parameters"]
    params = list(params_short_dict.keys())

    result_dir = Path(options.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    client = Client()
    print(f"Open client in browser: {client.dashboard_link}")

    start = time.time()

    filter_range = ["1980", "2018"]
    obs_mean_var = "land_ice_area_fraction_retreat"
    obs_std_var = "land_ice_area_fraction_retreat_uncertainty"
    sim_var = "land_ice_area_fraction_retreat"
    sum_dims = ["y", "x", "time"]
    obs_file = input_data_dir + "/front_retreat/pism_g450m_frontretreat_calfin_1972_2019_YE.nc"
    log_likelihood = log_jaccard_score_xr
    prepare_input = prepare_liafr

    crs = "EPSG:3413"
    basins_file = input_data_dir + "/basins/glaciers_ext.gpkg"
    basins = gp.read_file(basins_file).to_crs(crs)
    basins = basins[basins["GL_TYPE"] == "TW"]
    basins = basins[basins["SUBREGION1"] == "CW"]
    # basins = basins[basins["NAME"] == "JAKOBSHAVN_ISBRAE"]

    params = ["calving.vonmises_calving.sigma_max"]

    print("Loading ensemble.")
    time_decoder = xr.coders.CFDatetimeCoder(use_cftime=False)

    simulated = xr.open_mfdataset(
        spatial_files,
        preprocess=preprocess_config,
        parallel=True,
        decode_timedelta=True,
        decode_times=time_decoder,
        engine=engine,
    )

    observed = xr.open_mfdataset(obs_file, chunks="auto")

    simulated = simulated.sel({"time": slice(*filter_range)})
    observed = observed.sel({"time": slice(*filter_range)})

    stats = simulated[["pism_config"]].isel(time=0).sel(pism_config_axis=params).astype(float)

    obs, sim = prepare_input(
        observed,
        simulated,
        obs_mean_var,
        obs_std_var,
        sim_var,
    )

    obs = obs.chunk({"time": -1, "x": 1000, "y": 1000})
    sim = sim.chunk({"time": -1, "exp_id": 1, "x": 1000, "y": 1000})
    futures = []
    results = []
    print("=" * 80)
    print("Processing glaciers")
    print("=" * 80)
    for b, basin in basins.iterrows():
        try:
            print(basin.NAME)
            obs_glacier = (
                obs.rio.set_spatial_dims(x_dim="x", y_dim="y")
                .rio.write_crs(crs)
                .rio.clip([basin.geometry], drop=True)
                .expand_dims({"basin": [basin["NAME"]]})
            )
            sim_glacier = (
                sim.rio.set_spatial_dims(x_dim="x", y_dim="y")
                .rio.write_crs(crs)
                .rio.clip([basin.geometry], drop=True)
                .expand_dims({"basin": [basin["NAME"]]})
            )

            f = importance_sampling(
                observed=obs_glacier,
                simulated=sim_glacier,
                obs_mean_var=obs_mean_var,
                obs_std_var=obs_std_var,
                sim_var=sim_var,
                n_samples=sim_glacier.sizes["exp_id"],
                log_likelihood=log_likelihood,
                fudge_factor=fudge_factor,
                sum_dims=sum_dims,
                compute=False,
            )
            fut = client.compute(f)
            futures.append(fut)
        except:
            print(f"Not processed: {basin.NAME}")

    progress(futures)  # <-- now shows live progress
    results = client.gather(futures)
    end = time.time()
    time_elapsed = end - start
    print(f"Time elapsed {time_elapsed:.0f}s")

    result = xr.concat(results, dim="basin")

    importance_sampled_ids = result["exp_id_sampled"]
    simulated_posterior = stats.sel(exp_id=importance_sampled_ids).sel(pism_config_axis=params)
    posterior_config = simulated_posterior.pism_config
    df = config_to_dataframe(posterior_config).drop(columns=["exp_id", "aux_id", "time", "spatial_ref"])
    m = df.groupby(by="basin").median().reset_index()

    client.close()
