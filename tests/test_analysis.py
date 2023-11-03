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
# Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA

import pandas as pd
import xarray as xr
from pandas.testing import assert_frame_equal

from pism_ragis.analysis import prepare_df, sensitivity_analysis
from pism_ragis.processing import preprocess_nc

seed = 42

ens_vars_dict = {
    "grounding_line_flux": "grounding_line_flux (Gt year-1)",
    "limnsw": "limnsw (kg)",
}


def test_prepare_df():
    """Test preparing a DataFrame"""
    true_infile = "tests/data/test_scalar_YM.parquet"
    for suffix in ["parquet"]:
        test_infile = f"tests/data/test_scalar_YM.{suffix}"
        df = prepare_df(test_infile)
        df_true = pd.read_parquet(true_infile)

        assert_frame_equal(df, df_true)


def test_sensitivity_analysis():
    X_df = (
        pd.read_parquet("tests/data/test_scalar_YM.parquet")
        .drop(columns=["Year", "resolution_m"])
        .sort_values(by=["time", "id"])
    )
    Y_true_df = pd.read_parquet("tests/data/test_sensitivity.parquet")
    ensemble_file = "tests/data/gris_ragis_ocean_simple_lhs_50_w_posterior.csv"
    for n_jobs in [1, 2]:
        Y_df = sensitivity_analysis(
            X_df, ensemble_file=ensemble_file, n_jobs=n_jobs, seed=seed
        )
        assert_frame_equal(Y_df, Y_true_df, atol=1e-1, rtol=1e-6)


def test_sensitivity_analysis_from_xarray():
    ds = xr.open_mfdataset(
        "tests/data/ts_gris_g1200m_v2023_RAGIS_id_*_1980-1-1_2020-1-1.nc",
        combine="nested",
        concat_dim="id",
        preprocess=preprocess_nc,
        parallel=True,
    )
    ens = (
        ds.sel(time=slice("1980-01-01", "1982-01-01"))[ens_vars_dict.keys()]
        .resample(time="1AS")
        .mean()
    )
    X_df = (
        ens.to_dataframe()
        .rename(columns=ens_vars_dict)
        .reset_index()
        .dropna()
        .sort_values(by=["time", "id"])
        .reset_index(drop=True)
    )

    Y_true_df = pd.read_parquet("tests/data/test_sensitivity.parquet")
    ensemble_file = "tests/data/gris_ragis_ocean_simple_lhs_50_w_posterior.csv"
    for n_jobs in [1, 2]:
        Y_df = sensitivity_analysis(
            X_df,
            ensemble_file=ensemble_file,
            n_jobs=n_jobs,
            seed=seed,
        )
        assert_frame_equal(Y_df, Y_true_df, atol=1e-1, rtol=1e-6)
