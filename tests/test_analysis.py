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
from numpy.testing import assert_array_almost_equal
from pandas.testing import assert_frame_equal

from pismragis.analysis import prepare_df, sensitivity_analysis


def test_prepare_df():
    true_infile = "tests/data/test_scalar_YM.parquet"
    for suffix in ["parquet"]:
        test_infile = f"tests/data/test_scalar_YM.{suffix}"
        df = prepare_df(test_infile)
        df_true = pd.read_parquet(true_infile)

        assert_frame_equal(df, df_true)


def test_sensitivity_analysis():
    sens_vars = [
        "vcm",
        "gamma_T",
        "thickness_calving_threshold",
        "ocean_file",
        "sia_e",
        "ssa_n",
        "pseudo_plastic_q",
        "till_effective_fraction_overburden",
        "phi_min",
        "phi_max",
        "z_min",
        "z_max",
    ]
    X_df = pd.read_parquet("tests/data/test_scalar_YM.parquet")
    Y_true = pd.read_parquet("tests/data/test_sensitivity.parquet")[sens_vars].mean()
    ensemble_file = "tests/data/gris_ragis_ocean_simple_lhs_50_w_posterior.csv"
    for n_jobs in [1, 2, 4]:
        Y = sensitivity_analysis(X_df, ensemble_file=ensemble_file, n_jobs=n_jobs)[
            sens_vars
        ].mean()
        assert_array_almost_equal(Y, Y_true, decimal=True)


def test_sensitivity_analysis_from_xarray():
    sens_vars = [
        "vcm",
        "gamma_T",
        "thickness_calving_threshold",
        "ocean_file",
        "sia_e",
        "ssa_n",
        "pseudo_plastic_q",
        "till_effective_fraction_overburden",
        "phi_min",
        "phi_max",
        "z_min",
        "z_max",
    ]
    ds = xr.open_mfdataset(
        "tests/data/ts_gris_g1200m_v2023_RAGIS_id_*.nc",
        combine="nested",
        concat_dim="id",
        parallel=True,
    )
    ens_vars = "grounding_line_flux"
    X = (
        ds.sel(time=slice("1980-01-01", "1983-01-01"))[ens_var]
        .resample(time="1AS")
        .mean()
    )
    X_df = ens.to_dataframe().reset_index().dropna()

    Y_true = pd.read_parquet("tests/data/test_sensitivity.parquet")[sens_vars].mean()
    ensemble_file = "tests/data/gris_ragis_ocean_simple_lhs_50_w_posterior.csv"
    for n_jobs in [1, 2, 4]:
        Y = sensitivity_analysis(X_df, ensemble_file=ensemble_file, n_jobs=n_jobs)[
            sens_vars
        ].mean()
        assert_array_almost_equal(Y, Y_true, decimal=True)
