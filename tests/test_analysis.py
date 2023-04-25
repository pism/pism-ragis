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

from pismragis.analysis import sensitivity_analysis


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
