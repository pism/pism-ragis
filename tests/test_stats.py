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

"""
Test stats module.
"""

from glob import glob

import pandas as pd
from pandas.testing import assert_frame_equal

from pism_ragis.stats import run_stats


def test_run_stats():
    """
    Test retrieving run_stats from PISM output file and return pd.DataFrame.
    """
    infiles = sorted(
        glob("tests/data/ts_gris_g1200m_v2023_RAGIS_id_*_1980-1-1_2020-1-1.nc")
    )
    df = run_stats(infiles)
    df_true = pd.read_csv("tests/data/test_run_stats.csv")

    assert_frame_equal(df, df_true)
