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
Tests for procesing module
"""

from glob import glob

import pandas as pd
from pandas.testing import assert_frame_equal

from pism_ragis.processing import convert_netcdf_to_dataframe, ncfile2dataframe


def test_ncfiletodataframe_basic():
    """
    Reading in a netCDF file and return a pandas.DataFrame:
      - add variables
      - resampling
    """
    infile = "tests/data/ts_gris_g1200m_v2023_RAGIS_id_0_1980-1-1_2020-1-1.nc"

    df_parquet_true = pd.read_parquet("tests/data/test_scalar_file.parquet")
    df_csv_true = pd.read_csv(
        "tests/data/test_scalar_file.csv",
        index_col=0,
        parse_dates=["time"],
    )

    df = ncfile2dataframe(infile, add_vars=False)
    assert_frame_equal(df, df_parquet_true)
    assert_frame_equal(df, df_csv_true)


def test_ncfiletodataframe_norm():
    """
    Reading in a netCDF file and return a pandas.DataFrame:
      - normalization by year
    """
    infile = "tests/data/ts_gris_g1200m_v2023_RAGIS_id_0_1980-1-1_2020-1-1.nc"

    df_parquet_true = pd.read_parquet("tests/data/test_scalar_file_norm_1992.parquet")
    df = ncfile2dataframe(infile, add_vars=False, norm_year=1992.0)
    assert_frame_equal(df, df_parquet_true)


def test_ncfiletodataframe_resampling():
    """
    Reading in a netCDF file and return a pandas.DataFrame:
      - resampling
    """
    infile = "tests/data/ts_gris_g1200m_v2023_RAGIS_id_0_1980-1-1_2020-1-1.nc"

    df_parquet_true = pd.read_parquet("tests/data/test_scalar_file_YM.parquet")
    df_csv_true = pd.read_csv(
        "tests/data/test_scalar_file_YM.csv",
        index_col=0,
        parse_dates=["time"],
    )

    df = ncfile2dataframe(infile, resample="yearly", add_vars=False)
    assert_frame_equal(df, df_parquet_true)
    assert_frame_equal(df, df_csv_true)


def test_ncfiletodataframe_addvars():
    """
    Reading in a netCDF file and return a pandas.DataFrame:
      - add variables
    """
    infile = "tests/data/ts_gris_g1200m_v2023_RAGIS_id_0_1980-1-1_2020-1-1.nc"

    df_parquet_true = pd.read_parquet("tests/data/test_scalar_file_add_vars.parquet")
    df = ncfile2dataframe(infile, add_vars=True)
    assert_frame_equal(df, df_parquet_true)


def test_ncfiletodataframe_addvars_norm():
    """
    Reading in a netCDF file and return a pandas.DataFrame:
      - add variables
    """
    infile = "tests/data/ts_gris_g1200m_v2023_RAGIS_id_0_1980-1-1_2020-1-1.nc"

    df_parquet_true = pd.read_parquet(
        "tests/data/test_scalar_file_add_vars_norm_1992.parquet"
    )
    df = ncfile2dataframe(infile, add_vars=True, norm_year=1992)
    assert_frame_equal(df, df_parquet_true)


def test_convert_netcdf_to_dataframe_basic():
    """
    Reading in a list of netCDF files and return a pandas.DataFrame:
    """

    df_parquet_true = pd.read_parquet("tests/data/test_scalar.parquet")
    df_csv_true = pd.read_csv(
        "tests/data/test_scalar.csv",
        index_col=0,
        parse_dates=["time"],
    )
    infiles = glob("tests/data/ts_gris_g1200m_v2023_RAGIS_id_*_1980-1-1_2020-1-1.nc")
    df = convert_netcdf_to_dataframe(infiles, add_vars=False)
    assert_frame_equal(df, df_parquet_true)
    assert_frame_equal(df, df_csv_true)


def test_convert_netcdf_to_dataframe_resampling():
    """
    Reading in a list of netCDF files and return a pandas.DataFrame:
      - resampling
    """

    df_parquet_true = pd.read_parquet("tests/data/test_scalar_YM.parquet")
    df_csv_true = pd.read_csv(
        "tests/data/test_scalar_YM.csv",
        index_col=0,
        parse_dates=["time"],
    )
    infiles = glob("tests/data/ts_gris_g1200m_v2023_RAGIS_id_*_1980-1-1_1982-1-1.nc")
    df = convert_netcdf_to_dataframe(infiles, resample="yearly", add_vars=False)
    assert_frame_equal(df, df_parquet_true)
    assert_frame_equal(df, df_csv_true)
