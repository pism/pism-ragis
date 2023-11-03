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
Module for data processing
"""

import contextlib
import os
import pathlib
import re
import shutil
import time
from datetime import datetime
from typing import Union

import joblib
import numpy as np
import pandas as pd
import xarray as xr
from joblib import Parallel, delayed
from tqdm.auto import tqdm

kg2cmsle = 1 / 1e12 * 1.0 / 362.5 / 10.0
gt2cmsle = 1 / 362.5 / 10.0


def preprocess_nc(ds):
    """
    Add experiment 'id'
    """
    m_id_re = re.search("id_(.+?)_", ds.encoding["source"])
    ds.expand_dims("id")
    assert m_id_re is not None
    m_id: Union[str, int]
    try:
        m_id = int(m_id_re.group(1))
    except:
        m_id = str(m_id_re.group(1))
    ds["id"] = m_id
    return ds


@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""

    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        """TQDM Callback"""

        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


def convert_netcdf_to_dataframe(
    infiles: list,
    resample: Union[str, None] = None,
    n_jobs: int = 4,
    add_vars: bool = True,
    norm_year: Union[None, float] = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Convert list of netCDF files to Pandas DataFrame.


    """
    n_files = len(infiles)
    print("Converting netcdf files to pandas.DataFrame")
    print("-------------------------------------------")
    start_time = time.perf_counter()
    with tqdm_joblib(tqdm(desc="Processing files", total=n_files)) as progress_bar:
        result = Parallel(n_jobs=n_jobs)(
            delayed(ncfile2dataframe)(infile, resample, add_vars, norm_year, verbose)
            for infile in infiles
        )
        del progress_bar
    finish_time = time.perf_counter()
    time_elapsed = finish_time - start_time
    print(f"Conversion finished in {time_elapsed:.0f} seconds")
    print("-------------------------------------------")

    df = pd.concat(result)

    return df.sort_values(by=["time", "id"]).reset_index(drop=True)


def ncfile2dataframe(
    infile: Union[str, pathlib.Path],
    resample: Union[str, None] = None,
    add_vars: bool = True,
    norm_year: Union[None, float] = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """Convert netCDF file to pandas.DataFrame"""

    if isinstance(infile, pathlib.Path):
        assert infile.exists()
    else:
        assert os.path.isfile(infile)
    if verbose:
        print(f"Opening {infile}")
    with xr.open_dataset(infile) as ds:
        if resample == "monthly":
            ds = ds.resample(time="1MS").mean()
        elif resample == "yearly":
            ds = ds.resample(time="1YS").mean()
        else:
            pass
        if isinstance(infile, pathlib.Path):
            m_id_re = re.search("id_(.+?)_", str(infile))
        else:
            m_id_re = re.search("id_(.+?)_", infile)
        assert m_id_re is not None
        m_id: Union[str, int]
        try:
            m_id = int(m_id_re.group(1))
        except:
            m_id = str(m_id_re.group(1))

        if isinstance(infile, pathlib.Path):
            m_dx_re = re.search("gris_g(.+?)m", str(infile))
        else:
            m_dx_re = re.search("gris_g(.+?)m", infile)
        assert m_dx_re is not None
        m_dx = int(m_dx_re.group(1))
        datetimeindex = ds.indexes["time"]
        years = [to_decimal_year(x.to_pydatetime()) for x in datetimeindex]
        nt = len(datetimeindex)
        id_S = pd.Series(data=np.repeat(m_id, nt), index=datetimeindex, name="id")
        S = [id_S]
        for m_var in ds.data_vars:
            if m_var not in (
                "time_bounds",
                "time_bnds",
                "timestamp",
                "run_stats",
                "pism_config",
            ):
                if hasattr(ds[m_var], "units"):
                    m_units = ds[m_var].units
                    m_S_name = f"{m_var} ({m_units})"
                else:
                    m_units = ""
                    m_S_name = f"{m_var}"
                data = np.squeeze(ds[m_var].values)
                m_S = pd.Series(data=data, index=datetimeindex, name=m_S_name)
                S.append(m_S)
        m_Y = pd.Series(data=years, index=datetimeindex, name="Year")
        S.append(m_Y)
        df = pd.concat(S, axis=1).reset_index()
        df["resolution_m"] = m_dx

        if add_vars:
            df = add_vars_to_dataframe(df)

        if norm_year:
            norm_year_idx = np.nonzero(np.array(years) == norm_year)[0][0]
            df["limnsw (kg)"] -= df["limnsw (kg)"][norm_year_idx]
            if add_vars:
                df["Cumulative ice sheet mass change (Gt)"] -= df[
                    "Cumulative ice sheet mass change (Gt)"
                ][norm_year_idx]
                df["SLE (cm)"] -= df["SLE (cm)"][norm_year_idx]

    return df


def add_vars_to_dataframe(df: pd.DataFrame):
    """Add additional variables to DataFrame"""

    if "limnsw (kg)" in df.columns:
        df["Cumulative ice sheet mass change (Gt)"] = (
            df["limnsw (kg)"] - df["limnsw (kg)"][0]
        ) / 1e12
        df["SLE (cm)"] = -df["Cumulative ice sheet mass change (Gt)"] * gt2cmsle
        if "grounding_line_flux (Gt year-1)" in df.columns:
            df["Rate of ice discharge (Gt/yr)"] = -df["grounding_line_flux (Gt year-1)"]
        if "tendency_of_ice_mass_due_to_surface_mass_flux (Gt year-1)" in df.columns:
            df["Rate of surface mass balance (Gt/yr)"] = df[
                "tendency_of_ice_mass_due_to_surface_mass_flux (Gt year-1)"
            ]
    return df


def to_decimal_year(date):
    """Convert datetime date to decimal year"""
    year = date.year
    start_of_this_year = datetime(year=year, month=1, day=1)
    start_of_next_year = datetime(year=year + 1, month=1, day=1)
    year_elapsed = (date - start_of_this_year).total_seconds()
    year_duration = (start_of_next_year - start_of_this_year).total_seconds()
    fraction = year_elapsed / year_duration

    return date.year + fraction


def check_file(
    infile: Union[str, pathlib.Path], norm_year: Union[None, float] = None
) -> bool:
    """Check netCDF file"""
    with xr.open_dataset(infile) as ds:
        is_ok: bool = False
        if "time" in ds.indexes:
            datetimeindex = ds.indexes["time"]
            years = np.array(
                [to_decimal_year(x.to_pydatetime()) for x in datetimeindex]
            )
            monotonically_increasing = np.all(
                years.reshape(1, -1)[:, 1:] >= years.reshape(1, -1)[:, :-1], axis=1
            )[0]
            if norm_year:
                if (years[-1] >= norm_year) and monotonically_increasing:
                    is_ok = True
            else:
                print(
                    f"{infile} time-series too short or not monotonically-increasing."
                )
        return is_ok


def check_paleo_file(
    infile: Union[str, pathlib.Path], norm_year: Union[None, float] = None
) -> bool:
    """Check netCDF file"""
    with xr.open_dataset(infile) as ds:
        is_ok: bool = False
        if "time" in ds.indexes:
            datetimeindex = ds.indexes["time"]
            years = datetimeindex.year
            monotonically_increasing = np.all(
                years.reshape(1, -1)[:, 1:] >= years.reshape(1, -1)[:, :-1], axis=1
            )[0]
            if norm_year:
                if (years[-1] >= norm_year) and monotonically_increasing:
                    is_ok = True
            else:
                print(
                    f"{infile} time-series too short or not monotonically-increasing."
                )
        return is_ok


def copy_file(
    infile: Union[str, pathlib.Path], outdir: Union[str, pathlib.Path]
) -> None:
    """
    Copy infile to outdir
    """
    if infile is not pathlib.Path:
        in_path = pathlib.Path(infile)
    else:
        in_path = infile  # type: ignore
    if outdir is not pathlib.Path:
        out_path = pathlib.Path(outdir)
    else:
        out_path = outdir  # type: ignore
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    shutil.copy(in_path, out_path)
