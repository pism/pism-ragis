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

import contextlib
import os
import re
import time
from datetime import datetime
from typing import Union

import joblib
import numpy as np
import pandas as pd
import xarray as xr
from joblib import Parallel, delayed
from tqdm import tqdm


@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""

    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
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


def ncfile2dataframe(infile, resample: Union[str, None] = None) -> pd.DataFrame:

    if os.path.isfile(infile):
        with xr.open_dataset(infile) as ds:
            if resample == "monthly":
                ds = ds.resample(time="1MS").mean()
            elif resample == "yearly":
                ds = ds.resample(time="1YS").mean()
            else:
                pass
            m_id_re = re.search("id_(.+?)_", infile)
            assert m_id_re is not None
            m_id = int(m_id_re.group(1))
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

        return pd.concat(S, axis=1).reset_index()


def convert_netcdf_to_dataframe(
    infiles,
    resample: Union[str, None] = None,
    n_jobs: int = 4,
) -> pd.DataFrame:
    n_files = len(infiles)

    start_time = time.perf_counter()
    with tqdm_joblib(tqdm(desc="Processing files", total=n_files)) as progress_bar:
        result = Parallel(n_jobs=n_jobs)(
            delayed(ncfile2dataframe)(infile, resample) for infile in infiles
        )
        del progress_bar
    finish_time = time.perf_counter()
    time_elapsed = finish_time - start_time
    print(f"Program finished in {time_elapsed:.0f} seconds")

    df = pd.concat(result)
    return df


def to_decimal_year(date):

    year = date.year
    start_of_this_year = datetime(year=year, month=1, day=1)
    start_of_next_year = datetime(year=year + 1, month=1, day=1)
    year_elapsed = (date - start_of_this_year).total_seconds()
    year_duration = (start_of_next_year - start_of_this_year).total_seconds()
    fraction = year_elapsed / year_duration

    return date.year + fraction
