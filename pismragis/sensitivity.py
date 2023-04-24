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

import pathlib

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from pandas.api.types import is_string_dtype
from SALib.analyze import delta
from tqdm import tqdm

from pismragis.processing import tqdm_joblib


def prepare_df(ifile: str):
    suffix = pathlib.Path(ifile).suffix
    if suffix in (".csv", ".gz"):
        df = pd.read_csv(ifile, parse_dates=["time"])
    elif suffix in (".parquet"):
        df = pd.read_parquet(ifile)
    else:
        print(f"{suffix} not recognized")

    return df


def sensitivity_analysis(
    df: pd.DataFrame,
    ensemble_file: str,
    calc_variables: list = ["grounding_line_flux (Gt year-1)", "limnsw (kg)"],
    n_jobs: int = 4,
    sensitivity_indices: list = ["delta", "S1"],
):

    # remove True/False
    id_df = (pd.read_csv(ensemble_file) * 1).replace(np.nan, 0)

    param_names = id_df.drop(columns="id").columns.values.tolist()
    for k, col in id_df.items():
        if is_string_dtype(col):
            u = col.unique()
            u.sort()
            v = [k for k, v in enumerate(u)]
            col.replace(to_replace=dict(zip(u, v)), inplace=True)
    # Define a salib "problem"
    problem = {
        "num_vars": len(id_df.drop(columns="id").columns.values),
        "names": param_names,  # Parameter names
        "bounds": zip(
            id_df.drop(columns="id").min().values,
            id_df.drop(columns="id").max().values,
        ),  # Parameter bounds
    }

    df = pd.merge(id_df, df, on="id")
    # filter out dates with only 1 experiment, e.g., due to
    # CDO changing the mid-point time when running averaging
    df = pd.concat([x for _, x in df.groupby(by="time") if len(x) > 1])
    n_dates = len(df["time"].unique())
    if n_jobs == 1:
        Sobol_dfs = []
        for m_date, s_df in df.groupby(by="time"):
            Sobol_dfs.append(
                compute_sensitivity_indices(
                    m_date,
                    s_df,
                    id_df,
                    problem,
                    calc_variables,
                    sensitivity_indices=sensitivity_indices,
                )
            )
    else:
        with tqdm_joblib(tqdm(desc="Processing file", total=n_dates)) as progress_bar:
            Sobol_dfs = Parallel(n_jobs=n_jobs)(
                delayed(compute_sensitivity_indices)(
                    m_date,
                    s_df,
                    id_df,
                    problem,
                    calc_variables,
                    sensitivity_indices=sensitivity_indices,
                )
                for m_date, s_df in df.groupby(by="time")
            )
            del progress_bar

    Sobol_df = pd.concat(Sobol_dfs)
    Sobol_df.reset_index(inplace=True, drop=True)
    return Sobol_df


def compute_sensitivity_indices(
    m_date,
    s_df,
    id_df,
    problem,
    calc_variables,
    sensitivity_indices=["delta", "S1"],
    verbose: bool = False,
):
    print(f"Processing {m_date}")
    missing_ids = list(set(id_df["id"]).difference(s_df["id"]))
    if missing_ids:
        if verbose:
            print(
                "The following simulation ids are missing:\n   {}".format(missing_ids)
            )

        id_df_missing_removed = id_df[~id_df["id"].isin(missing_ids)]
        params = np.array(
            id_df_missing_removed.drop(columns="id").values, dtype=np.float32
        )
    else:
        params = np.array(id_df.drop(columns="id").values, dtype=np.float32)
    Sobol_dfs = []
    for calc_variable in calc_variables:
        response_matrix = s_df[calc_variable].values
        Si = delta.analyze(
            problem,
            params,
            response_matrix,
            num_resamples=100,
            print_to_console=False,
        )
        Si_df = Si.to_df()

        s_dfs = []
        for s_index in sensitivity_indices:
            m_df = pd.DataFrame(
                data=Si_df[s_index].values.reshape(1, -1),
                columns=Si_df.transpose().columns,
            )
            m_df["Date"] = m_date
            m_df["Si"] = s_index
            m_df["Variable"] = calc_variable

            m_conf_df = pd.DataFrame(
                data=Si_df[s_index + "_conf"].values.reshape(1, -1),
                columns=Si_df.transpose().columns,
            )
            m_conf_df["Date"] = m_date
            m_conf_df["Si"] = s_index + "_conf"
            m_conf_df["Variable"] = calc_variable
            s_dfs.append(pd.concat([m_df, m_conf_df]))

        a_df = pd.concat(s_dfs)
        Sobol_dfs.append(a_df)
    return pd.concat(Sobol_dfs)
