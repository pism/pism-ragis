#!/usr/bin/env python

# pylint: disable=unsupported-assignment-operation

"""
Uncertainty quantification using Latin Hypercube Sampling or Sobol Sequences.
"""

from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
from pyDOE2 import lhs
from SALib.sample import sobol
from scipy.stats.distributions import randint, uniform

rcms: Dict[int, str] = {
    0: "HIRHAM5-monthly-ERA5_1975_2021.nc",
    1: "MARv3.14-monthly-ERA5_1940_2023.nc",
    2: "RACMO2.3p2_ERA5_FGRN055_1940_2023.nc",
}
gcms: Dict[int, str] = {
    0: "ACCESS1-3_rcp85",
    1: "CNRM-CM6_ssp126",
    2: "CNRM-ESM2_ssp585",
    3: "CSIRO-Mk3.6_rcp85",
    4: "HadGEM2-ES_rcp85",
    5: "IPSL-CM5-MR_rcp85",
    6: "MIROC-ESM-CHEM_rcp26",
    7: "NorESM1-M_rcp85",
    8: "UKESM1-CM6_ssp585",
}

initialstates: Dict[int, str] = {
    0: "gris_g900m_v2023_GIMP_id_CTRL_0_25.nc",
    1: "gris_g900m_v2023_RAGIS_id_CTRL_0_25.nc",
}

retreatfiles: Dict[int, str | bool] = {
    0: "pism_g450m_frontretreat_calfin_1972_2019.nc",
    1: "",
}

slidinglaw: Dict[int, str] = {
    0: "pseudo_plastic",
    1: "regularized_coulomb",
}

sb_dict = {0: "ssa+sia", 1: "blatter"}

dists: Dict[str, Any] = {
    "ragis": {
        "uq": {
            "calving.vonmises_calving.sigma_max": uniform(loc=300_000, scale=500_000),
            "ocean.th.gamma_T": uniform(loc=0.75e-4, scale=0.75e-4),
            "ocean_file": randint(0, len(gcms)),
            "climate_file": randint(0, len(rcms)),
            "frontal_melt.routing.parameter_a": uniform(loc=2.4e-4, scale=1.2e-4),
            "frontal_melt.routing.parameter_b": uniform(loc=1.0, scale=0.70),
            "frontal_melt.routing.power_alpha": uniform(loc=0.3, scale=0.55),
            "frontal_melt.routing.power_beta": uniform(loc=1.1, scale=0.7),
            "prescribed_retreat_file": randint(0, len(retreatfiles)),
        },
        "default_values": {
            "basal_resistance.pseudo_plastic.q": 0.7508221,
            "basal_yield_stress.mohr_coulomb.till_effective_fraction_overburden": 0.01845403,
            "basal_yield_stress.mohr_coulomb.topg_to_phi.phi_max": 42.79528,
            "basal_yield_stress.mohr_coulomb.topg_to_phi.phi_min": 7.193718,
            "basal_yield_stress.mohr_coulomb.topg_to_phi.topg_max": 243.8239,
            "basal_yield_stress.mohr_coulomb.topg_to_phi.topg_min": -369.6359,
            "calving.thickness_calving.threshold": 50,
            "calving.vonmises_calving.sigma_max": 750000,
            "climate": "given_smb",
            "climate_file": "HIRHAM5-monthly-ERA5_1975_2021.nc",
            "fractures": "false",
            "frontal_melt": "routing",
            "hydrology": "routing",
            "ocean.models": "th",
            "ocean.th.gamma_T": 0.0001,
            "ocean_file": "MAR3.9_CNRM-ESM2_ssp585_ocean_1960-2100_v4.nc",
            "sliding_law": "pseudo_plastic",
            "stress_balance.sia.enhancement_factor": 2.608046,
            "stress_balance.ssa.Glen_exponent": 3.309718,
        },
    },
    "calving-calib": {
        "uq": {
            "calving.vonmises_calving.sigma_max": uniform(loc=250_000, scale=750_000),
        },
        "default_values": {
            "basal_resistance.pseudo_plastic.q": 0.7508221,
            "basal_yield_stress.mohr_coulomb.till_effective_fraction_overburden": 0.01845403,
            "basal_yield_stress.mohr_coulomb.topg_to_phi.phi_max": 42.79528,
            "basal_yield_stress.mohr_coulomb.topg_to_phi.phi_min": 7.193718,
            "basal_yield_stress.mohr_coulomb.topg_to_phi.topg_max": 243.8239,
            "basal_yield_stress.mohr_coulomb.topg_to_phi.topg_min": -369.6359,
            "calving.thickness_calving.threshold": 50,
            "calving.vonmises_calving.sigma_max": 750000,
            "climate": "given_smb",
            "climate_file": "HIRHAM5-monthly-ERA5_1975_2021.nc",
            "fractures": "false",
            "frontal_melt": "routing",
            "hydrology": "routing",
            "ocean.models": "th",
            "ocean.th.gamma_T": 0.0001,
            "ocean_file": "MAR3.9_CNRM-ESM2_ssp585_ocean_1960-2100_v4.nc",
            "prescribed_retreat_file": "",
            "sliding_law": "pseudo_plastic",
            "stress_balance.sia.enhancement_factor": 2.608046,
            "stress_balance.ssa.Glen_exponent": 3.309718,
        },
    },
    "flow": {
        "uq": {
            "basal_resistance.pseudo_plastic.q": uniform(0.25, 0.75),
            "basal_yield_stress.mohr_coulomb.till_effective_fraction_overburden": uniform(
                loc=0.01, scale=0.03
            ),
            "basal_yield_stress.mohr_coulomb.topg_to_phi.phi_max": uniform(
                loc=40.0, scale=20.0
            ),
            "basal_yield_stress.mohr_coulomb.topg_to_phi.phi_min": uniform(
                loc=5.0, scale=30.0
            ),
            "basal_yield_stress.mohr_coulomb.topg_to_phi.topg_min": uniform(
                loc=-1000, scale=1000
            ),
            "basal_yield_stress.mohr_coulomb.topg_to_phi.topg_max": uniform(
                loc=0, scale=1500
            ),
            "stress_balance.sia.enhancement_factor": uniform(loc=1.0, scale=3.0),
            "stress_balance.ssa.Glen_exponent": uniform(loc=2.75, scale=0.75),
            "stress_balance.sia.Glen_exponent": uniform(loc=1.0, scale=3.0),
        },
        "default_values": {
            "calving.thickness_calving.threshold": 50,
            "calving.vonmises_calving.sigma_max": 750000,
            "climate": "given_smb",
            "climate_file": "HIRHAM5-monthly-ERA5_1975_2021.nc",
            "fractures": "false",
            "frontal_melt": "off",
            "ocean.th.gamma_T": 0.0001,
            "hydrology": "diffuse",
            "ocean.models": "const",
            "ocean_file": None,
            "prescribed_retreat_file": "pism_g450m_frontretreat_calfin_2007.nc",
            "sliding_law": "pseudo_plastic",
        },
    },
    "dem": {
        "uq": {
            "calving.vonmises_calving.sigma_max": uniform(loc=350_000, scale=350_000),
            "delta_T": uniform(-4, 8.0),
            "frac_P": uniform(0.9, 0.4),
            "surface.pdd.factor_ice": uniform(loc=1, scale=9),
            "surface.pdd.factor_snow": uniform(loc=0.5, scale=4.5),
            "surface.pdd.std_dev.value": uniform(loc=1, scale=5),
        },
        "default_values": {
            "climate": "given_pdd_delta",
            "hydrology": "diffuse",
            "ocean": "constant",
            "climate_file": "RACMO2.3p2_ERA5_FGRN055_1940_2023.nc",
            "salinity": 34,
            "fractures": "false",
            "frontal_melt": "off",
            "ocean.th.gamma_T": 0.0001,
            "ocean.models": "const",
            "ocean_file": "MAR3.9_CNRM-ESM2_ssp585_ocean_1960-2100_v4.nc",
            "prescribed_retreat_file": "pism_g450m_frontretreat_calfin_1972_2019.nc",
            "surface.pdd.refreeze": 0.6,
            "sliding_law": "pseudo_plastic",
        },
    },
}

parser = ArgumentParser()
parser.description = "Generate UQ using Latin Hypercube or Sobol Sequences."
parser.add_argument(
    "-s",
    "--n_samples",
    dest="n_samples",
    type=int,
    help="""number of samples to draw. default=16.""",
    default=16,
)
parser.add_argument(
    "-d",
    "--distribution",
    dest="distribution",
    choices=dists.keys(),
    help="""Choose set.""",
    default="ragis",
)
parser.add_argument(
    "--second_order",
    action="store_true",
    help="""Second order interactions.""",
    default=False,
)
parser.add_argument(
    "-m",
    "--method",
    dest="method",
    type=str,
    choices=["lhs", "sobol"],
    help="""Sampling method, Latin Hypercube or Sobol. default=lhs.""",
    default="lhs",
)
parser.add_argument(
    "--posterior_file",
    help="Posterior predictive parameter file",
    default=None,
)
parser.add_argument(
    "OUTFILE",
    nargs=1,
    help="Ouput file (CSV)",
    default="velocity_calibration_samples.csv",
)
options = parser.parse_args()
n_draw_samples = options.n_samples
calc_second_order = options.second_order
method = options.method
outfile = Path(options.OUTFILE[-1])
distribution_name = options.distribution
posterior_file = options.posterior_file

print(f"\nDrawing {n_draw_samples} samples from distribution set {distribution_name}")
distributions = dists[distribution_name]["uq"]

problem = {
    "num_vars": len(distributions.keys()),
    "names": distributions.keys(),
    "bounds": [[0, 1]] * len(distributions.keys()),
}

keys_prior = list(distributions.keys())
print("Prior Keys")
print("-----------------------")
print(keys_prior)

# Generate uniform samples (i.e. one unit hypercube)
if method == "sobol":
    unif_sample = sobol.sample(
        problem, n_draw_samples, calc_second_order=calc_second_order, seed=42
    )
else:
    unif_sample = lhs(len(keys_prior), n_draw_samples)


def add_default_values(df, dists, distribution_name):
    """
    Add default values to the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to which default values will be added.
    dists : dict
        Dictionary containing the default values.
    distribution_name : str
        The name of the distribution set.

    Returns
    -------
    pd.DataFrame
        The DataFrame with default values added.
    """
    print("\nAdding default values\n")
    for key, val in dists[distribution_name]["default_values"].items():
        if key not in df.columns:
            df[key] = val
            print(f"{key}: {val}")

    return df


def convert_samples(unif_sample):
    """
    Convert uniform samples to the specified distributions.

    Parameters
    ----------
    unif_sample : np.ndarray
        Array of uniform samples.

    Returns
    -------
    tuple
        A tuple containing the converted samples and the number of samples.
    """
    n_samples = unif_sample.shape[0]
    # To hold the transformed variables
    dist_sample = np.zeros_like(unif_sample, dtype="object")

    # For each variable, transform with the inverse of the CDF (inv(CDF)=ppf)
    for i, key in enumerate(keys_prior):
        if key == "sliding_law":
            dist_sample[:, i] = [
                slidinglaw[id] for id in distributions[key].ppf(unif_sample[:, i])
            ]
        elif key == "climate_file":
            dist_sample[:, i] = [
                rcms[id] for id in distributions[key].ppf(unif_sample[:, i])
            ]
        elif key == "input.regrid.file":
            dist_sample[:, i] = [
                initialstates[id] for id in distributions[key].ppf(unif_sample[:, i])
            ]
        elif key == "prescribed_retreat_file":
            dist_sample[:, i] = [
                retreatfiles[id] for id in distributions[key].ppf(unif_sample[:, i])
            ]
        elif key == "stress_balance":
            dist_sample[:, i] = [
                f"{sb_dict[int(id)]}"
                for id in distributions[key].ppf(unif_sample[:, i])
            ]

        elif key == "ocean_file":
            dist_sample[:, i] = [
                f"MAR3.9_{gcms[int(id)]}_ocean_1960-2100_v4.nc"
                for id in distributions[key].ppf(unif_sample[:, i])
            ]
        else:
            dist_sample[:, i] = distributions[key].ppf(unif_sample[:, i])
    return dist_sample, n_samples


dist_sample, n_samples = convert_samples(unif_sample)
dist_median_sample, _ = convert_samples(np.median(unif_sample, axis=0, keepdims=True))

# dist_median_sample[0, 3] = gcms[0]
# dist_median_sample = np.vstack([dist_median_sample] * 2)
# dist_median_sample[:, -1] = retreatfiles.values()

if posterior_file:
    X_posterior = pd.read_csv(posterior_file).drop(
        columns=["Unnamed: 0", "exp_id"], errors="ignore"
    )
    keys_mc = list(X_posterior.keys())
    keys = list(set(keys_prior + keys_mc))
    print(f"Prior: {keys_prior}")
    print(f"Posterior: {keys_mc}")

    print(keys_prior, keys_mc)
    if len(keys_prior) + len(keys_mc) != len(keys):
        print("Duplicate keys, exciting.")
    keys = keys_prior + keys_mc
    mc_indices = np.random.choice(range(X_posterior.shape[0]), n_samples)
    X_sample = X_posterior.to_numpy()[mc_indices, :]

    dist_sample = np.hstack((dist_sample, X_sample))

else:
    keys = keys_prior


# Convert to Pandas dataframe, append column headers, output as csv
df = pd.DataFrame(dist_sample, columns=keys)
df.to_csv(outfile, index=True, index_label="id")

ensemble_outfile = outfile.parent / Path(f"ensemble_{outfile.name}")
ensemble_df = add_default_values(df, dists, distribution_name)
ensemble_df.to_csv(ensemble_outfile, index=True, index_label="id")


# median_outfile = outfile.parent / Path(f"median_{outfile.name}")
# median_df = pd.DataFrame(
#     dist_median_sample, columns=keys_prior, index=["MEDIAN-FREE", "MEDIAN-PRESCRIBED"]
# )
# median_df.to_csv(median_outfile, index=True, index_label="id")


# ensemble_median_outfile = outfile.parent.parent / Path(
#     f"ensemble_median_{outfile.name}"
# )
# ensemble_median_df = add_default_values(median_df, dists, distribution_name)
# ensemble_median_df.to_csv(
#     ensemble_median_outfile,
#     index=["MEDIAN-FREE", "MEDIAN-PRESCRIBED"],
#     index_label="id",
# )
