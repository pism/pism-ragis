#!/usr/bin/env python

"""
Uncertainty quantification using Latin Hypercube Sampling or Sobol Sequences.
"""

from argparse import ArgumentParser
from typing import Any, Dict

import numpy as np
import pandas as pd
from pyDOE2 import lhs
from SALib.sample import sobol
from scipy.stats.distributions import randint, uniform

short2long: Dict[str, str] = {
    "SIAE": "sia_e",
    "SSAN": "ssa_n",
    "PPQ": "pseudo_plastic_q",
    "TEFO": "till_effective_fraction_overburden",
    "PHIMIN": "phi_min",
    "PHIMAX": "phi_max",
    "ZMIN": "z_min",
    "ZMAX": "z_max",
}

climate: Dict[int, str] = {
    0: "HIRHAM5-monthly-ERA5_1975_2021.nc",
    1: "MARv3.14-monthly-ERA5_1940_2023.nc",
    2: "RACMO2.3p2_ERA5_FGRN055_1940_2023.nc",
}
gcms: Dict[int, str] = {
    0: "ACCESS1-3_rcp85",
    1: "CNRM-CM6_ssp126",
    2: "CNRM-CM6_ssp585",
    3: "CNRM-ESM2_ssp585",
    4: "CSIRO-Mk3.6_rcp85",
    5: "HadGEM2-ES_rcp85",
    6: "IPSL-CM5-MR_rcp85",
    7: "MIROC-ESM-CHEM_rcp26",
    8: "MIROC-ESM-CHEM_rcp85",
    9: "NorESM1-M_rcp85",
    10: "UKESM1-CM6_ssp585",
}

tcts: Dict[int, str] = {
    0: "tct_forcing_200myr_74n_50myr_76n.nc",
    1: "tct_forcing_300myr_74n_50myr_76n.nc",
    2: "tct_forcing_400myr_74n_50myr_76n.nc",
}

initialstates: Dict[int, str] = {
    0: "gris_g900m_v2023_GIMP_id_CTRL_0_25.nc",
    1: "gris_g900m_v2023_RAGIS_id_CTRL_0_25.nc",
}

retreatfiles: Dict[int, str|bool] = {
    0: "pism_g450m_frontretreat_calfin_1972_2019.nc",
    1: ""}

dists: Dict[str, Any] = {
    "ragis": {
        "uq": {
            "calving.vonmises_calving.sigma_max": uniform(loc=350_000, scale=300_000),
            "calving.rate_scaling.file": randint(0, 7),
            "ocean.th.gamma_T": uniform(loc=0.75e-4, scale=0.75e-4),
            "ocean_file": randint(0, len(gcms)),
            "climate_file": randint(0, len(climate)),
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
    "dem": {
        "uq": {
            "calving.vonmises_calving.sigma_max": uniform(loc=350_000, scale=300_000),
            "calving.rate_scaling.file": randint(0, 7),
            "ocean.th.gamma_T": uniform(loc=0.5e-4, scale=1.00e-4),
            "ocean_file": randint(0, len(gcms)),
            "frontal_melt.routing.parameter_a": uniform(loc=2.4e-4, scale=1.2e-4),
            "frontal_melt.routing.parameter_b": uniform(loc=1.0, scale=0.70),
            "frontal_melt.routing.power_alpha": uniform(loc=0.3, scale=0.55),
            "frontal_melt.routing.power_beta": uniform(loc=1.1, scale=0.7),
            "delta_T": uniform(-2, 0.5),
            "frac_P": uniform(0.0, 0.5),
            "surface.pdd.factor_ice": uniform(loc=4, scale=8),
            "surface.pdd.factor_snow": uniform(loc=0.5, scale=5.5),
            "surface.pdd.std_dev.value": uniform(loc=1, scale=5),
        },
        "default_values": {
            "climate": "given_pdd_delta",
            "hydrology": "diffuse",
            "frontal_melt": "off",
            "ocean": "constant",
            "ocean_file": "MAR3.9_CNRM-ESM2_ssp585_ocean_1960-2100_v4.nc",
            "climate_file": "RACMO2.3p2_ERA5_FGRN055_1940_2023.nc",
            "salinity": 34,
            "fractures": "false",
            "frontal_melt": "routing",
            "hydrology": "routing",
            "ocean.models": "th",
            "ocean.th.gamma_T": 0.0001,
            "ocean_file": "MAR3.9_CNRM-ESM2_ssp585_ocean_1960-2100_v4.nc",
            "surface.pdd.refreeze": 0.6,
            "till_effective_fraction_overburden": 0.01845403,
            "sliding_law": "pseudo_plastic",
            "pseudo_plastic_q": 0.7508221,
            "sia_e": 2.608046,
            "ssa_n": 3.309718,
            "phi_min": 7.193718,
            "phi_max": 42.79528,
            "z_min": -369.6359,
            "z_max": 243.8239,
            "prescribed_retreat_file": None,
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
    "--calc_second_order",
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
calc_second_order = options.calc_second_order
method = options.method
outfile = options.OUTFILE[-1]
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
print(keys_prior)

# Generate uniform samples (i.e. one unit hypercube)
if method == "sobol":
    unif_sample = sobol.sample(
        problem, n_draw_samples, calc_second_order=calc_second_order
    )
else:
    unif_sample = lhs(len(keys_prior), n_draw_samples)

n_samples = unif_sample.shape[0]
# To hold the transformed variables
dist_sample = np.zeros_like(unif_sample, dtype="object")

sb_dict = {0: "ssa+sia", 1: "blatter"}
# For each variable, transform with the inverse of the CDF (inv(CDF)=ppf)
for i, key in enumerate(keys_prior):
    if key == "calving.rate_scaling.file":
        dist_sample[:, i] = [
            f"seasonal_calving_id_{int(id)}_1975_2025.nc"
            for id in distributions[key].ppf(unif_sample[:, i])
        ]
    elif key == "climate_file":
        dist_sample[:, i] = [
            climate[id] for id in distributions[key].ppf(unif_sample[:, i])
        ]
    elif key == "calving.thickness_calving.file":
        dist_sample[:, i] = [
            tcts[id] for id in distributions[key].ppf(unif_sample[:, i])
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
            f"{sb_dict[int(id)]}" for id in distributions[key].ppf(unif_sample[:, i])
        ]

    elif key == "ocean_file":
        dist_sample[:, i] = [
            f"MAR3.9_{gcms[int(id)]}_ocean_1960-2100_v4.nc"
            for id in distributions[key].ppf(unif_sample[:, i])
        ]
    else:
        dist_sample[:, i] = distributions[key].ppf(unif_sample[:, i])

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

print("\nAdding default values\n")
for key, val in dists[distribution_name]["default_values"].items():
    if key not in df.columns:
        df[key] = val
        print(f"{key}: {val}")

df.to_csv(f"ensemble_{outfile}", index=True, index_label="id")
