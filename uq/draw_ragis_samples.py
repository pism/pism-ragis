#!/usr/bin/env python

"""
Uncertainty quantification using Latin Hypercube Sampling or Sobol Sequences
"""

from argparse import ArgumentParser
from typing import Any, Dict

import numpy as np
import pandas as pd
from pyDOE import lhs
from SALib.sample import saltelli
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


dists: Dict[str, Any] = {
    "init": {
        "uq": {},
        "default_values": {
            "climate": "given_smb",
            "hydrology": "routing",
            "frontal_melt": "discharge_routing",
            "ocean": "th",
            "ocean_file": "MAR3.9_CNRM-ESM2_ssp585_ocean_1960-2100_v4.nc",
            "climate_file": "DMI-HIRHAM5_ERA_1980_2020_EPSG3413_4500M_MM.nc",
            "runoff_file": "DMI-HIRHAM5_ERA_1980_2020_EPSG3413_4500M_MM.nc",
            "gamma_T": 1.00e-4,
            "salinity": 34,
            "pseudo_plastic_q": 0.6,
            "sia_e": 1.25,
            "ssa_n": 3.25,
            "calving.thickness_calving.threshold": 100,
            "vcm": 0.75,
            "fractures": "false",
            "z_min": -700,
            "z_max": 700,
            "phi_min": 5,
            "phi_max": 40,
            "till_effective_fraction_overburden": 0.02,
            "sliding_law": "pseudo_plastic",
        },
    },
    "stress_balance": {
        "uq": {"stress_balance": randint(0, 2)},
        "default_values": {
            "climate": "given_smb",
            "hydrology": "diffuse",
            "frontal_melt": "off",
            "ocean": "th",
            "ocean_file": "MAR3.9_CNRM-ESM2_ssp585_ocean_1960-2100_v4.nc",
            "climate_file": "DMI-HIRHAM5_ERA_1980_2020_EPSG3413_4500M_MM.nc",
            "runoff_file": "DMI-HIRHAM5_ERA_1980_2020_EPSG3413_4500M_MM.nc",
            "gamma_T": 1.00e-4,
            "salinity": 34,
            "pseudo_plastic_q": 0.6,
            "sia_e": 1.25,
            "ssa_n": 3.25,
            "calving.thickness_calving.threshold": 100,
            "vcm": 0.75,
            "fractures": "false",
            "z_min": -700,
            "z_max": 700,
            "phi_min": 5,
            "phi_max": 40,
            "till_effective_fraction_overburden": 0.02,
            "sliding_law": "pseudo_plastic",
        },
    },
    "calving": {
        "uq": {
            "vcm": uniform(loc=0.25, scale=0.5),
            "calving.thickness_calving.threshold": uniform(loc=100, scale=300),
        },
        "default_values": {
            "climate": "given_smb",
            "hydrology": "diffuse",
            "frontal_melt": "off",
            "ocean": "th",
            "ocean_file": "MAR3.9_CNRM-ESM2_ssp585_ocean_1960-2100_v4.nc",
            "climate_file": "DMI-HIRHAM5_ERA_1980_2020_EPSG3413_4500M_MM.nc",
            "runoff_file": "DMI-HIRHAM5_ERA_1980_2020_EPSG3413_4500M_MM.nc",
            "gamma_T": 1.00e-4,
            "salinity": 34,
            "pseudo_plastic_q": 0.6,
            "sia_e": 1.25,
            "ssa_n": 3.25,
            "calving.thickness_calving.threshold": 100,
            "fractures": "false",
            "sliding_law": "pseudo_plastic",
        },
    },
    "ocean-calving": {
        "uq": {
            "vcm": uniform(loc=0.40, scale=0.20),
            "gamma_T": uniform(loc=0.75e-4, scale=0.75e-4),
            "ocean_file": randint(0, len(gcms)),
        },
        "default_values": {
            "climate": "given_smb",
            "hydrology": "routing",
            "frontal_melt": "routing",
            "ocean_file": "MAR3.9_CNRM-ESM2_ssp585_ocean_1960-2100_v4.nc",
            "climate_file": "DMI-HIRHAM5_ERA_1980_2020_EPSG3413_4500M_MM.nc",
            "runoff_file": "DMI-HIRHAM5_ERA_1980_2020_EPSG3413_4500M_MM.nc",
            "pseudo_plastic_q": 0.6,
            "sia_e": 1.25,
            "ssa_n": 3.25,
            "fractures": "false",
            "sliding_law": "pseudo_plastic",
            "z_min": -700,
            "z_max": 700,
            "phi_min": 5,
            "phi_max": 40,
            "till_effective_fraction_overburden": 0.02,
            "calving.thickness_calving.threshold": 100,
        },
    },
    "ocean-calving-init": {
        "uq": {
            "vcm": uniform(loc=0.25, scale=0.75),
            "gamma_T": uniform(loc=1e-4, scale=0.5e-4),
            "ocean_file": randint(0, len(gcms)),
            "calving.thickness_calving.file": randint(0, len(tcts)),
            "input.regrid.file": randint(0, len(initialstates)),
        },
        "default_values": {
            "climate": "given_smb",
            "hydrology": "routing",
            "frontal_melt": "routing",
            "ocean_file": "MAR3.9_CNRM-ESM2_ssp585_ocean_1960-2100_v4.nc",
            "climate_file": "DMI-HIRHAM5_ERA_1980_2020_EPSG3413_4500M_MM.nc",
            "runoff_file": "DMI-HIRHAM5_ERA_1980_2020_EPSG3413_4500M_MM.nc",
            "pseudo_plastic_q": 0.6,
            "sia_e": 1.25,
            "ssa_n": 3.25,
            "fractures": "false",
            "sliding_law": "pseudo_plastic",
            "z_min": -700,
            "z_max": 700,
            "phi_min": 5,
            "phi_max": 40,
            "till_effective_fraction_overburden": 0.02,
        },
    },
    "thickness-calving": {
        "uq": {
            "vcm": uniform(loc=0.2, scale=0.8),
            "gamma_T": uniform(loc=1e-4, scale=0.5e-4),
            "ocean_file": randint(0, len(gcms)),
            "calving.thickness_calving.threshold": uniform(loc=50, scale=450),
        },
        "default_values": {
            "climate": "given_smb",
            "hydrology": "routing",
            "frontal_melt": "routing",
            "ocean_file": "MAR3.9_CNRM-ESM2_ssp585_ocean_1960-2100_v4.nc",
            "climate_file": "DMI-HIRHAM5_ERA_1980_2020_EPSG3413_4500M_MM.nc",
            "runoff_file": "DMI-HIRHAM5_ERA_1980_2020_EPSG3413_4500M_MM.nc",
            "pseudo_plastic_q": 0.6,
            "sia_e": 1.25,
            "ssa_n": 3.25,
            "fractures": "false",
            "sliding_law": "pseudo_plastic",
            "z_min": -700,
            "z_max": 700,
            "phi_min": 5,
            "phi_max": 40,
            "till_effective_fraction_overburden": 0.02,
        },
    },
    "ocean": {
        "uq": {
            "vcm": uniform(loc=0.2, scale=0.8),
            "gamma_T": uniform(loc=1e-4, scale=0.5e-4),
            "ocean_file": randint(0, len(gcms)),
        },
        "default_values": {
            "climate": "given_smb",
            "hydrology": "routing",
            "frontal_melt": "routing",
            "ocean_file": "MAR3.9_CNRM-ESM2_ssp585_ocean_1960-2100_v4.nc",
            "climate_file": "DMI-HIRHAM5_ERA_1980_2020_EPSG3413_4500M_MM.nc",
            "runoff_file": "DMI-HIRHAM5_ERA_1980_2020_EPSG3413_4500M_MM.nc",
            "salinity": 34,
            "pseudo_plastic_q": 0.6,
            "sia_e": 1.25,
            "ssa_n": 3.25,
            "fractures": "false",
            "sliding_law": "pseudo_plastic",
            "z_min": -700,
            "z_max": 700,
            "phi_min": 5,
            "phi_max": 40,
            "till_effective_fraction_overburden": 0.02,
            "calving.thickness_calving.threshold": 50,
        },
    },
    "ocean-simple": {
        "uq": {
            "vcm": uniform(loc=0.25, scale=0.75),
            "gamma_T": uniform(loc=1e-4, scale=0.5e-4),
            "ocean_file": randint(0, len(gcms)),
        },
        "default_values": {
            "climate": "given_smb",
            "hydrology": "diffuse",
            "frontal_melt": "off",
            "ocean": "th",
            "ocean_file": "MAR3.9_CNRM-ESM2_ssp585_ocean_1960-2100_v4.nc",
            "climate_file": "DMI-HIRHAM5_ERA_1980_2020_EPSG3413_4500M_MM.nc",
            "runoff_file": "DMI-HIRHAM5_ERA_1980_2020_EPSG3413_4500M_MM.nc",
            "salinity": 34,
            "pseudo_plastic_q": 0.6,
            "sia_e": 1.25,
            "ssa_n": 3.25,
            "fractures": "false",
            "sliding_law": "pseudo_plastic",
            "z_min": -700,
            "z_max": 700,
            "phi_min": 5,
            "phi_max": 40,
            "till_effective_fraction_overburden": 0.02,
            "calving.thickness_calving.threshold": 50,
        },
    },
    "calving-simple": {
        "uq": {
            "vcm": uniform(loc=0.25, scale=0.75),
            "gamma_T": uniform(loc=1e-4, scale=0.5e-4),
            "ocean_file": randint(0, len(gcms)),
            "calving.thickness_calving.file": randint(0, len(tcts)),
        },
        "default_values": {
            "climate": "given_smb",
            "hydrology": "diffuse",
            "frontal_melt": "off",
            "ocean": "th",
            "ocean_file": "MAR3.9_CNRM-ESM2_ssp585_ocean_1960-2100_v4.nc",
            "climate_file": "DMI-HIRHAM5_ERA_1980_2020_EPSG3413_4500M_MM.nc",
            "runoff_file": "DMI-HIRHAM5_ERA_1980_2020_EPSG3413_4500M_MM.nc",
            "salinity": 34,
            "pseudo_plastic_q": 0.6,
            "sia_e": 1.25,
            "ssa_n": 3.25,
            "fractures": "false",
            "sliding_law": "pseudo_plastic",
            "z_min": -700,
            "z_max": 700,
            "phi_min": 5,
            "phi_max": 40,
            "till_effective_fraction_overburden": 0.02,
        },
    },
    "dem": {
        "uq": {
            "calving.thickness_calving.threshold": uniform(loc=50, scale=250),
            "surface.pdd.factor_ice": uniform(loc=0.5, scale=12.5),
            "surface.pdd.factor_snow": uniform(loc=0.5, scale=5.5),
            "surface.pdd.std_dev.value": uniform(loc=2, scale=4),
        },
        "default_values": {
            "climate": "given_pdd_delta",
            "hydrology": "diffuse",
            "frontal_melt": "off",
            "ocean": "const",
            "ocean_file": "MAR3.9_CNRM-ESM2_ssp585_ocean_1960-2100_v4.nc",
            "climate_file": "DMI-HIRHAM5_ERA_1980_2020_EPSG3413_4500M_MM.nc",
            "runoff_file": "DMI-HIRHAM5_ERA_1980_2020_EPSG3413_4500M_MM.nc",
            "salinity": 34,
            "pseudo_plastic_q": 0.7508221,
            "sia_e": 2.608046,
            "ssa_n": 3.309718,
            "fractures": "false",
            "vcm": 1,
            "gamma_T": 1e-4,
            "surface.pdd.refreeze": 0.6,
            "till_effective_fraction_overburden": 0.01845403,
            "sliding_law": "pseudo_plastic",
            "phi_min": 7.193718,
            "phi_max": 42.79528,
            "z_min": -369.6359,
            "z_max": 243.8239,
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
    help="""number of samples to draw. default=10.""",
    default=10,
)
parser.add_argument(
    "-d",
    "--distribution",
    dest="distribution",
    choices=dists.keys(),
    help="""Choose set.""",
    default="calving",
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
    choices=["lhs", "saltelli"],
    help="""number of samples to draw. default=lhs.""",
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

# Generate uniform samples (i.e. one unit hypercube)
if method == "saltelli":
    unif_sample = saltelli.sample(
        problem, n_draw_samples, calc_second_order=calc_second_order
    )
elif method == "lhs":
    unif_sample = lhs(len(keys_prior), n_draw_samples)
else:
    print(f"Method {method} not available")

n_samples = unif_sample.shape[0]
# To hold the transformed variables
dist_sample = np.zeros_like(unif_sample, dtype="object")

sb_dict = {0: "ssa+sia", 1: "blatter"}
# For each variable, transform with the inverse of the CDF (inv(CDF)=ppf)
for i, key in enumerate(keys_prior):
    if key == "calving_rate_scaling_file":
        dist_sample[:, i] = [
            f"seasonal_calving_{int(id)}_1980_2020.nc"
            for id in distributions[key].ppf(unif_sample[:, i])
        ]
    elif key == "calving.thickness_calving.file":
        dist_sample[:, i] = [
            tcts[id] for id in distributions[key].ppf(unif_sample[:, i])
        ]
    elif key == "input.regrid.file":
        dist_sample[:, i] = [
            initialstates[id] for id in distributions[key].ppf(unif_sample[:, i])
        ]
    elif key == "stress_balance":
        dist_sample[:, i] = [
            f"{sb_dict[int(id)]}" for id in distributions[key].ppf(unif_sample[:, i])
        ]

    elif key == "frontal_melt_file":
        dist_sample[:, i] = [
            f"jib_ocean_forcing_{int(id)}_1980_2020.nc"
            for id in distributions[key].ppf(unif_sample[:, i])
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
        columns=["Unnamed: 0", "Model"], errors="ignore"
    )
    keys_mc = list(X_posterior.keys())
    keys = list(set(keys_prior + keys_mc))
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
