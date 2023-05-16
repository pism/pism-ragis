"""
resources
=========

Provides:
  - general resources such as grid constructors, calving, hydrology, etc.
    for the Greenland Ice Sheet and sub-regions thereof

"""

import inspect
import math
import os
import os.path
import shlex
import subprocess
import sys
from collections import OrderedDict
from typing import Dict, Union


def generate_prefix_str(pism_exec):
    """
    Generate prefix string.

    Returns: string
    """

    return os.path.join(os.environ.get("PISM_PREFIX", ""), pism_exec)


def generate_domain(domain):
    """
    Generate domain specific options

    Returns: string
    """

    if domain.lower() in ("greenland", "gris", "gris_ext", "ismip6"):
        pism_exec = "pismr"
    elif domain.lower() in ("synth_jib", "synth_ellps"):
        pism_exec = "pismr -regional -calving_wrap_around -ssa_dirichelt_bc"
    elif domain.lower() in ("hia"):
        x_min = -652200.0
        x_max = -232600.0
        y_min = -1263900.0
        y_max = -943500.0
        pism_exec = """pismr -x_range {x_min},{x_max} -y_range {y_min},{y_max} -bootstrap""".format(
            x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max
        )

    elif domain.lower() in ("jakobshavn", "jib"):
        x_min = -282650.0
        x_max = 293350.0
        y_min = -2417600.0
        y_max = -2021600.0
        pism_exec = """pismr -regional -x_range {x_min},{x_max} -y_range {y_min},{y_max}  -bootstrap -regional.zero_gradient true -regional.no_model_strip 4.5""".format(
            x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max
        )
    elif domain.lower() in ("qaanaaq"):
        x_min = -507650.0
        x_max = -363650.0
        y_min = -1310600.0
        y_max = -1157600.0
        pism_exec = """pismr -regional -x_range {x_min},{x_max} -y_range {y_min},{y_max}  -bootstrap -regional.zero_gradient true -regional.no_model_strip 4.5""".format(
            x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max
        )
    elif domain.lower() in ("qaamerujup"):
        x_min = -250000.0
        x_max = -153000.0
        y_min = -2075000.0
        y_max = -2021000.0
        pism_exec = """pismr -regional -x_range {x_min},{x_max} -y_range {y_min},{y_max}  -bootstrap -regional.zero_gradient true -regional.no_model_strip 4.5""".format(
            x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max
        )
    elif domain.lower() in ("nw"):
        x_min = -400000.0
        x_max = 320000.0
        y_min = -2022000.0
        y_max = -1500000.0
        pism_exec = """pismr -regional -x_range {x_min},{x_max} -y_range {y_min},{y_max}  -bootstrap -regional.zero_gradient true -regional.no_model_strip 4.5""".format(
            x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max
        )
    else:
        print(("Domain {} not recognized, exiting".format(domain)))

        sys.exit(0)

    return pism_exec


spatial_ts_vars = {}


spatial_ts_vars["ismip6"] = ["ismip6"]


spatial_ts_vars["basic"] = [
    "basal_melt_rate_grounded",
    "beta",
    "dHdt",
    "height_above_flotation",
    "grounding_line_flux",
    "frontal_melt_rate",
    "frontal_melt_retreat_rate",
    "ice_mass",
    "mask",
    "mass_fluxes",
    "sftgif",
    "thk",
    "topg",
    "usurf",
    "velbase_mag",
    "velsurf_mag",
    "vonmises_calving_rate",
    "vonmises_stress",
]

spatial_ts_vars["calib"] = [
    "beta",
    "velsurf_mag",
    "tillwat",
]


spatial_ts_vars["paleo"] = [
    "bmelt",
    "climatic_mass_balance",
    "effective_air_temp",
    "effective_precipitation",
    "dHdt",
    "ice_mass",
    "mask",
    "thk",
    "usurf",
    "velbase_mag",
    "velsurf_mag",
]

spatial_ts_vars["paleo_tracer"] = [
    "climatic_mass_balance",
    "effective_air_temp",
    "effective_precipitation",
    "dHdt",
    "grounding_line_flux",
    "ice_mass",
    "mask",
    "mass_fluxes",
    "sftgif",
    "thk",
    "topg",
    "usurf",
    "velbase_mag",
    "velsurf_mag",
    "vonmises_calving_rate",
    "lat",
    "lon",
    "uvel",
    "vvel",
    "wvel",
    "wvel_rel",
]


spatial_ts_vars["ragis"] = [
    "dHdt",
    "grounding_line_flux",
    "ice_mass",
    "mask",
    "mass_fluxes",
    "thk",
    "usurf",
    "velsurf_mag",
    "flux_divergence",
    "frontal_melt_rate",
    "frontal_melt_retreat_rate",
    "vonmises_calving_rate",
]

spatial_ts_vars["standard"] = [
    "bmelt",
    "basal_mass_flux_floating",
    "beta",
    "bwat",
    "dHdt",
    "diffusivity",
    "fracture_density",
    "fracture_growth_rate",
    "fracture_healing_rate",
    "fracture_flow_enhancement",
    "fracture_age",
    "fracture_toughness",
    "height_above_flotation",
    "grounding_line_flux",
    "frontal_melt_rate",
    "frontal_melt_retreat_rate",
    "ice_mass",
    "mask",
    "mass_fluxes",
    "sftgif",
    "strain_rates",
    "subglacial_discharge",
    "tauc",
    "taud_mag",
    "tendency_of_subglacial_water_mass",
    "thk",
    "tillwat",
    "topg",
    "usurf",
    "velbase_mag",
    "velsurf_mag",
    "vonmises_calving_rate",
    "vonmises_stress",
]

spatial_ts_vars["fractures"] = [
    "bmelt",
    "dHdt",
    "fracture_density",
    "fracture_growth_rate",
    "fracture_healing_rate",
    "fracture_flow_enhancement",
    "fracture_toughness",
    "height_above_flotation",
    "grounding_line_flux",
    "frontal_melt_rate",
    "frontal_melt_retreat_rate",
    "ice_mass",
    "mask",
    "mass_fluxes",
    "sftgif",
    "strain_rates",
    "thk",
    "topg",
    "usurf",
    "velbase_mag",
    "velsurf_mag",
    "vonmises_calving_rate",
    "vonmises_stress",
]

spatial_ts_vars["hydro"] = [
    "basal_melt_rate_grounded",
    "bwat",
    "bwp",
    "bwatvel",
    "beta",
    "dHdt",
    "frontal_melt_rate",
    "frontal_melt_retreat_rate",
    "grounding_line_flux",
    "hydraulic_potential",
    "ice_mass",
    "mask",
    "mass_fluxes",
    "sftgif",
    "temppabase",
    "tillwat",
    "thk",
    "topg",
    "usurf",
    "velbase_mag",
    "velsurf_mag",
    "vonmises_calving_rate",
]

spatial_ts_vars["outlet"] = [
    "beta",
    "bmelt",
    "bwatvel",
    "dHdt",
    "climatic_mass_balance",
    "diffusivity",
    "diffusivity_staggered",
    "frontal_melt_rate",
    "frontal_melt_retreat_rate",
    "grounding_line_flux",
    "height_above_flotation",
    "hydraulic_potential",
    "hydraulic_potential_adjustment",
    "ice_mass",
    "mask",
    "mass_fluxes",
    "nuH",
    "subglacial_water_flux",
    "sftgif",
    "tauc",
    "tillphi",
    "taud",
    "tendency_of_subglacial_water_mass",
    "thk",
    "topg",
    "usurf",
    "velbase",
    "velbase_mag",
    "velsurf_mag",
    "vonmises_calving_rate",
]

spatial_ts_vars["strain"] = [
    "beta",
    "dHdt",
    "deviatoric_stresses",
    "diffusivity",
    "effective_viscosity",
    "frontal_melt_rate",
    "frontal_melt_retreat_rate",
    "ice_mass",
    "mask",
    "mass_fluxes",
    "sftgif",
    "strain_rates",
    "thk",
    "tillwat",
    "topg",
    "usurf",
    "velbase_mag",
    "velsurf_mag",
]


def generate_spatial_ts(
    outfile, exvars, step, start=None, end=None, split=None, odir=None
):
    """
    Return dict to generate spatial time series

    Returns: OrderedDict
    """

    # check if list or comma-separated string is given.
    try:
        exvars = ",".join(exvars)
    except:
        pass

    params_dict = OrderedDict()
    if split is True:
        outfile, ext = os.path.splitext(outfile)
        params_dict["extra_split"] = ""
    if odir is None:
        params_dict["extra_file"] = "ex_" + outfile
    else:
        params_dict["extra_file"] = os.path.join(odir, "ex_" + outfile)
    params_dict["extra_vars"] = exvars

    if step is None:
        step = "yearly"

    if start is not None and end is not None:
        times = "{start}:{step}:{end}".format(start=start, step=step, end=end)
    else:
        times = step

    params_dict["extra_times"] = times

    return params_dict


def generate_scalar_ts(outfile, step, odir=None, **kwargs):
    """
    Return dict to create scalar time series

    Returns: OrderedDict
    """

    params_dict = OrderedDict()
    if odir is None:
        params_dict["ts_file"] = "ts_" + outfile
    else:
        params_dict["ts_file"] = os.path.join(odir, "ts_" + outfile)

    if step is None:
        step = "yearly"
    else:
        times = step
    params_dict["ts_times"] = times

    return params_dict


def generate_snap_shots(outfile, times, odir=None):
    """
    Return dict to generate snap shots

    Returns: OrderedDict
    """

    params_dict = OrderedDict()
    if odir is None:
        params_dict["save_file"] = "save_" + outfile.split(".nc")[0]
    else:
        params_dict["save_file"] = os.path.join(odir, "save_" + outfile.split(".nc")[0])

    params_dict["save_times"] = ",".join(str(e) for e in times)
    params_dict["save_split"] = ""
    params_dict["save_force_output_times"] = ""

    return params_dict


def generate_grid_description(grid_resolution, domain, restart=False):
    """
    Generate grid description dict

    Returns: OrderedDict
    """

    Lz = 4000
    Lbz = 2000

    if domain.lower() in ("greenland_ext", "gris_ext"):
        mx_max = 14400
        my_max = 24080
    else:
        mx_max = 10560
        my_max = 18240

    resolution_max = 150

    accepted_resolutions = [
        150,
        300,
        450,
        600,
        900,
        1200,
        1500,
        1800,
        2400,
        3000,
        3600,
        4500,
        6000,
        9000,
        18000,
        36000,
    ]

    assert grid_resolution in accepted_resolutions

    if grid_resolution < 1200:
        skip_max = 200
        mz = 201
        mzb = 21
    elif 1200 <= grid_resolution < 4500:
        skip_max = 100
        mz = 201
        mzb = 21
    else:
        skip_max = 50
        mz = 201
        mzb = 21

    grid_div = grid_resolution / resolution_max

    mx = int(mx_max / grid_div)
    my = int(my_max / grid_div)

    horizontal_grid: Dict[str, Union[str, int, float]] = OrderedDict()
    horizontal_grid["Mx"] = mx
    horizontal_grid["My"] = my

    vertical_grid: Dict[str, Union[str, int, float]] = OrderedDict()
    vertical_grid["Lz"] = Lz
    vertical_grid["Lbz"] = Lbz
    vertical_grid["z_spacing"] = "equal"
    vertical_grid["Mz"] = mz
    vertical_grid["Mbz"] = mzb

    grid_options: Dict[str, Union[str, int, float]] = {}
    grid_options["skip"] = ""
    grid_options["skip_max"] = skip_max

    grid_dict = merge_dicts(horizontal_grid, vertical_grid, grid_options)

    if restart:
        g_dict = grid_options
    else:
        g_dict = grid_dict

    return g_dict


def merge_dicts(*dict_args):
    """
    Given any number of dicts, shallow copy and merge into a new dict,
    precedence goes to key value pairs in latter dicts.

    Returns: OrderedDict
    """
    result = OrderedDict()
    for dictionary in dict_args:
        result.update(dictionary)
    return result


def uniquify_list(seq, idfun=None):
    """
    Remove duplicates from a list, order preserving.
    From http://www.peterbe.com/plog/uniqifiers-benchmark
    """

    if idfun is None:

        def idfun(x):
            return x

    seen = {}
    result = []
    for item in seq:
        marker = idfun(item)
        if marker in seen:
            continue
        seen[marker] = 1
        result.append(item)
    return result


def generate_stress_balance(stress_balance, additional_params_dict):
    """
    Generate stress balance params

    Returns: OrderedDict
    """

    accepted_stress_balances = ("sia", "ssa+sia", "blatter")

    if stress_balance not in accepted_stress_balances:
        print(f"{stress_balance} not in {accepted_stress_balances}")
        print(f"available stress balance solvers are {accepted_stress_balances}")

    params_dict = OrderedDict()
    params_dict["stress_balance"] = stress_balance
    if stress_balance in ("ssa+sia", "blatter"):
        params_dict["options_left"] = ""
        params_dict["cfbc"] = ""
        params_dict["kill_icebergs"] = ""
        params_dict["part_grid"] = ""
        params_dict["part_redist"] = ""
        params_dict["sia_flow_law"] = "gpbld"
        params_dict["tauc_slippery_grounding_lines"] = ""

    if stress_balance == "blatter":
        params_dict["stress_balance.blatter.coarsening_factor"] = 4
        params_dict["blatter_Mz"] = 17
        params_dict["bp_ksp_type"] = "gmres"
        params_dict["bp_pc_type"] = "mg"
        params_dict["bp_pc_mg_levels"] = 3
        params_dict["bp_mg_levels_ksp_type"] = "richardson"
        params_dict["bp_mg_levels_pc_type"] = "sor"
        params_dict["bp_mg_coarse_ksp_type"] = "gmres"
        params_dict["bp_mg_coarse_pc_type"] = "bjacobi"
        params_dict["bp_snes_monitor_ratio"] = ""
        params_dict["bp_ksp_monitor"] = ""
        params_dict["bp_ksp_view_singularvalues"] = ""
        params_dict["bp_snes_ksp_ew"] = 1
        params_dict["bp_snes_ksp_ew_version"] = 3

    return merge_dicts(additional_params_dict, params_dict)


def generate_hydrology(hydro, **kwargs):
    """
    Generate hydrology params

    Returns: OrderedDict
    """

    params_dict = OrderedDict()
    if hydro in ("null"):
        params_dict["hydrology"] = "null"
    elif hydro in ("diffuse"):
        params_dict["hydrology"] = "null"
        params_dict["hydrology_null_diffuse_till_water"] = ""
    elif hydro in ("routing"):
        params_dict["hydrology"] = "routing"
    elif hydro in ("steady"):
        params_dict["hydrology"] = "steady"
    elif hydro in ("routing_coupled"):
        params_dict["hydrology"] = "routing"
    elif hydro in ("distributed"):
        params_dict["hydrology"] = "distributed"
        params_dict["basal_yield_stress.add_transportable_water"] = "true"
    elif hydro in ("distributed_coupled"):
        params_dict["hydrology"] = "distributed"
        params_dict["basal_yield_stress.add_transportable_water"] = "true"
    else:
        print((f"hydrology {hydro} not recognized, exiting"))

        sys.exit(0)

    return merge_dicts(params_dict, kwargs)


def generate_calving(calving, **kwargs):
    """
    Generate calving params

    Returns: OrderedDict
    """

    params_dict = OrderedDict()
    if calving in ("thickness_calving"):
        params_dict["calving"] = calving
    elif calving in (
        "eigen_calving",
        "vonmises_calving",
        "hayhurst_calving",
    ):
        params_dict["calving"] = f"{calving},thickness_calving"
    elif calving in ("hybrid_calving"):
        params_dict["calving"] = "eigen_calving,vonmises_calving,thickness_calving"
    elif calving in ("float_kill",):
        params_dict["calving"] = calving
    else:
        print((f"calving {calving} not recognized, exiting"))

        sys.exit(0)
    if "frontal_melt" in kwargs and kwargs["frontal_melt"] is True:
        params_dict["calving"] += ",frontal_melt"
        # need to delete the entry
        del kwargs["frontal_melt"]
    return merge_dicts(params_dict, kwargs)


def generate_climate(climate, **kwargs):
    """
    Generate climate params

    Returns: OrderedDict
    """

    climate_dict = {
        "given_pdd": {"atmosphere": "given", "surface": "pdd"},
        "given_smb": {"atmosphere": "given", "surface": "given"},
    }

    if climate in climate_dict:
        params_dict = climate_dict.get(climate)

    return merge_dicts(params_dict, kwargs)


def generate_ocean(ocean, **kwargs):
    """
    Generate ocean params

    Returns: OrderedDict
    """

    ocean_dict = {"th": {"ocean": "th"}}
    if ocean in ocean_dict:
        params_dict = ocean_dict.get(ocean)

    return merge_dicts(params_dict, kwargs)


def list_systems():
    """
    Return a list of supported systems.
    """
    return sorted(systems.keys())


def list_queues():
    """
    Return a list of supported queues.
    """
    result = set()
    for s in list(systems.values()):
        for q in list(s["queue"].keys()):
            result.add(q)

    return result


def list_bed_types():
    """
    Return a list of supported bed types.
    """

    bed_types = [
        "ctrl",
        "cresis",
        "cresisp",
        "minus",
        "plus",
        "ba01_bed",
        "970mW_hs",
        "jak_1985",
        "no_bath",
        "wc",
        "rm",
    ]

    return bed_types


# information about systems
systems: dict = {}

systems["debug"] = {
    "mpido": "mpiexec -n {cores}",
    "submit": "echo",
    "job_id": "PBS_JOBID",
    "queue": {},
}

systems["chinook"] = {
    "mpido": "mpirun -np {cores} -machinefile ./nodes_$SLURM_JOBID",
    "submit": "sbatch",
    "work_dir": "SLURM_SUBMIT_DIR",
    "job_id": "SLURM_JOBID",
    "queue": {
        "t1standard": 24,
        "t1small": 24,
        "t2standard": 24,
        "t2small": 24,
        "debug": 24,
        "analysis": 24,
    },
}

systems["pleiades"] = {
    "mpido": "mpiexec -n {cores}",
    "submit": "qsub",
    "work_dir": "PBS_O_WORKDIR",
    "job_id": "PBS_JOBID",
    "queue": {"long": 20, "normal": 20, "debug": 20},
}

systems["stampede2"] = {
    "mpido": "ibrun",
    "submit": "sbatch",
    "work_dir": "SLURM_SUBMIT_DIR",
    "job_id": "SLURM_JOBID",
    "queue": {
        "normal": 68,
        "development": 68,
    },
}

systems["frontera"] = {
    "mpido": "ibrun",
    "submit": "sbatch",
    "work_dir": "SLURM_SUBMIT_DIR",
    "job_id": "SLURM_JOBID",
    "queue": {
        "small": 56,
        "normal": 56,
        "development": 56,
    },
}


systems["pleiades_haswell"] = systems["pleiades"].copy()
systems["pleiades_haswell"]["queue"] = {"long": 24, "normal": 24, "debug": 24}

systems["pleiades_ivy"] = systems["pleiades"].copy()
systems["pleiades_ivy"]["queue"] = {"long": 20, "normal": 20, "debug": 20}

systems["pleiades_sandy"] = systems["pleiades"].copy()
systems["pleiades_sandy"]["queue"] = {"long": 16, "normal": 16, "debug": 16}

systems["pleiades_broadwell"] = systems["pleiades"].copy()
systems["pleiades_broadwell"]["queue"] = {"long": 28, "normal": 28, "debug": 28}

systems["electra_broadwell"] = systems["pleiades_broadwell"].copy()

systems["electra_skylake"] = systems["pleiades"].copy()
systems["electra_skylake"]["queue"] = {"long": 40, "normal": 40, "debug": 40}


# headers for batch jobs
#
# Available keywords:
#
# cores    - number of cores (MPI tasks)
# queue    - queue (partition) name
# nodes    - number of nodes
# ppn      - number of tasks per node
# walltime - wall time limit

systems["debug"]["header"] = ""

systems["chinook"][
    "header"
] = """#!/bin/sh
#SBATCH --partition={queue}
#SBATCH --ntasks={cores}
#SBATCH --tasks-per-node={ppn}
#SBATCH --time={walltime}
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --output=pism.%j

module list

umask 007

cd $SLURM_SUBMIT_DIR

# Generate a list of compute node hostnames reserved for this job,
# this ./nodes file is necessary for slurm to spawn mpi processes
# across multiple compute nodes
srun -l /bin/hostname | sort -n | awk '{{print $2}}' > ./nodes_$SLURM_JOBID

ulimit -l unlimited
ulimit -s unlimited
ulimit

"""

systems["stampede2"][
    "header"
] = """#!/bin/sh
#SBATCH -n {cores}
#SBATCH -N {nodes}
#SBATCH --time={walltime}
#SBATCH -p {queue}
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --output=pism.%j

module list

umask 007

cd $SLURM_SUBMIT_DIR

# Generate a list of compute node hostnames reserved for this job,
# this ./nodes file is necessary for slurm to spawn mpi processes
# across multiple compute nodes
srun -l /bin/hostname | sort -n | awk '{{print $2}}' > ./nodes_$SLURM_JOBID

ulimit -l unlimited
ulimit -s unlimited
ulimit

"""

systems["frontera"][
    "header"
] = """#!/bin/sh
#SBATCH -n {cores}
#SBATCH -N {nodes}
#SBATCH --time={walltime}
#SBATCH -p {queue}
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --output=pism.%j

module list

umask 007

cd $SLURM_SUBMIT_DIR

# Generate a list of compute node hostnames reserved for this job,
# this ./nodes file is necessary for slurm to spawn mpi processes
# across multiple compute nodes
srun -l /bin/hostname | sort -n | awk '{{print $2}}' > ./nodes_$SLURM_JOBID

ulimit -l unlimited
ulimit -s unlimited
ulimit

"""


systems["chinook"][
    "footer"
] = """
# clean up the list of hostnames
rm -rf ./nodes_$SLURM_JOBID
"""

systems["electra_broadwell"][
    "header"
] = """#PBS -S /bin/bash
#PBS -N cfd
#PBS -l walltime={walltime}
#PBS -m e
#PBS -q {queue}
#PBS -lselect={nodes}:ncpus={ppn}:mpiprocs={ppn}:model=bro_ele
#PBS -j oe

module list

cd $PBS_O_WORKDIR

"""

systems["pleiades"][
    "header"
] = """#PBS -S /bin/bash
#PBS -N cfd
#PBS -l walltime={walltime}
#PBS -m e
#PBS -W group_list=s2457
#PBS -q {queue}
#PBS -lselect={nodes}:ncpus={ppn}:mpiprocs={ppn}:model=ivy
#PBS -j oe

module list

cd $PBS_O_WORKDIR

"""

systems["pleiades_broadwell"][
    "header"
] = """#PBS -S /bin/bash
#PBS -N cfd
#PBS -l walltime={walltime}
#PBS -m e
#PBS -W group_list={gid}
#PBS -q {queue}
#PBS -lselect={nodes}:ncpus={ppn}:mpiprocs={ppn}:model=bro
#PBS -j oe

module list

cd $PBS_O_WORKDIR

"""

systems["pleiades_sandy"][
    "header"
] = """#PBS -S /bin/bash
#PBS -N cfd
#PBS -l walltime={walltime}
#PBS -m e
#PBS -W group_list={gid}
#PBS -q {queue}
#PBS -lselect={nodes}:ncpus={ppn}:mpiprocs={ppn}:model=san
#PBS -j oe

module list

cd $PBS_O_WORKDIR

"""

systems["pleiades_haswell"][
    "header"
] = """#PBS -S /bin/bash
#PBS -N cfd
#PBS -l walltime={walltime}
#PBS -m e
#PBS -W group_list={gid}
#PBS -q {queue}
#PBS -lselect={nodes}:ncpus={ppn}:mpiprocs={ppn}:model=has
#PBS -j oe

module list

cd $PBS_O_WORKDIR

"""

systems["pleiades_ivy"][
    "header"
] = """#PBS -S /bin/bash
#PBS -N cfd
#PBS -l walltime={walltime}
#PBS -m e
#PBS -W group_list={gid}
#PBS -q {queue}
#PBS -lselect={nodes}:ncpus={ppn}:mpiprocs={ppn}:model=ivy
#PBS -j oe

module list

cd $PBS_O_WORKDIR

"""

systems["electra_skylake"][
    "header"
] = """#PBS -S /bin/bash
#PBS -N cfd
#PBS -l walltime={walltime}
#PBS -m e
#PBS -W group_list={gid}
#PBS -q {queue}
#PBS -lselect={nodes}:ncpus={ppn}:mpiprocs={ppn}:model=sky_ele
#PBS -j oe

module list

cd $PBS_O_WORKDIR

"""

systems["debug"][
    "header"
] = """

"""

# headers for post-processing jobs

post_headers = {}
post_headers[
    "default"
] = """#!/bin/bash

"""

post_headers[
    "pbs"
] = """#PBS -S /bin/bash
#PBS -l select=1:mem=94GB
#PBS -l walltime=8:00:00
#PBS -q ldan

cd $PBS_O_WORKDIR

"""

post_headers[
    "slurm"
] = """#!/bin/bash
#SBATCH --partition=analysis
#SBATCH --ntasks=1
#SBATCH --tasks-per-node=1
#SBATCH --time=48:00:00
#SBATCH --output=pism.%j
#SBATCH --mem=214G

cd $SLURM_SUBMIT_DIR

ulimit -l unlimited
ulimit -s unlimited
ulimit

"""


def make_batch_header(system_name, n_cores, walltime, queue, gid="s2457"):
    """
    Generate header file for different HPC system.

    Returns: String
    """

    # get system info; use "debug" if the requested name was not found
    system = systems.get(system_name, systems["debug"]).copy()

    assert n_cores > 0

    if system_name == "debug":
        # when debugging, assume that all we need is one node
        ppn = n_cores
        nodes = 1
    else:
        try:
            ppn = system["queue"][queue]
        except:
            queues = list(system["queue"].keys())
            raise ValueError(
                f"There is no queue {queue} on {system_name}. Pick one of {queues}."
            )
        # round up when computing the number of nodes needed to run on 'n_cores' cores
        nodes = int(math.ceil(float(n_cores) / ppn))

        if nodes * ppn != n_cores:
            print(
                f"Warning! Running {n_cores} tasks on {nodes} {ppn}-processor nodes, wasting {ppn * nodes - n_cores} processors!"
            )

    system["mpido"] = system["mpido"].format(cores=n_cores)
    system["header"] = system["header"].format(
        queue=queue,
        walltime=walltime,
        nodes=nodes,
        ppn=ppn,
        cores=n_cores,
        gid=gid,
    )
    system["header"] += git_version_header()

    return system["header"], system


def make_batch_post_header(system):
    """
    Make a post header
    """
    v = git_version_header()

    if system in (
        "electra_broadwell",
        "pleiades",
        "pleiades_ivy",
        "pleiades_broadwell",
        "pleiades_haswell",
    ):
        post_header = post_headers["pbs"] + v

    elif system in ("chinook", "stampede2"):
        post_header = post_headers["slurm"] + v
    else:
        post_header = post_headers["default"] + v
    return post_header


def make_batch_header_test():
    "print headers of all supported systems and queues (for testing)"
    for s in list(systems.keys()):
        for q in list(systems[s]["queue"].keys()):
            print(f"# system: {s}, queue: {q}")
            print(make_batch_header(s, 100, "1:00:00", q)[0])


def git_version():
    """Return the path to the top directory of the Git repository
    containing this script, the URL of the "origin" remote and the version."""

    def output(command):
        path = os.path.realpath(os.path.dirname(inspect.stack(0)[0][1]))
        return subprocess.check_output(shlex.split(command), cwd=path).strip()

    return (
        output("git rev-parse --show-toplevel"),
        output("git remote get-url origin"),
        output("git describe --always"),
    )


def git_version_header():
    "Return shell comments containing version info."
    version_info = git_version()
    script = (os.path.realpath(sys.argv[0]),)
    command = (" ".join(sys.argv),)
    path = (version_info[0],)
    url = (version_info[1],)
    version = (version_info[2],)

    return f"""
# Generated by {script}
# Command: {command}
# Git top level: {path}
# URL: {url}
# Version: {version}

"""
