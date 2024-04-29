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
Generate scrips to hindcasts of the Greenland Ice Sheet using the Parallel Ice Sheet Model (PISM)
"""

import inspect
import os
import shlex
import subprocess as sub
import sys
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from os.path import abspath, dirname, join, realpath
from typing import Any, Dict, List, Union

import pandas as pd
import xarray as xr


def current_script_directory() -> str:
    """
    Return the current directory
    """

    m_filename = inspect.stack(0)[0][1]
    return realpath(dirname(m_filename))


script_directory = current_script_directory()

sys.path.append(join(script_directory, "../pism_ragis"))
import computing  # pylint: disable=C0413
from systems import Systems  # pylint: disable=C0413

grid_choices = [
    18000,
    9000,
    6000,
    4500,
    3600,
    3000,
    2400,
    1800,
    1500,
    1200,
    900,
    600,
    450,
    300,
    150,
]


def create_offset_file(file_name: str, delta_T: float = 0.0, frac_P: float = 1.0):
    """
    Generate offset file using xarray
    """
    dT = [delta_T]
    fP = [frac_P]
    time = [0]
    time_bounds = [[-1, 1]]

    ds = xr.Dataset(
        data_vars=dict(  # pylint: disable=R1735
            delta_T=(["time"], dT, {"units": "K"}),
            frac_P=(["time"], fP, {"units": ""}),
            time_bounds=(["time", "bnds"], time_bounds, {}),
        ),
        coords=dict(  # pylint: disable=R1735
            time=(
                "time",
                time,
                {
                    "units": "seconds since 01-01-01",
                    "axis": "T",
                    "calendar": "365_day",
                    "bounds": "time_bounds",
                },
            )
        ),
    )
    ds.to_netcdf(file_name)


available_systems = Systems()
available_systems.default_path = "../hpc-systems"

if __name__ == "__main__":
    # set up the option parser
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.description = "Generating scripts for warming experiments."
    parser.add_argument(
        "FILE", nargs=1, help="Input file to restart from", default=None
    )
    parser.add_argument(
        "-n",
        "--n_procs",
        dest="n",
        type=int,
        help="""number of cores/processors. default=140.""",
        default=140,
    )
    parser.add_argument(
        "-w",
        "--wall_time",
        dest="walltime",
        help="""walltime. default: 48:00:00.""",
        default="48:00:00",
    )
    parser.add_argument(
        "-q",
        "--queue",
        dest="queue",
        choices=computing.list_queues(),
        help="""queue. default=long.""",
        default="long",
    )
    parser.add_argument(
        "--options",
        dest="commandline_options",
        help="""Here you can add command-line options""",
    )
    parser.add_argument(
        "-d",
        "--domain",
        dest="domain",
        choices=[
            "gris",
            "gris_ext",
            "qaamerujup",
        ],
        help="sets the modeling domain",
        default="gris",
    )
    parser.add_argument(
        "--exstep",
        dest="exstep",
        help="Writing interval for spatial time series",
        default="monthly",
    )
    parser.add_argument(
        "--tsstep",
        dest="tsstep",
        help="Writing interval for scalar time series",
        default="daily",
    )
    parser.add_argument(
        "-f",
        "--o_format",
        dest="oformat",
        choices=["netcdf3", "netcdf4_parallel", "netcdf4_serial", "pnetcdf"],
        help="output format",
        default="netcdf4_parallel",
    )
    parser.add_argument(
        "-L",
        "--comp_level",
        dest="compression_level",
        help="Compression level for output file.",
        default=2,
    )
    parser.add_argument(
        "-g",
        "--grid",
        dest="grid",
        type=int,
        choices=grid_choices,
        help="horizontal grid resolution",
        default=1800,
    )
    parser.add_argument(
        "--i_dir",
        dest="input_dir",
        help="input directory",
        default=abspath(join(script_directory, "..")),
    )
    parser.add_argument(
        "--data_dir",
        dest="data_dir",
        help="data directory",
        default=abspath(join(script_directory, "../data_sets/")),
    )
    parser.add_argument(
        "--o_dir", dest="output_dir", help="output directory", default="test_dir"
    )
    parser.add_argument(
        "--o_size",
        dest="osize",
        choices=["small", "medium", "big", "big_2d", "custom"],
        help="output size type",
        default="custom",
    )
    parser.add_argument(
        "--test_climate_models",
        dest="test_climate_models",
        action="store_true",
        help="Turn off ice dynamics and mass transport to test climate models",
        default=False,
    )
    parser.add_argument(
        "-s",
        "--system",
        dest="system",
        #        choices=available_systems.list_systems(),
        help="computer system to use.",
        default="debug",
    )
    parser.add_argument(
        "-b",
        "--bed_type",
        dest="bed_type",
        choices=computing.list_bed_types(),
        help="output size type",
        default="wc",
    )
    parser.add_argument(
        "--spatial_ts",
        dest="spatial_ts",
        choices=[
            "basic",
            "standard",
            "none",
            "ismip6",
            "strain",
            "fractures",
            "ragis",
            "qaamerujup",
            "dem",
        ],
        help="output size type",
        default="ragis",
    )
    parser.add_argument(
        "--hydrology",
        dest="hydrology",
        choices=["routing", "routing_steady", "diffuse"],
        help="Basal hydrology model.",
        default="diffuse",
    )
    parser.add_argument(
        "--calving",
        dest="calving",
        choices=["vonmises_calving", "hayhurst_calving"],
        help="Choose calving law",
        default="vonmises_calving",
    )
    parser.add_argument(
        "--stable_gl",
        dest="float_kill_calve_near_grounding_line",
        action="store_false",
        help="Stable grounding line",
        default=True,
    )
    parser.add_argument(
        "--stress_balance",
        dest="stress_balance",
        choices=["sia", "ssa+sia", "ssa", "blatter"],
        help="stress balance solver",
        default="ssa+sia",
    )
    parser.add_argument(
        "--dataset_version",
        dest="version",
        choices=[
            "2023_GIMP",
            "2023_GRIMP",
            "2023_RAGIS",
            "2024_KRIG",
            "2023-12_RAGIS",
            "2024-02_RAGIS",
            "1_GrIMP",
            "2_GrIMP",
            "3_GrIMP",
            "4_GrIMP",
            "5_GrIMP",
        ],
        help="input data set version",
        default="2023_GRIMP",
    )
    parser.add_argument("--start", help="Simulation start year", default="1980-1-1")
    parser.add_argument("--end", help="Simulation end year", default="2020-1-1")
    parser.add_argument(
        "-e",
        "--ensemble_file",
        dest="ensemble_file",
        help="File that has all combinations for ensemble study",
        default=None,
    )

    options = parser.parse_args()
    commandline_options = options.commandline_options

    start_date = options.start
    end_date = options.end

    nn = options.n
    input_dir = abspath(options.input_dir)
    data_dir = abspath(options.data_dir)
    output_dir = abspath(options.output_dir)

    compression_level = options.compression_level
    oformat = options.oformat
    osize = options.osize
    queue = options.queue
    walltime = options.walltime
    # system = available_systems[options.system]

    spatial_ts = options.spatial_ts
    test_climate_models = options.test_climate_models
    bed_type = options.bed_type
    exstep = options.exstep
    tsstep = options.tsstep
    float_kill_calve_near_grounding_line = options.float_kill_calve_near_grounding_line
    grid = options.grid
    hydrology = options.hydrology

    stress_balance = options.stress_balance
    version = options.version
    ensemble_file = options.ensemble_file
    domain = options.domain
    pism_exec = computing.generate_domain(domain)

    if options.FILE is None:
        print("Missing input file")
    else:
        input_file = abspath(options.FILE[0])

    if domain.lower() in ("greenland_ext", "gris_ext"):
        pism_dataname = (
            f"$data_dir/bed_dem/pism_Greenland_ext_{grid}m_v{version}_{bed_type}.nc"
        )
    else:
        pism_dataname = (
            f"$data_dir/bed_dem/pism_Greenland_{grid}m_v{version}_{bed_type}.nc"
        )

    master_config_file = computing.get_path_to_config()

    # Removed "thk" from regrid vars
    # regridvars = "litho_temp,enthalpy,age,tillwat,bmelt,ice_area_specific_volume"
    regridvars = "litho_temp,enthalpy,age,tillwat,bmelt,ice_area_specific_volume,thk"

    dirs = {"output": "$output_dir"}
    for d in ["performance", "state", "scalar", "spatial", "jobs", "basins"]:
        dirs[d] = f"$output_dir/{d}"

    if spatial_ts == "none":
        del dirs["spatial"]

    # use the actual path of the run scripts directory (we need it now and
    # not during the simulation)
    scripts_dir = join(output_dir, "run_scripts")
    if not os.path.isdir(scripts_dir):
        os.makedirs(scripts_dir)

    # use the actual path of the time file directory (we need it now and
    # not during the simulation)
    time_dir = join(output_dir, "time_forcing")
    if not os.path.isdir(time_dir):
        os.makedirs(time_dir)

    # use the actual path of the uq directory
    uq_dir = join(output_dir, "uq")
    if not os.path.isdir(uq_dir):
        os.makedirs(uq_dir)

    # generate the config file *after* creating the output directory
    pism_config = "pism"
    pism_config_nc = join(output_dir, pism_config + ".nc")

    nc_cmd = f"ncgen -o {pism_config_nc} {input_dir}/config/{pism_config}.cdl"
    sub.call(shlex.split(nc_cmd))

    m_dirs = " ".join(list(dirs.values()))
    # these Bash commands are added to the beginning of the run scrips
    run_header = f"""# stop if a variable is not defined
set -u
# stop on errors
set -e

# path to the config file
config="{pism_config_nc}"
# path to the input directory (input data sets are contained in this directory)
input_dir="{input_dir}"
# path to data directory
data_dir="{data_dir}"
# output directory
output_dir="{output_dir}"

# create required output directories
for each in {m_dirs};
    do
      mkdir -p $each
done\n\n
"""
    if options.system != "debug":
        cmd = f"""lfs setstripe -c -1 {dirs["output"]}"""
        sub.call(shlex.split(cmd))

    ensemble_infile = os.path.split(ensemble_file)[-1]
    ensemble_outfile = join(uq_dir, ensemble_infile)

    ens_cmd = f"cp {ensemble_file} {ensemble_outfile}"
    sub.call(shlex.split(ens_cmd))

    ensemble_infile_2 = ensemble_infile.split("ensemble_")[-1]
    ensemble_outfile_2 = join(uq_dir, ensemble_infile_2)

    cmd = f"cp {ensemble_file} {ensemble_outfile}"
    sub.call(shlex.split(cmd))

    pism_timefile = join(time_dir, f"timefile_{start_date}_{end_date}.nc")
    try:
        os.remove(pism_timefile)
    except OSError:
        pass

    periodicity = "daily"
    if os.environ.get("PISM_PREFIX") == "":
        pism_path = "~/pism"
    else:
        pism_path = os.environ.get("PISM_PREFIX")  # type: ignore

    print(os.environ.get("PISM_PREFIX"))
    tm_cmd: List[Any] = [
        join(pism_path, "bin/create_timeline.py"),
        "-a",
        start_date,
        "-e",
        end_date,
        "-p",
        periodicity,
        "-d",
        "2008-01-01",
        pism_timefile,
    ]
    sub.call(tm_cmd)

    # ########################################################
    # set up model initialization
    # ########################################################

    ssa_n = 3.25
    ssa_e = 1.0

    uq_df = pd.read_csv(ensemble_file)
    uq_df.fillna(False, inplace=True)

    scripts = []

    simulation_start_year = options.start
    simulation_end_year = options.end

    batch_header, batch_system = computing.make_batch_header(
        options.system, nn, walltime, queue
    )
    post_header = computing.make_batch_post_header(options.system)

    for n, row in enumerate(uq_df.iterrows()):
        combination = row[1]
        print(combination)

        name_options = {}
        name_options["id"] = combination["id"]

        vversion = "v" + str(version)
        full_exp_name = "_".join(
            [
                vversion,
                "_".join(
                    ["_".join([k, str(v)]) for k, v in list(name_options.items())]
                ),
            ]
        )

        experiment = "_".join(
            [
                vversion,
                "_".join(
                    ["_".join([k, str(v)]) for k, v in list(name_options.items())]
                ),
                f"{start_date}",
                f"{end_date}",
            ]
        )

        script = join(scripts_dir, f"{domain}_g{grid}m_{experiment}.sh")
        scripts.append(script)

        for filename in script:
            try:
                os.remove(filename)
            except OSError:
                pass

        with open(script, "w", encoding="utf-8") as f:
            f.write(batch_header)
            f.write(run_header)

            pism = computing.generate_prefix_str(pism_exec)

            general_params_dict = {
                "profile": join(
                    dirs["performance"], f"""profile_${batch_system["job_id"]}.py"""
                ),
                "time.file": pism_timefile,
                "output.format": oformat,
                "output.compression_level": compression_level,
                "config_override": "$config",
                "stress_balance.ice_free_thickness_standard": 5,
                "input.forcing.time_extrapolation": "true",
                "energy.ch_warming.enabled": "false",
                "energy.bedrock_thermal.file": "$data_dir/bheatflux/Geothermal_heatflux_map_v2.1_g450m.nc",
            }

            outfile = f"{domain}_g{grid}m_{experiment}.nc"

            general_params_dict["output.file"] = join(dirs["state"], outfile)
            general_params_dict["bootstrap"] = ""
            #              general_params_dict["input.file"] = pism_dataname
            general_params_dict["i"] = pism_dataname
            if hasattr(combination, "input.regrid.file"):
                regrid_file = (
                    f"""$data_dir/initial_states/{combination["input.regrid.file"]}"""
                )
                general_params_dict["input.regrid.file"] = regrid_file
            else:
                general_params_dict["input.regrid.file"] = input_file
            general_params_dict["input.regrid.vars"] = regridvars
            if test_climate_models:
                general_params_dict["test_climate_models"] = ""

            if osize != "custom":
                general_params_dict["output.size"] = osize
            else:
                general_params_dict[
                    "output.sizes.medium"
                ] = "sftgif,velsurf_mag,mask,usurf,bmelt"

            grid_params_dict = computing.generate_grid_description(grid, domain)

            sb_params_dict: Dict[str, Union[str, int, float]] = {
                "stress_balance.sia.enhancement_factor": combination["sia_e"],
                "stress_balance.ssa.enhancement_factor": ssa_e,
                "stress_balance.ssa.Glen_exponent": ssa_n,
                "basal_resistance.pseudo_plastic.q": combination["pseudo_plastic_q"],
                "basal_yield_stress.mohr_coulomb.topg_to_phi.enabled": "yes",
                "basal_yield_stress.mohr_coulomb.till_effective_fraction_overburden": combination[
                    "till_effective_fraction_overburden"
                ],
                "stress_balance.blatter.enhancement_factor": combination["sia_e"],
            }
            phi_min = combination["phi_min"]
            phi_max = combination["phi_max"]
            z_min = combination["z_min"]
            z_max = combination["z_max"]

            sb_params_dict[
                "basal_yield_stress.mohr_coulomb.topg_to_phi.phi_max"
            ] = phi_max
            sb_params_dict[
                "basal_yield_stress.mohr_coulomb.topg_to_phi.phi_min"
            ] = phi_min
            sb_params_dict[
                "basal_yield_stress.mohr_coulomb.topg_to_phi.topg_max"
            ] = z_max
            sb_params_dict[
                "basal_yield_stress.mohr_coulomb.topg_to_phi.topg_min"
            ] = z_min

            if (hasattr(combination, "fractures")) and (
                combination["fractures"] is True
            ):
                sb_params_dict["fractures"] = True
                sb_params_dict["fracture_density.include_grounded_ice"] = True
                sb_params_dict["fracture_density.constant_healing"] = True
                sb_params_dict["fracture_weighted_healing"] = True
                sb_params_dict["fracture_density.borstad_limit"] = True
                sb_params_dict["write_fd_fields"] = True
                sb_params_dict["scheme_fd2d"] = True
                sb_params_dict["fracture_gamma"] = combination["fracture_gamma"]
                sb_params_dict["fracture_gamma_h"] = combination["fracture_gamma_h"]
                sb_params_dict["fracture_softening"] = combination["fracture_softening"]
                sb_params_dict["fracture_initiation_threshold"] = combination[
                    "fracture_initiation_threshold"
                ]
                sb_params_dict["healing_threshold"] = combination["healing_threshold"]

            sliding_law = "pseudo_plastic"
            if hasattr(combination, "sliding_law"):
                sliding_law = combination["sliding_law"]
            sb_params_dict[f"basal_resistance.{sliding_law}.enabled"] = "yes"

            stress_balance_params_dict = computing.generate_stress_balance(
                stress_balance, sb_params_dict
            )

            climate_file_p = f"""$data_dir/climate/{combination["climate_file"]}"""
            climate_offset_file_p = join(
                uq_dir, f"""ragis_offset_file_id_{combination["id"]}.nc"""
            )

            climate_parameters: Dict[str, Union[str, int, float]] = {
                "atmosphere.given.file": climate_file_p,
                "surface.given.file": climate_file_p,
            }

            if combination["climate"] in ("given_pdd", "given_pdd_delta"):
                climate_parameters["surface.pdd.factor_ice"] = (
                    combination["surface.pdd.factor_ice"] / 910.0
                )
                climate_parameters["surface.pdd.factor_snow"] = (
                    combination["surface.pdd.factor_snow"] / 910.0
                )
                climate_parameters["surface.pdd.refreeze"] = combination[
                    "surface.pdd.refreeze"
                ]
                climate_parameters["surface.pdd.std_dev.value"] = combination[
                    "surface.pdd.std_dev.value"
                ]
                create_offset_file(
                    realpath(climate_offset_file_p),
                    combination["delta_T"],
                    combination["frac_P"],
                )
                climate_parameters["atmosphere.delta_T.file"] = climate_offset_file_p
                climate_parameters["atmosphere.frac_P.file"] = climate_offset_file_p
            climate_params_dict = computing.generate_climate(
                combination["climate"], **climate_parameters
            )

            runoff_file_p = f"""$data_dir/climate/{combination["runoff_file"]}"""
            hydrology_parameters: Dict[str, Union[str, int, float]] = {
                "hydrology.routing.include_floating_ice": True,
                "hydrology.surface_input.file": runoff_file_p,
                "hydrology.add_water_input_to_till_storage": False,
            }

            hydro_params_dict = computing.generate_hydrology(
                combination["hydrology"], **hydrology_parameters
            )

            ocean_file_p = f"""$data_dir/ocean/{combination["ocean_file"]}"""
            frontal_melt = combination["frontal_melt"]
            if frontal_melt == "discharge_routing":
                hydrology_parameters["hydrology.surface_input.file"] = ocean_file_p

                frontalmelt_parameters = {
                    "frontal_melt.models": "routing",
                    "frontal_melt.routing.file": ocean_file_p,
                }
            elif frontal_melt == "off":
                frontalmelt_parameters = {}

            else:
                frontalmelt_parameters = {
                    "frontal_melt.models": "discharge_given",
                    "frontal_melt.discharge_given.file": ocean_file_p,
                }

            frontalmelt_params_dict = frontalmelt_parameters

            ocean_parameters = {
                "ocean.th.file": ocean_file_p,
                "ocean.th.clip_salinity": False,
                "ocean.th.gamma_T": combination["gamma_T"],
            }
            if hasattr(combination, "salinity"):
                if combination["salinity"] is not False:
                    ocean_parameters["constants.sea_water.salinity"] = combination[
                        "salinity"
                    ]

            ocean_params_dict = computing.generate_ocean(
                combination["ocean"], **ocean_parameters
            )

            calving_parameters: Dict[str, Union[str, int, float]] = {
                "calving.float_kill.calve_near_grounding_line": float_kill_calve_near_grounding_line,
                "calving.vonmises_calving.use_custom_flow_law": True,
                "calving.vonmises_calving.Glen_exponent": 3.0,
                "geometry.front_retreat.use_cfl": True,
            }

            if hasattr(combination, "prescribed_retreat_file") & (
                combination["prescribed_retreat_file"] is not False
            ):
                calving_parameters[
                    "geometry.front_retreat.prescribed.file"
                ] = f"""$data_dir/front_retreat/{combination["prescribed_retreat_file"]}"""

            vcm = combination["vcm"]
            try:
                vcm = float(vcm)
                calving_parameters["calving.vonmises_calving.sigma_max"] = vcm * 1e6
            except:  # pylint: disable=W0702
                vonmises_calving_threshold_file_p = "$data_dir/calving/{vcm}"
                calving_parameters[
                    "calving.vonmises_calving.threshold_file"
                ] = vonmises_calving_threshold_file_p
            if "calving.thickness_calving.threshold" in combination:
                calving_parameters["calving.thickness_calving.threshold"] = combination[
                    "calving.thickness_calving.threshold"
                ]
            if "calving.thickness_calving.file" in combination:
                calving_parameters[
                    "calving.thickness_calving.file"
                ] = f"""$data_dir/calving/{combination[
                    "calving.thickness_calving.file"]}"""
                if "calving.thickness_calving.threshold" in calving_parameters:
                    del calving_parameters["calving.thickness_calving.threshold"]

            if "calving.rate_scaling.file" in combination:
                if combination["calving.rate_scaling.file"]:
                    calving_parameters[
                        "calving.rate_scaling.file"
                    ] = f"""$data_dir/calving/{combination[
                    "calving.rate_scaling.file"]}"""
                    calving_parameters["calving.rate_scaling.period"] = 0

            calving = options.calving
            calving_params_dict = computing.generate_calving(
                calving, **calving_parameters
            )

            scalar_ts_dict = computing.generate_scalar_ts(
                outfile, tsstep, odir=dirs["scalar"]
            )
            solver_dict: Dict[str, Union[str, int, float]] = {}

            all_params_dict = computing.merge_dicts(
                general_params_dict,
                grid_params_dict,
                stress_balance_params_dict,
                climate_params_dict,
                ocean_params_dict,
                hydro_params_dict,
                frontalmelt_params_dict,
                calving_params_dict,
                scalar_ts_dict,
                solver_dict,
            )
            if spatial_ts != "none":
                exvars = computing.spatial_ts_vars[spatial_ts]
                spatial_ts_dict = computing.generate_spatial_ts(
                    outfile, exvars, exstep, odir=dirs["spatial"]
                )
                all_params_dict = computing.merge_dicts(
                    all_params_dict, spatial_ts_dict
                )

            print("\nChecking parameters")
            print("------------------------------------------------------------")
            with xr.open_dataset(master_config_file) as m_ds:
                for key in all_params_dict:
                    if hasattr(m_ds["pism_config"], key) is False:
                        print(f"  - {key} not found in pism_config")
            print("------------------------------------------------------------\n")

            all_params = " \\\n  ".join(
                [f"-{k} {v}" for k, v in list(all_params_dict.items())]
            )

            if commandline_options is not None:
                all_params = f"{all_params} \\\n  {commandline_options[1:-1]}"

            print("\nChecking input files")
            print("------------------------------------------------------------")
            for key, m_f in all_params_dict.items():
                if key.split(".")[-1] == "file":
                    m_f_abs = m_f.replace("$data_dir", options.data_dir)
                    print(f"  - {m_f_abs}: {os.path.isfile(m_f_abs)}")
            print("------------------------------------------------------------\n")

            if options.system == "debug":
                redirect = " 2>&1 | tee {jobs}/job.${job_id}"
            else:
                redirect = " > {jobs}/job.${job_id} 2>&1"

            template = "{mpido} {pism} {params}" + redirect

            context = computing.merge_dicts(
                batch_system, dirs, {"pism": pism, "params": all_params}
            )
            cmd = template.format(**context)
            f.write(cmd)
            f.write("\n")

            f.write("\n")
            run_id = combination["id"]
            id_cmd = f"ncatted -a id,global,a,c,{run_id}"
            for m_file in [
                scalar_ts_dict["output.timeseries.filename"],
                join(dirs["state"], outfile),
            ]:
                cmd = f"{id_cmd} {m_file}\n"
                f.write(cmd)
            f.write("\n")
            f.write("\n")
            f.write(batch_system.get("footer", ""))

        scripts.append(script)

    scripts = computing.uniquify_list(scripts)
    print("\n".join([script for script in scripts]))  # pylint: disable=R1721
    print("\nwritten\n")
