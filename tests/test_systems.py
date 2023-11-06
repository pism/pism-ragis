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
Tests for Systems class

We need a way to register machine. Maybe the machine list should be its own repo?
"""

from pathlib import Path

import pytest

from pism_ragis.systems import System


@pytest.fixture(name="machine_file")
def fixture_machine_file():
    """
    Return Path to toml file
    """
    return Path("tests/data/chinook.toml")


@pytest.fixture(name="machine_dict")
def fixture_machine_dict():
    """
    Return system dict
    """
    return {
        "machine": "chinook",
        "MPI": {"mpido": "mpirun -np {cores} -machinefile ./nodes_$SLURM_JOBID"},
        "scheduler": {"name": "SLRUM", "submit": "sbatch", "job_id": "SLURM_JOBID"},
        "filesystem": {"work_dir": "SLURM_SUBMIT_DIR"},
        "queues": {
            "t1standard": 24,
            "t1small": 24,
            "t2standard": 24,
            "t2small": 24,
            "debug": 24,
            "analysis": 24,
        },
    }


def test_system_from_dict(machine_dict):
    """
    Test creating a System from a dictionary
    """
    s = System(machine_dict)
    assert s.machine == "chinook"  # type: ignore[attr-defined] # pylint: disable=E1101


def test_system_from_file(machine_file):
    """
    Test creating a System from a toml file
    """
    s = System(machine_file)
    assert s.machine == "chinook"  # type: ignore[attr-defined] # pylint: disable=E1101


# @pytest.fixture
# def systems():
#     """
#     Return a basic systems
#     """

#     p = "tests/data"
#     return Systems(p)


# def test_systems_len(systems):
#     assert len(systems) == 2
