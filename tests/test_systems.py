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

"""

from pathlib import Path

import pytest

from pism_ragis.systems import System, Systems


@pytest.fixture(name="machine_file")
def fixture_machine_file() -> Path:
    """
    Return Path to toml file
    """
    return Path("tests/data/chinook.toml")


@pytest.fixture(name="machine_dict")
def fixture_machine_dict() -> dict:
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


@pytest.fixture(name="system")
def fixture_system(machine_file):
    """
    Return a basic systems
    """

    return System(machine_file)


@pytest.fixture(name="systems")
def fixture_systems():
    """
    Return a basic systems
    """

    return Systems()


def test_system_from_dict(machine_dict):
    """
    Test creating a System from a dictionary
    """
    s = System(machine_dict)
    assert s["machine"] == "chinook"


def test_system_from_file(machine_file):
    """
    Test creating a System from a toml file
    """
    s = System(machine_file)
    assert s["machine"] == "chinook"


def test_system_list_queues(system):
    """
    Test listing queues
    """
    assert system.list_queues() == ["t1standard", "t1small", "t2standard", "t2small"]
    assert system.list_queues("old") == [
        "t1standard",
        "t1small",
        "t2standard",
        "t2small",
    ]


def test_systems_to_dict(system):
    """
    Test class returning a dictionary

    """
    assert system.to_dict()["machine"] == "chinook"


def test_system_list_partitions(system):
    """
    Test listing partitions
    """
    assert system.list_partitions() == ["old-chinook", "new-chinook"]


def test_systems_len(systems):
    """
    Test len of Systems
    """
    assert len(systems) == 3


def test_systems_default_path(systems):
    """
    Test default path
    """
    assert systems.default_path == Path("hpc-systems")


def test_systems_from_pathlib_path():
    """
    Test adding systems from a pathlib path
    """
    systems = Systems()
    systems.add_from_path(Path("tests/data"))
    assert len(systems) == 2


def test_systems_from_str_path():
    """
    Test adding systems from a str path
    """
    systems = Systems()
    systems.add_from_path("tests/data")
    assert len(systems) == 2


def test_systems_add_system(systems):
    """
    Test adding a system
    Test checking if system exists
    """
    system = System("tests/data/debug.txt")
    systems.add_system(system)
    assert len(systems) == 4
    assert systems.add_system(system) == "debugger already exists"
