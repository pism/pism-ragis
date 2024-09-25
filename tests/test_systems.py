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
Tests for Systems class.
"""

from pathlib import Path

import pytest

from pism_ragis.systems import System, Systems


@pytest.fixture(name="machine_file")
def fixture_machine_file() -> Path:
    """
    Fixture providing the path to a TOML file.

    This fixture returns the path to a TOML file located at
    "tests/data/chinook.toml".

    Returns
    -------
    Path
        The path to the TOML file.
    """
    return Path("tests/data/chinook.toml")


@pytest.fixture(name="system")
def fixture_system(machine_file):
    """
    Fixture providing a basic system instance.

    This fixture returns an instance of the `System` class, initialized
    with the provided `machine_file`.

    Parameters
    ----------
    machine_file : Path
        The path to the machine configuration file, provided by the
        `machine_file` fixture.

    Returns
    -------
    System
        An instance of the `System` class initialized with the `machine_file`.
    """
    return System(machine_file)


@pytest.fixture(name="systems")
def fixture_systems():
    """
    Fixture providing a basic `Systems` instance.

    This fixture returns an instance of the `Systems` class.

    Returns
    -------
    Systems
        An instance of the `Systems` class.
    """
    return Systems()


def test_system_from_file(machine_file):
    """
    Test creating a `System` from a TOML file.

    This function asserts that a `System` instance created from the provided
    TOML file has the key "machine" with the value "chinook".

    Parameters
    ----------
    machine_file : Path
        The path to the machine configuration file, provided by the
        `machine_file` fixture.

    Raises
    ------
    AssertionError
        If the `System` instance does not have the key "machine" with the
        value "chinook".
    """
    s = System(machine_file)
    assert s["machine"] == "chinook"


def test_system_list_queues(system):
    """
    Test the `list_queues` method of the `System` class.

    This function asserts that the `list_queues` method of the `System` class
    returns the expected list of queue names.

    Parameters
    ----------
    system : System
        An instance of the `System` class, provided by the `system` fixture.

    Raises
    ------
    AssertionError
        If the `list_queues` method does not return the expected list of queue names.
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
    Test the `to_dict` method of the `system` class.

    This function asserts that the `to_dict` method of the `system` class
    returns a dictionary with the key "machine" having the value "chinook".

    Parameters
    ----------
    system : object
        An instance of the class that has a `to_dict` method.

    Raises
    ------
    AssertionError
        If the `to_dict` method does not return a dictionary with the key
        "machine" having the value "chinook".
    """
    assert system.to_dict()["machine"] == "chinook"


def test_system_list_partitions(system):
    """
    Test the `list_partitions` method of the `System` class.

    This function asserts that the `list_partitions` method of the `System` class
    returns the expected list of partition names.

    Parameters
    ----------
    system : System
        An instance of the `System` class, provided by the `system` fixture.

    Raises
    ------
    AssertionError
        If the `list_partitions` method does not return the expected list of partition names.
    """
    assert system.list_partitions() == ["old-chinook", "new-chinook"]


def test_systems_len(systems):
    """
    Test the length of the `Systems` instance.

    This function asserts that the length of the `Systems` instance is 3.

    Parameters
    ----------
    systems : Systems
        An instance of the `Systems` class, provided by the `systems` fixture.

    Raises
    ------
    AssertionError
        If the length of the `Systems` instance is not 3.
    """
    assert len(systems) == 3


def test_systems_default_path(systems):
    """
    Test the default path of the `Systems` instance.

    This function asserts that the `default_path` attribute of the `Systems` instance
    is equal to "hpc-systems".

    Parameters
    ----------
    systems : Systems
        An instance of the `Systems` class, provided by the `systems` fixture.

    Raises
    ------
    AssertionError
        If the `default_path` attribute of the `Systems` instance is not equal to "hpc-systems".
    """
    assert systems.default_path == Path("hpc-systems")


def test_systems_from_pathlib_path():
    """
    Test adding systems from a `pathlib.Path`.

    This function creates an instance of the `Systems` class and adds systems
    from the specified `pathlib.Path`. It asserts that the number of systems
    added is 2.

    Raises
    ------
    AssertionError
        If the number of systems added is not 2.
    """
    systems = Systems()
    systems.add_from_path(Path("tests/data"))
    assert len(systems) == 2


def test_systems_from_str_path():
    """
    Test adding systems from a string path.

    This function creates an instance of the `Systems` class and adds systems
    from the specified string path. It asserts that the number of systems
    added is 2.

    Raises
    ------
    AssertionError
        If the number of systems added is not 2.
    """
    systems = Systems()
    systems.add_from_path("tests/data")
    assert len(systems) == 2


def test_systems_add_system(systems):
    """
    Test adding a system to the `Systems` instance.

    This function adds a system to the `Systems` instance and asserts that
    the number of systems increases to 4. It also checks if adding the same
    system again returns the expected message.

    Parameters
    ----------
    systems : Systems
        An instance of the `Systems` class, provided by the `systems` fixture.

    Raises
    ------
    AssertionError
        If the number of systems is not 4 after adding a new system.
        If adding the same system again does not return the expected message.
    """
    system = System("tests/data/debug.txt")
    systems.add_system(system)
    assert len(systems) == 4
    assert systems.add_system(system) == "debugger already exists"
