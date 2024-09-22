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

# pylint: disable=unused-import,too-many-positional-arguments
# mpypy --untyped-calls-exclude toml

"""
Module provides System class
"""

import math
from pathlib import Path
from typing import Any, Iterator, Union

import toml


class System:
    """
    Class for a system
    """

    def __init__(self, d: Union[dict, Path, str]):
        self._values = {}
        if isinstance(d, dict):
            for key, value in d.items():
                self._values[key] = value
        elif isinstance(d, (Path, str)):
            for key, value in toml.load(d).items():
                self._values[key] = value
        else:
            print(f"{d} not recognized")

    def __getitem__(self, name) -> Any:
        return self._values[name]

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __iter__(self) -> Iterator:
        return iter(self._values)

    def keys(self):
        """
        Return keys
        """
        return self._values.keys()

    def items(self):
        """
        Return items
        """
        return self._values.items()

    def values(self):
        """
        Return values
        """
        return self._values.values()

    def make_batch_header(
        self,
        partition: str = "chinook_new",
        queue: str = "t2standard",
        walltime: str = "8:00:00",
        n_cores: int = 40,
        gid: Union[None, str] = None,
    ) -> str:
        """
        Create a batch header from system and kwargs
        """
        assert n_cores > 0

        assert partition in self.list_partitions()
        assert queue in self.list_queues(partition)

        partition = partition.split("_")[-1]
        ppn = self.partitions[partition]["cores_per_node"]  # type: ignore[attr-defined] # pylint: disable=E1101
        nodes = int(math.ceil(float(n_cores) / ppn))

        if nodes * ppn != n_cores:
            print(
                f"Warning! Running {n_cores} tasks on {nodes} {ppn}-processor nodes, wasting {ppn * nodes - n_cores} processors!"
            )

        lines = (
            self.job["header"]  # type: ignore[attr-defined] # pylint: disable=E1101
            .format(
                queue=queue,
                walltime=walltime,
                cores=n_cores,
                ppn=ppn,
                partition=partition,
                gid=gid,
            )
            .split("\n")
        )
        m_str = "\n".join(list(lines))
        return m_str

    def list_partitions(self) -> list:
        """
        List all partitions
        """
        return [
            values["name"]
            for key, values in self["partitions"].items()
            if key != "default"
        ]

    def list_queues(self, partition: Union[None, str] = None) -> list:
        """
        List available queues for partition. If no partition
        is given return default partition
        """

        if not partition:
            p = self["partitions"]["default"]
        else:
            p = partition
        partition = p.split("_")[-1]
        return self["partitions"][partition]["queues"]

    def to_dict(self) -> dict:
        """
        Returns self as dictionary
        """
        return self._values

    def __repr__(self) -> str:
        repr_str = ""

        repr_str += toml.dumps(self.to_dict())

        return f"""

System
------------

{repr_str}
        """


class Systems:
    """
    Class to hold Systems of base class System
    """

    def __init__(self):
        self._default_path: Path = Path("hpc-systems")
        self.add_from_path(self._default_path)

    @property
    def default_path(self):
        """
        Return default path to glob for toml files
        """
        return self._default_path

    @default_path.setter
    def default_path(self, value):
        self._default_path = value
        self.add_from_path(self._default_path)

    def __getitem__(self, name) -> Any:
        return self._values[name]

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __iter__(self) -> Iterator:
        return iter(self._values)

    def keys(self):
        """
        Return keys
        """
        return self._values.keys()

    def items(self):
        """
        Return items
        """
        return self._values.items()

    def values(self):
        """
        Return values
        """
        return self._values.values()

    def __repr__(self) -> str:
        return self.dump()

    def __len__(self) -> int:
        return len(self.values())

    def list_systems(self) -> list:
        """
        Return name of machines as list
        """
        return list(self.keys())

    def add_system(self, system: System):
        """
        Add a system
        """
        machine = system["machine"]
        if machine not in self.keys():
            self._values[machine] = system
            return None
        else:
            msg = f"{machine} already exists"
            print(msg)
            return msg

    def add_from_path(self, path: Union[Path, str]):
        """
        Add systems from a pathlib.Path or str.

        Use glob to add all files with suffix `toml`.
        """
        p = Path(path).glob("*.toml")
        sys = {}
        for p_ in p:
            s = toml.load(p_)
            machine = s["machine"]
            sys[machine] = System(s)
        self._values = sys

    def add_system_from_file(self, path: Union[Path, str]):
        """
        Add a system from a pathlib.Path or str.

        """
        s = toml.load(path)
        machine = s["machine"]
        self._values[machine] = System(s)

    def dump(self) -> str:
        """
        Dump class to string
        """
        repr_str = ""
        for s in self.values():
            repr_str += "\n------------\n\n"
            repr_str += toml.dumps(s.to_dict())
            repr_str += "\n"
        return repr_str
