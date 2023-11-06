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
Module provides System class
"""

from pathlib import Path
from typing import Union

import toml


class System:
    """
    Class fo a system
    """

    def __init__(self, d: Union[dict, Path, str]):
        if isinstance(d, dict):
            for key, value in d.items():
                setattr(self, key, value)
        elif isinstance(d, Path):
            for key, value in toml.load(d).items():
                setattr(self, key, value)
        elif isinstance(d, str):
            self._values = toml.loads(d)
        else:
            print(f"{d} not recognized")

    def to_dict(self):
        """
        Returns self as dictionary
        """
        return self.__dict__

    def __repr__(self):
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
        self._default_path: Path = Path("data/")
        self.add_systems_from_path(self.default_path)

    @property
    def default_path(self):
        """
        Return default path to glob for toml files
        """
        return self._default_path

    @default_path.setter
    def default_path(self, value):
        self._default_path = value

    def __getitem__(self, name):
        return self._values[name]

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __iter__(self):
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

    def add_system(self, system):
        """
        Add a system from a System class
        """
        system.to_dict()

    def add_systems_from_path(self, path):
        """

        Add systems from a pathlib.Path.

        Use glob to add all files with suffix `toml`.
        """
        p = Path(path).glob("*.toml")
        sys = {}
        for p_ in p:
            s = toml.load(p_)
            machine = s["machine"]
            sys[machine] = s
        self._values = sys

    # def __len__(self):
    #     return len(self.systems)

    def dump(self):
        """
        Dump class to string
        """
        repr_str = ""
        for s in self.values():
            repr_str += s["machine"]
            repr_str += "\n------------\n\n"
            repr_str += toml.dumps(s)
            repr_str += "\n"
        return repr_str

    def __repr__(self):
        return self.dump()
