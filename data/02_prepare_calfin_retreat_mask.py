# Copyright (C) 2024 Andy Aschwanden
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
Prepare CALFIN front retreat.
"""

from typing import Union
import earthaccess
from pathlib import Path

def download_calfin(doi: str = "10.5067/7FILV218JZA2", filter_str: str = "Greenland_polygons", result_dir: Union[Path, str] = "."):
    earthaccess.login()
    result = earthaccess.search_data(doi=doi)
    result_filtered = [granule for granule in result if filter_str in granule["umm"]["DataGranule"]["Identifiers"][0]["Identifier"]]
    files = earthaccess.download(result_filtered, result_dir)
    
if __name__ == "__main__":
    download_calfin()
