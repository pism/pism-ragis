#!/bin/bash
#!/usr/bin/env python3
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


r=$1
mkdir -p glaciers
for ugid in 225; do
    python make_basin_mask.py -r $r --attribute_value $ugid --attribute UGID --layers Greenland_Basins_PS_v1.4.2_1980_epsg3413 --basin_file glaciers/Greenland_Basins_PS_v1.4.2_1980_epsg3413.shp glaciers/gris_g${r}m_mask_basin_${b}.nc;
done
