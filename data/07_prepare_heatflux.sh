#!/bin/bash

# (c) 2021-22 Andy Aschwanden

set -x -e

mkdir -p bheatflux
OUTFILE=bheatflux/geothermal_heat_flow_map_10km.nc
wget -O  $OUTFILE  https://dataverse.geus.dk/api/access/datafile/:persistentId?persistentId=doi:10.22008/FK2/F9P03L/IWCOYX


ncrename -v GHF,bheatflx $OUTFILE
ncatted -a proj4,global,d,, -a units,bheatflx,o,c,"mW m^-2" -a _FillValue,bheatflx,d,, -a missing_value,bheatflx,d,, $OUTFILE
