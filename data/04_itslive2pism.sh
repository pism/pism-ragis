#!/bin/bash

set -x -e

export HDF5_USE_FILE_LOCKING=FALSE
options='-overwrite  -s_srs EPSG:3413 -t_srs EPSG:3413 -r average -co FORMAT=NC4 -co COMPRESS=DEFLATE -co ZLEVEL=2'
x_min=-678650
y_min=-3371600
x_max=905350
y_max=-635600

itslivedir=itslive
itslive=$1
for grid in 1800 1500 1200 900 600 450; do
    itslivepism=g{$grid}m_$itslive
    for var in v vx vy vx_err vy_err; do
	gdalwarp $options  -dstnodata 0 -te $x_min $y_min $x_max $y_max -tr $grid $grid  NETCDF:$itslive:$var $itslivedir/${var}_$itslivepism
	ncatted -a units,Band1,o,c,"m/yr" $itslivedir/${var}_$itslivepism
	ncrename -v Band1,$var $itslivedir/${var}_$itslivepism
    done
    cdo -f nc4 -z zip_2 merge $itslivedir/*_$itslivepism $itslivepism
done
