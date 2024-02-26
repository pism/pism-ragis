#!/bin/bash

set -x -e

export HDF5_USE_FILE_LOCKING=FALSE
options='-overwrite  -t_srs EPSG:3413 -r average -co FORMAT=NC4 -co COMPRESS=DEFLATE -co ZLEVEL=2'
x_min=-678650
y_min=-3371600
x_max=905350
y_max=-635600

itslivedir=itslive
itslive=GRE_G0240_0000.nc
for grid in 1800 1500 1200 900 600 450; do
    itslivepism=GRE_G${grid}_0000.nc
    gdalwarp $options  -dstnodata 0 -te $x_min $y_min $x_max $y_max -tr $grid $grid  NETCDF:$itslivedir/$itslive:v $itslivedir/$itslivepism
    ncrename -v Band1,velsurf_mag $itslivedir/$itslivepism
done

itslive=GRE_G0240_1985.nc
for grid in 1800 1500 1200 900 600 450; do
    itslivepism=GRE_G${grid}_1985.nc
    gdalwarp $options  -dstnodata 0 -te $x_min $y_min $x_max $y_max -tr $grid $grid  NETCDF:$itslivedir/$itslive:v $itslivedir/$itslivepism
    ncrename -v Band1,velsurf_mag $itslivedir/$itslivepism
done
