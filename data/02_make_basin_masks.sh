#!/bin/bash

ores=$1

x_min=-678650
x_max=905350
y_min=-3371600
y_max=-635600


mkdir -p basin_masks
for region in "SE" "CE" "NE" "SW" "CW" "NW"; do
gdal_rasterize  -where "SUBREGION1='${region}'" -l GRE_Basins_IMBIE2_v1.3_epsg3413 -tr $ores $ores -a_nodata -9999 -burn 1 -te $x_min $y_min $x_max $y_max -ot Byte basins/GRE_Basins_IMBIE2_v1.3_epsg3413.shp basin_masks/${region}_mask_epsg3413_g${ores}m.nc
done
