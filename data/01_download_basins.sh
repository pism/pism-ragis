
#!/bin/bash

mkdir -p basins
wget -nc http://imbie.org/wp-content/uploads/2016/09/GRE_Basins_IMBIE2_v1.3.zip
unzip -u GRE_Basins_IMBIE2_v1.3.zip
mv GRE_Basins_IMBIE2_v1.3.cpg  GRE_Basins_IMBIE2_v1.3.dbf  GRE_Basins_IMBIE2_v1.3.prj GRE_Basins_IMBIE2_v1.3.qml  GRE_Basins_IMBIE2_v1.3.qpj GRE_Basins_IMBIE2_v1.3.shp  GRE_Basins_IMBIE2_v1.3.shx  basins

ogr2ogr -t_srs EPSG:3413 basins/GRE_Basins_IMBIE2_v1.3_epsg3413.shp basins/GRE_Basins_IMBIE2_v1.3.shp

wget -nc http://www.cpom.ucl.ac.uk/imbie/basins/mouginot_2019/Greenland_Basins_PS_v1.4.2.tar.gz
tar -xf Greenland_Basins_PS_v1.4.2.tar.gz
mv Greenland_Basins_PS_v1.4.2.* basins/
