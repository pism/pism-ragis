
#!/bin/bash
odir=2024_02_ragis
grid=900
mkdir -p ${odir}/processed
for year in 1985 2000 2007; do
for id in BAYES-MEDIAN BAYES-MEDIAN-FR BAYES-MEDIAN-FR-OFF; do
cdo -L -f nc4 -z zip_2 -O ifthen -timmean -selvar,velsurf_mag -selyear,$year $odir/spatial/ex_gris_g${grid}m_v2024-02_RAGIS_id_${id}_1980-1-1_2020-1-1.nc -timmean -selvar,velsurf_mag,usurf -selyear,$year $odir/spatial/ex_gris_g${grid}m_v2024-02_RAGIS_id_${id}_1980-1-1_2020-1-1.nc $odir/processed/velsurf_mag_ex_gris_g${grid}m_v2024-02_RAGIS_id_${id}_${year}.nc
done
done

odir=2024_02_grimp
grid=900
mkdir -p ${odir}/processed
for year in 1985 2000 2007; do
for id in BAYES-MEDIAN BAYES-MEDIAN-FR BAYES-MEDIAN-FR-OFF; do cdo -L -f nc4 -z zip_2 -O ifthen -timmean -selvar,velsurf_mag -selyear,$year $odir/spatial/ex_gris_g${grid}m_v2023_GRIMP_id_${id}_1980-1-1_2020-1-1.nc -timmean -selvar,velsurf_mag,usurf -selyear,$year $odir/spatial/ex_gris_g${grid}m_v2023_GRIMP_id_${id}_1980-1-1_2020-1-1.nc $odir/processed/velsurf_mag_ex_gris_g${grid}m_v2023_GRIMP_id_${id}_${year}.nc
done
done
