export PISM_PREFIX=$HOME/local/pism
export RAGIS_DIR=/import/c1/ICESHEET/ICESHEET/pism-ragis/
export odir=${RAGIS_DIR}/hindcasts/2026_01_ml
export grid=4800

python $RAGIS_DIR/hindcasts/hindcast.py --start 1910-01-01 --end 2010-01-01 --o_size medium --boot_file $RAGIS_DIR/data/dem/BedMachineGreenland-v5_g450m_0.nc --exstep monthly --spatial_ts ml_mass --data_dir $RAGIS_DIR/data  --grid_file $RAGIS_DIR/data/grids/pism-bedmachine-greenland.nc -s chinook-40 -n 20 -q t2small -w 8:00:00 --o_dir $odir -r $grid -e $RAGIS_DIR/uq/ensemble_gris_ragis_ml_calibrate.csv $RAGIS_DIR/calibration/2025_08_grimp_tw/state/g${grid}m_id_BAYES-MEDIAN_1980-1-1_1984-12-31.nc

export PISM_PREFIX=$HOME/local/pism
export RAGIS_DIR=/import/c1/ICESHEET/ICESHEET/pism-ragis/
export odir=${RAGIS_DIR}/hindcasts/2026_01_ml
export grid=4500

python $RAGIS_DIR/hindcasts/hindcast.py --start 1910-01-01 --end 2010-01-01 --o_size medium --boot_file $RAGIS_DIR/data/dem/BedMachineGreenland-v5_g450m_0.nc --exstep monthly --spatial_ts ml_mass --data_dir $RAGIS_DIR/data  --grid_file $RAGIS_DIR/data/grids/pism-bedmachine-greenland.nc -s chinook-40 -n 20 -q t2small -w 8:00:00 --o_dir $odir -r $grid -e $RAGIS_DIR/uq/ensemble_gris_ragis_ml_calibrate.csv $RAGIS_DIR/calibration/2025_08_grimp_tw/state/g${grid}m_id_BAYES-MEDIAN_1980-1-1_1984-12-31.nc

export PISM_PREFIX=$HOME/local/pism
export RAGIS_DIR=/import/c1/ICESHEET/ICESHEET/pism-ragis/
export odir=${RAGIS_DIR}/hindcasts/2026_01_ml
export grid=3600

python $RAGIS_DIR/hindcasts/hindcast.py --start 1910-01-01 --end 2010-01-01 --o_size medium --boot_file $RAGIS_DIR/data/dem/BedMachineGreenland-v5_g450m_0.nc --exstep monthly --spatial_ts ml_mass --data_dir $RAGIS_DIR/data  --grid_file $RAGIS_DIR/data/grids/pism-bedmachine-greenland.nc -s chinook-40 -n 40 -q t2small  -w 8:00:00 --o_dir $odir -r $grid -e $RAGIS_DIR/uq/ensemble_gris_ragis_ml_calibrate.csv $RAGIS_DIR/calibration/2025_08_grimp_tw/state/g${grid}m_id_BAYES-MEDIAN_1980-1-1_1984-12-31.nc

export PISM_PREFIX=$HOME/local/pism
export RAGIS_DIR=/import/c1/ICESHEET/ICESHEET/pism-ragis/
export odir=${RAGIS_DIR}/hindcasts/2026_01_ml
export grid=2400

python $RAGIS_DIR/hindcasts/hindcast.py --start 1910-01-01 --end 2010-01-01 --o_size medium --boot_file $RAGIS_DIR/data/dem/BedMachineGreenland-v5_g450m_0.nc --exstep monthly --spatial_ts ml_mass --data_dir $RAGIS_DIR/data  --grid_file $RAGIS_DIR/data/grids/pism-bedmachine-greenland.nc -s chinook-40 -n 80 -q t2small -w 8:00:00 --o_dir $odir -r $grid -e $RAGIS_DIR/uq/ensemble_gris_ragis_ml_calibrate.csv $RAGIS_DIR/calibration/2025_08_grimp_tw/state/g${grid}m_id_BAYES-MEDIAN_1980-1-1_1984-12-31.nc

export PISM_PREFIX=$HOME/local/pism
export RAGIS_DIR=/import/c1/ICESHEET/ICESHEET/pism-ragis/
export odir=${RAGIS_DIR}/hindcasts/2026_01_ml
export grid=1800

python $RAGIS_DIR/hindcasts/hindcast.py --start 1910-01-01 --end 2010-01-01 --o_size medium --boot_file $RAGIS_DIR/data/dem/BedMachineGreenland-v5_g450m_0.nc --exstep monthly --spatial_ts ml_mass --data_dir $RAGIS_DIR/data  --grid_file $RAGIS_DIR/data/grids/pism-bedmachine-greenland.nc -s chinook-40 -n 80 -q t2small -w 20:00:00 --o_dir $odir -r $grid -e $RAGIS_DIR/uq/ensemble_gris_ragis_ml_calibrate.csv $RAGIS_DIR/calibration/2025_08_grimp_tw/state/g${grid}m_id_BAYES-MEDIAN_1980-1-1_1984-12-31.nc

export PISM_PREFIX=$HOME/local/pism
export RAGIS_DIR=/import/c1/ICESHEET/ICESHEET/pism-ragis/
export odir=${RAGIS_DIR}/hindcasts/2026_01_ml
export grid=1200

python $RAGIS_DIR/hindcasts/hindcast.py --start 1910-01-01 --end 2010-01-01 --o_size medium --boot_file $RAGIS_DIR/data/dem/BedMachineGreenland-v5_g450m_0.nc --exstep monthly --spatial_ts ml_mass --data_dir $RAGIS_DIR/data  --grid_file $RAGIS_DIR/data/grids/pism-bedmachine-greenland.nc -s chinook-40 -n 160 -q t2standard -w 96:00:00 --o_dir $odir -r $grid -e $RAGIS_DIR/uq/ensemble_gris_ragis_ml_calibrate.csv $RAGIS_DIR/calibration/2025_08_grimp_tw/state/g${grid}m_id_BAYES-MEDIAN_1980-1-1_1984-12-31.nc

export PISM_PREFIX=$HOME/local/pism
export RAGIS_DIR=/import/c1/ICESHEET/ICESHEET/pism-ragis/
export odir=${RAGIS_DIR}/hindcasts/2026_01_ml
export grid=900

python $RAGIS_DIR/hindcasts/hindcast.py --start 1910-01-01 --end 2010-01-01 --o_size medium --boot_file $RAGIS_DIR/data/dem/BedMachineGreenland-v5_g450m_0.nc --exstep monthly --spatial_ts ml_mass --data_dir $RAGIS_DIR/data  --grid_file $RAGIS_DIR/data/grids/pism-bedmachine-greenland.nc -s chinook-40 -n 240 -q t2standard -w 96:00:00 --o_dir $odir -r $grid -e $RAGIS_DIR/uq/ensemble_gris_ragis_ml_calibrate.csv $RAGIS_DIR/calibration/2025_08_grimp_tw/state/g${grid}m_id_BAYES-MEDIAN_1980-1-1_1984-12-31.nc

export PISM_PREFIX=$HOME/local/pism
export RAGIS_DIR=/import/c1/ICESHEET/ICESHEET/pism-ragis/
export odir=${RAGIS_DIR}/hindcasts/2026_01_ml
export grid=600

python $RAGIS_DIR/hindcasts/hindcast.py --start 1910-01-01 --end 2010-01-01 --o_size medium --boot_file $RAGIS_DIR/data/dem/BedMachineGreenland-v5_g450m_0.nc --exstep monthly --spatial_ts ml_mass --data_dir $RAGIS_DIR/data  --grid_file $RAGIS_DIR/data/grids/pism-bedmachine-greenland.nc -s chinook-40 -n 400 -q t2standard -w 168:00:00 --o_dir $odir -r $grid -e $RAGIS_DIR/uq/ensemble_gris_ragis_ml_calibrate.csv $RAGIS_DIR/calibration/2025_08_grimp_tw/state/g${grid}m_id_BAYES-MEDIAN_1980-1-1_1984-12-31.nc

