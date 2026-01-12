export PISM_PREFIX=$HOME/local/pism
export RAGIS_DIR=/import/c1/ICESHEET/ICESHEET/pism-ragis/
export odir=${RAGIS_DIR}/hindcasts/2026_01_grimp
export grid=1200

python $RAGIS_DIR/hindcasts/hindcast.py --start 1978-01-01 --end 2020-12-31 --boot_file $RAGIS_DIR/data/dem/BedMachineGreenland-v5_0.nc --exstep monthly --spatial_ts ragis --data_dir $RAGIS_DIR/data  --grid_file $RAGIS_DIR/data/grids/pism-bedmachine-greenland.nc -s chinook -n 160 -q t2standard -w 36:00:00 --o_dir $odir -r $grid -e $RAGIS_DIR/uq/ensemble_gris_ragis_mode.csv $RAGIS_DIR/calibration/2025_08_grimp_tw/state/g1200m_id_BAYES-MEDIAN_1980-1-1_1984-12-31.nc


export PISM_PREFIX=$HOME/local/pism
export RAGIS_DIR=/import/c1/ICESHEET/ICESHEET/pism-ragis/
export odir=${RAGIS_DIR}/hindcasts/2026_01_svd
export grid=1200

python $RAGIS_DIR/hindcasts/hindcast.py --start 1978-01-01 --end 2020-12-31 --boot_file $RAGIS_DIR/data/dem/bedmachine1980_reconstructed_g600_SVD_0.nc  --exstep monthly --spatial_ts ragis --data_dir $RAGIS_DIR/data  --grid_file $RAGIS_DIR/data/grids/pism-bedmachine-greenland.nc -s chinook -n 160 -q t2standard -w 36:00:00 --o_dir $odir -r $grid -e $RAGIS_DIR/uq/ensemble_gris_ragis_mode.csv $RAGIS_DIR/calibration/2026_01_svd_tw/state/g1200m_id_BAYES-MEDIAN_1980-1-1_1984-12-31.nc

export PISM_PREFIX=$HOME/local/pism
export RAGIS_DIR=/import/c1/ICESHEET/ICESHEET/pism-ragis/
export odir=${RAGIS_DIR}/hindcasts/2026_01_krig
export grid=1200

python $RAGIS_DIR/hindcasts/hindcast.py --start 1978-01-01 --end 2020-12-31 --boot_file $RAGIS_DIR/data/dem/bedmachine1980_reconstructed_g600_kriging_0.nc  --exstep monthly --spatial_ts ragis --data_dir $RAGIS_DIR/data  --grid_file $RAGIS_DIR/data/grids/pism-bedmachine-greenland.nc -s chinook -n 160 -q t2standard -w 36:00:00 --o_dir $odir -r $grid -e $RAGIS_DIR/uq/ensemble_gris_ragis_mode.csv $RAGIS_DIR/calibration/2026_01_krig_tw/state/g1200m_id_BAYES-MEDIAN_1980-1-1_1984-12-31.nc



export PISM_PREFIX=$HOME/local/pism
export RAGIS_DIR=/import/c1/ICESHEET/ICESHEET/pism-ragis/
export odir=${RAGIS_DIR}/hindcasts/2026_01_grimp
export grid=600

python $RAGIS_DIR/hindcasts/hindcast.py --start 1978-01-01 --end 1986-01-01 --boot_file $RAGIS_DIR/data/dem/BedMachineGreenland-v5_0.nc --exstep monthly --spatial_ts ragis --data_dir $RAGIS_DIR/data  --grid_file $RAGIS_DIR/data/grids/pism-bedmachine-greenland.nc -s chinook -n 320 -q t2standard -w 48:00:00 --o_dir $odir -r $grid -e $RAGIS_DIR/uq/ensemble_gris_ragis_mode.csv $RAGIS_DIR/calibration/2026_01_grimp/state/g600m_id_BAYES-MEDIAN_1980-1-1_1984-12-31.nc


export PISM_PREFIX=$HOME/local/pism
export RAGIS_DIR=/import/c1/ICESHEET/ICESHEET/pism-ragis/
export odir=${RAGIS_DIR}/hindcasts/2026_01_svd
export grid=600

python $RAGIS_DIR/hindcasts/hindcast.py --start 1978-01-01 --end 1986-01-01 --boot_file $RAGIS_DIR/data/dem/bedmachine1980_reconstructed_g600_SVD_0.nc  --exstep monthly --spatial_ts ragis --data_dir $RAGIS_DIR/data  --grid_file $RAGIS_DIR/data/grids/pism-bedmachine-greenland.nc -s chinook -n 320 -q t2standard -w 48:00:00 --o_dir $odir -r $grid -e $RAGIS_DIR/uq/ensemble_gris_ragis_mode.csv $RAGIS_DIR/calibration/2026_01_svd_tw/state/g600m_id_BAYES-MEDIAN_1980-1-1_1984-12-31.nc

export PISM_PREFIX=$HOME/local/pism
export RAGIS_DIR=/import/c1/ICESHEET/ICESHEET/pism-ragis/
export odir=${RAGIS_DIR}/hindcasts/2026_01_krig
export grid=600

python $RAGIS_DIR/hindcasts/hindcast.py --start 1978-01-01 --end 1986-01-01 --boot_file $RAGIS_DIR/data/dem/bedmachine1980_reconstructed_g600_kriging_0.nc  --exstep monthly --spatial_ts ragis --data_dir $RAGIS_DIR/data  --grid_file $RAGIS_DIR/data/grids/pism-bedmachine-greenland.nc -s chinook -n 320 -q t2standard -w 48:00:00 --o_dir $odir -r $grid -e $RAGIS_DIR/uq/ensemble_gris_ragis_mode.csv $RAGIS_DIR/calibration/2026_01_krig_tw/state/g600m_id_BAYES-MEDIAN_1980-1-1_1984-12-31.nc



####################################################

export PISM_PREFIX=$HOME/local/pism
export RAGIS_DIR=/import/c1/ICESHEET/ICESHEET/pism-ragis/
export odir=${RAGIS_DIR}/hindcasts/2025_08_ml
export grid=4800

python $RAGIS_DIR/hindcasts/hindcast.py --start 1900-01-01 --end 2010-01-01 --o_size medium --boot_file $RAGIS_DIR/data/dem/BedMachineGreenland-v5_g450m_0.nc --exstep monthly --spatial_ts ml --data_dir $RAGIS_DIR/data  --grid_file $RAGIS_DIR/data/grids/pism-bedmachine-greenland.nc -s chinook-40 -n 20 -q t2small -w 8:00:00 --o_dir $odir -r $grid -e $RAGIS_DIR/uq/ensemble_gris_ragis_mode.csv $RAGIS_DIR/calibration/2025_08_grimp_tw/state/g${grid}m_id_BAYES-MEDIAN_1980-1-1_1984-12-31.nc

export PISM_PREFIX=$HOME/local/pism
export RAGIS_DIR=/import/c1/ICESHEET/ICESHEET/pism-ragis/
export odir=${RAGIS_DIR}/hindcasts/2025_08_ml
export grid=4500

python $RAGIS_DIR/hindcasts/hindcast.py --start 1900-01-01 --end 2010-01-01 --o_size medium --boot_file $RAGIS_DIR/data/dem/BedMachineGreenland-v5_g450m_0.nc --exstep monthly --spatial_ts ml --data_dir $RAGIS_DIR/data  --grid_file $RAGIS_DIR/data/grids/pism-bedmachine-greenland.nc -s chinook-40 -n 20 -q t2small -w 8:00:00 --o_dir $odir -r $grid -e $RAGIS_DIR/uq/ensemble_gris_ragis_mode.csv $RAGIS_DIR/calibration/2025_08_grimp_tw/state/g${grid}m_id_BAYES-MEDIAN_1980-1-1_1984-12-31.nc

export PISM_PREFIX=$HOME/local/pism
export RAGIS_DIR=/import/c1/ICESHEET/ICESHEET/pism-ragis/
export odir=${RAGIS_DIR}/hindcasts/2025_08_ml
export grid=3600

python $RAGIS_DIR/hindcasts/hindcast.py --start 1900-01-01 --end 2010-01-01 --o_size medium --boot_file $RAGIS_DIR/data/dem/BedMachineGreenland-v5_g450m_0.nc --exstep monthly --spatial_ts ml --data_dir $RAGIS_DIR/data  --grid_file $RAGIS_DIR/data/grids/pism-bedmachine-greenland.nc -s chinook-40 -n 40 -q t2small  -w 8:00:00 --o_dir $odir -r $grid -e $RAGIS_DIR/uq/ensemble_gris_ragis_mode.csv $RAGIS_DIR/calibration/2025_08_grimp_tw/state/g${grid}m_id_BAYES-MEDIAN_1980-1-1_1984-12-31.nc

export PISM_PREFIX=$HOME/local/pism
export RAGIS_DIR=/import/c1/ICESHEET/ICESHEET/pism-ragis/
export odir=${RAGIS_DIR}/hindcasts/2025_08_ml
export grid=2400

python $RAGIS_DIR/hindcasts/hindcast.py --start 1900-01-01 --end 2010-01-01 --o_size medium --boot_file $RAGIS_DIR/data/dem/BedMachineGreenland-v5_g450m_0.nc --exstep monthly --spatial_ts ml --data_dir $RAGIS_DIR/data  --grid_file $RAGIS_DIR/data/grids/pism-bedmachine-greenland.nc -s chinook-40 -n 80 -q t2small -w 8:00:00 --o_dir $odir -r $grid -e $RAGIS_DIR/uq/ensemble_gris_ragis_mode.csv $RAGIS_DIR/calibration/2025_08_grimp_tw/state/g${grid}m_id_BAYES-MEDIAN_1980-1-1_1984-12-31.nc

export PISM_PREFIX=$HOME/local/pism
export RAGIS_DIR=/import/c1/ICESHEET/ICESHEET/pism-ragis/
export odir=${RAGIS_DIR}/hindcasts/2025_08_ml
export grid=1800

python $RAGIS_DIR/hindcasts/hindcast.py --start 1900-01-01 --end 2010-01-01 --o_size medium --boot_file $RAGIS_DIR/data/dem/BedMachineGreenland-v5_g450m_0.nc --exstep monthly --spatial_ts ml --data_dir $RAGIS_DIR/data  --grid_file $RAGIS_DIR/data/grids/pism-bedmachine-greenland.nc -s chinook-40 -n 80 -q t2small -w 20:00:00 --o_dir $odir -r $grid -e $RAGIS_DIR/uq/ensemble_gris_ragis_mode.csv $RAGIS_DIR/calibration/2025_08_grimp_tw/state/g${grid}m_id_BAYES-MEDIAN_1980-1-1_1984-12-31.nc

export PISM_PREFIX=$HOME/local/pism
export RAGIS_DIR=/import/c1/ICESHEET/ICESHEET/pism-ragis/
export odir=${RAGIS_DIR}/hindcasts/2025_08_ml
export grid=1200

python $RAGIS_DIR/hindcasts/hindcast.py --start 1900-01-01 --end 2010-01-01 --o_size medium --boot_file $RAGIS_DIR/data/dem/BedMachineGreenland-v5_g450m_0.nc --exstep monthly --spatial_ts ml --data_dir $RAGIS_DIR/data  --grid_file $RAGIS_DIR/data/grids/pism-bedmachine-greenland.nc -s chinook-40 -n 160 -q t2standard -w 96:00:00 --o_dir $odir -r $grid -e $RAGIS_DIR/uq/ensemble_gris_ragis_mode.csv $RAGIS_DIR/calibration/2025_08_grimp_tw/state/g${grid}m_id_BAYES-MEDIAN_1980-1-1_1984-12-31.nc

export PISM_PREFIX=$HOME/local/pism
export RAGIS_DIR=/import/c1/ICESHEET/ICESHEET/pism-ragis/
export odir=${RAGIS_DIR}/hindcasts/2025_08_ml
export grid=900

python $RAGIS_DIR/hindcasts/hindcast.py --start 1910-01-01 --end 2010-01-01 --o_size medium --boot_file $RAGIS_DIR/data/dem/BedMachineGreenland-v5_g450m_0.nc --exstep monthly --spatial_ts ml --data_dir $RAGIS_DIR/data  --grid_file $RAGIS_DIR/data/grids/pism-bedmachine-greenland.nc -s chinook-40 -n 240 -q t2standard -w 96:00:00 --o_dir $odir -r $grid -e $RAGIS_DIR/uq/ensemble_gris_ragis_mode.csv $RAGIS_DIR/calibration/2025_08_grimp_tw/state/g${grid}m_id_BAYES-MEDIAN_1980-1-1_1984-12-31.nc

export PISM_PREFIX=$HOME/local/pism
export RAGIS_DIR=/import/c1/ICESHEET/ICESHEET/pism-ragis/
export odir=${RAGIS_DIR}/hindcasts/2025_08_ml
export grid=600

python $RAGIS_DIR/hindcasts/hindcast.py --start 1900-01-01 --end 2010-01-01 --o_size medium --boot_file $RAGIS_DIR/data/dem/BedMachineGreenland-v5_g450m_0.nc --exstep monthly --spatial_ts ml --data_dir $RAGIS_DIR/data  --grid_file $RAGIS_DIR/data/grids/pism-bedmachine-greenland.nc -s chinook-40 -n 400 -q t2standard -w 96:00:00 --o_dir $odir -r $grid -e $RAGIS_DIR/uq/ensemble_gris_ragis_mode.csv $RAGIS_DIR/calibration/2025_08_grimp_tw/state/g${grid}m_id_BAYES-MEDIAN_1980-1-1_1984-12-31.nc



export PISM_PREFIX=$HOME/local/pism
<<<<<<< Updated upstream
export RAGIS_DIR=/work2/08523/aaschwa/stampede3/pism-ragis/
export odir=${SCRATCH}/hindcasts/2025_02_mode
export grid=4500

python $RAGIS_DIR/hindcasts/hindcast.py --start 1978-01-01 --end 2020-12-31 --boot_file $RAGIS_DIR/data/dem/BedMachineGreenland-v5_0.nc --exstep monthly --spatial_ts ml --data_dir $RAGIS_DIR/data  --grid_file $RAGIS_DIR/data/grids/pism-bedmachine-greenland.nc -s stampede3 -n 96 -q skx -w 1:00:00 --o_dir $odir -r $grid -e $RAGIS_DIR/uq/ensemble_gris_ragis_mode.csv $RAGIS_DIR/calibration/2024_11_grimp_tw/state/g1200m_id_BAYES-MEDIAN_1980-1-1_1984-12-31.nc




export PISM_PREFIX=$HOME/local/pism
export RAGIS_DIR=/nobackup/aaschwan/pism-ragis/
export odir=${RAGIS_DIR}/hindcasts/2025_02_mode
export grid=4500

python $RAGIS_DIR/hindcasts/hindcast.py --start 1978-01-01 --end 2020-12-31 --boot_file $RAGIS_DIR/data/dem/BedMachineGreenland-v5_0.nc --exstep monthly --spatial_ts ml --data_dir $RAGIS_DIR/data  --grid_file $RAGIS_DIR/data/grids/pism-bedmachine-greenland.nc -s electra_skylake -n 80 -q normal -w 1:00:00 --o_dir $odir -r $grid -e $RAGIS_DIR/uq/ensemble_gris_ragis_mode.csv $RAGIS_DIR/calibration/2024_11_grimp_tw/state/g1200m_id_BAYES-MEDIAN_1980-1-1_1984-12-31.nc

export PISM_PREFIX=$HOME/local/pism
export RAGIS_DIR=/nobackup/aaschwan/pism-ragis/
export odir=${RAGIS_DIR}/hindcasts/2025_02_mode
export grid=3600

python $RAGIS_DIR/hindcasts/hindcast.py --start 1978-01-01 --end 2020-12-31 --boot_file $RAGIS_DIR/data/dem/BedMachineGreenland-v5_0.nc --exstep monthly --spatial_ts ml --data_dir $RAGIS_DIR/data  --grid_file $RAGIS_DIR/data/grids/pism-bedmachine-greenland.nc -s electra_skylake -n 80 -q normal -w 2:00:00 --o_dir $odir -r $grid -e $RAGIS_DIR/uq/ensemble_gris_ragis_mode.csv $RAGIS_DIR/calibration/2024_11_grimp_tw/state/g1200m_id_BAYES-MEDIAN_1980-1-1_1984-12-31.nc

export PISM_PREFIX=$HOME/local/pism
export RAGIS_DIR=/nobackup/aaschwan/pism-ragis/
export odir=${RAGIS_DIR}/hindcasts/2025_02_mode
export grid=2400

python $RAGIS_DIR/hindcasts/hindcast.py --start 1978-01-01 --end 2020-12-31 --boot_file $RAGIS_DIR/data/dem/BedMachineGreenland-v5_0.nc --exstep monthly --spatial_ts ml --data_dir $RAGIS_DIR/data  --grid_file $RAGIS_DIR/data/grids/pism-bedmachine-greenland.nc -s electra_skylake -n 80 -q normal -w 4:00:00 --o_dir $odir -r $grid -e $RAGIS_DIR/uq/ensemble_gris_ragis_mode.csv $RAGIS_DIR/calibration/2024_11_grimp_tw/state/g1200m_id_BAYES-MEDIAN_1980-1-1_1984-12-31.nc

export PISM_PREFIX=$HOME/local/pism
export RAGIS_DIR=/nobackup/aaschwan/pism-ragis/
export odir=${RAGIS_DIR}/hindcasts/2025_02_mode
export grid=1800

python $RAGIS_DIR/hindcasts/hindcast.py --start 1978-01-01 --end 2020-12-31 --boot_file $RAGIS_DIR/data/dem/BedMachineGreenland-v5_0.nc --exstep monthly --spatial_ts ml --data_dir $RAGIS_DIR/data  --grid_file $RAGIS_DIR/data/grids/pism-bedmachine-greenland.nc -s electra_skylake -n 80 -q normal -w 8:00:00 --o_dir $odir -r $grid -e $RAGIS_DIR/uq/ensemble_gris_ragis_mode.csv $RAGIS_DIR/calibration/2024_11_grimp_tw/state/g1200m_id_BAYES-MEDIAN_1980-1-1_1984-12-31.nc

export PISM_PREFIX=$HOME/local/pism
export RAGIS_DIR=/nobackup/aaschwan/pism-ragis/
export odir=${RAGIS_DIR}/hindcasts/2025_02_mode
export grid=1200


python $RAGIS_DIR/hindcasts/hindcast.py --start 1978-01-01 --end 2020-12-31 --boot_file $RAGIS_DIR/data/dem/BedMachineGreenland-v5_0.nc --exstep monthly --spatial_ts ml --data_dir $RAGIS_DIR/data  --grid_file $RAGIS_DIR/data/grids/pism-bedmachine-greenland.nc -s electra_skylake -n 160 -q long -w 36:00:00 --o_dir $odir -r $grid -e $RAGIS_DIR/uq/ensemble_gris_ragis_mode.csv $RAGIS_DIR/calibration/2024_11_grimp_tw/state/g1200m_id_BAYES-MEDIAN_1980-1-1_1984-12-31.nc

export PISM_PREFIX=$HOME/local/pism
export RAGIS_DIR=/nobackup/aaschwan/pism-ragis/
export odir=${RAGIS_DIR}/hindcasts/2025_02_mode
export grid=900

python $RAGIS_DIR/hindcasts/hindcast.py --start 1978-01-01 --end 2020-12-31 --boot_file $RAGIS_DIR/data/dem/BedMachineGreenland-v5_0.nc --exstep monthly --spatial_ts ml --data_dir $RAGIS_DIR/data  --grid_file $RAGIS_DIR/data/grids/pism-bedmachine-greenland.nc -s electra_skylake -n 240 -q long -w 72:00:00 --o_dir $odir -r $grid -e $RAGIS_DIR/uq/ensemble_gris_ragis_mode.csv $RAGIS_DIR/calibration/2024_11_grimp_tw/state/g1200m_id_BAYES-MEDIAN_1980-1-1_1984-12-31.nc





export PISM_PREFIX=$HOME/local-intel/pism
export RAGIS_DIR=/import/c1/ICESHEET/ICESHEET/pism-ragis/
export odir=${RAGIS_DIR}/hindcasts/2025_02_mode
export grid=4500

python $RAGIS_DIR/hindcasts/hindcast.py --start 1978-01-01 --end 2020-12-31 --boot_file $RAGIS_DIR/data/dem/BedMachineGreenland-v5_0.nc --exstep monthly --spatial_ts ml --data_dir $RAGIS_DIR/data  --grid_file $RAGIS_DIR/data/grids/pism-bedmachine-greenland.nc -s chinook-rl8 -n 80 -q t1small -w 2:00:00 --o_dir $odir -r $grid -e $RAGIS_DIR/uq/ensemble_gris_ragis_mode.csv $RAGIS_DIR/calibration/2024_11_grimp_tw/state/g1200m_id_BAYES-MEDIAN_1980-1-1_1984-12-31.nc


export PISM_PREFIX=$HOME/local-intel/pism
export RAGIS_DIR=/import/c1/ICESHEET/ICESHEET/pism-ragis/
export odir=${RAGIS_DIR}/hindcasts/2025_02_mode
export grid=3600

python $RAGIS_DIR/hindcasts/hindcast.py --start 1978-01-01 --end 2020-12-31 --boot_file $RAGIS_DIR/data/dem/BedMachineGreenland-v5_0.nc --exstep monthly --spatial_ts ml --data_dir $RAGIS_DIR/data  --grid_file $RAGIS_DIR/data/grids/pism-bedmachine-greenland.nc -s chinook-rl8 -n 80 -q t1small -w 4:00:00 --o_dir $odir -r $grid -e $RAGIS_DIR/uq/ensemble_gris_ragis_mode.csv $RAGIS_DIR/calibration/2024_11_grimp_tw/state/g1200m_id_BAYES-MEDIAN_1980-1-1_1984-12-31.nc

export PISM_PREFIX=$HOME/local-intel/pism
export RAGIS_DIR=/import/c1/ICESHEET/ICESHEET/pism-ragis/
export odir=${RAGIS_DIR}/hindcasts/2025_02_mode
export grid=2400

python $RAGIS_DIR/hindcasts/hindcast.py --start 1978-01-01 --end 2020-12-31 --boot_file $RAGIS_DIR/data/dem/BedMachineGreenland-v5_0.nc --exstep monthly --spatial_ts ml --data_dir $RAGIS_DIR/data  --grid_file $RAGIS_DIR/data/grids/pism-bedmachine-greenland.nc -s chinook-rl8 -n 80 -q t1small -w 6:00:00 --o_dir $odir -r $grid -e $RAGIS_DIR/uq/ensemble_gris_ragis_mode.csv $RAGIS_DIR/calibration/2024_11_grimp_tw/state/g1200m_id_BAYES-MEDIAN_1980-1-1_1984-12-31.nc

export PISM_PREFIX=$HOME/local-intel/pism
export RAGIS_DIR=/import/c1/ICESHEET/ICESHEET/pism-ragis/
export odir=${RAGIS_DIR}/hindcasts/2025_02_mode
export grid=1800

python $RAGIS_DIR/hindcasts/hindcast.py --start 1978-01-01 --end 2020-12-31 --boot_file $RAGIS_DIR/data/dem/BedMachineGreenland-v5_0.nc --exstep monthly --spatial_ts ml --data_dir $RAGIS_DIR/data  --grid_file $RAGIS_DIR/data/grids/pism-bedmachine-greenland.nc -s electra_skylake -n 80 -q normal -w 8:00:00 --o_dir $odir -r $grid -e $RAGIS_DIR/uq/ensemble_gris_ragis_mode.csv $RAGIS_DIR/calibration/2024_11_grimp_tw/state/g1200m_id_BAYES-MEDIAN_1980-1-1_1984-12-31.nc


export PISM_PREFIX=$HOME/local-intel/pism
export RAGIS_DIR=/import/c1/ICESHEET/ICESHEET/pism-ragis/
export odir=${RAGIS_DIR}/hindcasts/2025_02_mode
export grid=1200

python $RAGIS_DIR/hindcasts/hindcast.py --start 1978-01-01 --end 2020-12-31 --boot_file $RAGIS_DIR/data/dem/BedMachineGreenland-v5_0.nc --exstep monthly --spatial_ts ml --data_dir $RAGIS_DIR/data  --grid_file $RAGIS_DIR/data/grids/pism-bedmachine-greenland.nc -s chinook-rl8 -n 160 -q t1standard -w 36:00:00 --o_dir $odir -r $grid -e $RAGIS_DIR/uq/ensemble_gris_ragis_mode.csv $RAGIS_DIR/calibration/2024_11_grimp_tw/state/g1200m_id_BAYES-MEDIAN_1980-1-1_1984-12-31.nc

export PISM_PREFIX=$HOME/local-intel/pism
export RAGIS_DIR=/import/c1/ICESHEET/ICESHEET/pism-ragis/
export odir=${RAGIS_DIR}/hindcasts/2025_02_mode
export grid=900

python $RAGIS_DIR/hindcasts/hindcast.py --start 1978-01-01 --end 2020-12-31 --boot_file $RAGIS_DIR/data/dem/BedMachineGreenland-v5_0.nc --exstep monthly --spatial_ts ml --data_dir $RAGIS_DIR/data  --grid_file $RAGIS_DIR/data/grids/pism-bedmachine-greenland.nc -s electra_skylake -n 240 -q long -w 72:00:00 --o_dir $odir -r $grid -e $RAGIS_DIR/uq/ensemble_gris_ragis_mode.csv $RAGIS_DIR/calibration/2024_11_grimp_tw/state/g1200m_id_BAYES-MEDIAN_1980-1-1_1984-12-31.nc
=======
export RAGIS_DIR=/nobackup/aaschwan/pism-ragis/
export odir=${RAGIS_DIR}/hindcasts/2025_01_lhs_800
export grid=1200

python $RAGIS_DIR/hindcasts/hindcast.py --start 1978-01-01 --end 2020-12-31 --boot_file $RAGIS_DIR/data/dem/BedMachineGreenland-v5_0.nc --exstep yearly --spatial_ts ragis --data_dir $RAGIS_DIR/data  --grid_file $RAGIS_DIR/data/grids/pism-bedmachine-greenland.nc -s electra_skylake -n 160 -q long -w 72:00:00 --o_dir $odir -r $grid -e $RAGIS_DIR/uq/ensemble_gris_ragis_w_posterior_lhs_800.csv $RAGIS_DIR/calibration/2024_11_grimp_tw/state/g1200m_id_BAYES-MEDIAN_1980-1-1_1984-12-31.nc

for file in $odir/spatial/ex_g1200m_id_*_1978-01-01_2020-12-31.nc; do
qsub -v result_dir=$odir/basins,file=$file /nobackup/aaschwan/pism-ragis/hindcasts/extract_basins.sh ;
done
>>>>>>> Stashed changes



export PISM_PREFIX=$HOME/local/pism
export RAGIS_DIR=/nobackup/aaschwan/pism-ragis/
export odir=${RAGIS_DIR}/hindcasts/2024_12_mode
export grid=4500

<<<<<<< Updated upstream
python $RAGIS_DIR/hindcasts/hindcast.py --start 1978-01-01 --end 2020-12-31 --boot_file $RAGIS_DIR/data/dem/BedM
achineGreenland-v5_0.nc --exstep monthly --spatial_ts ragis --data_dir $RAGIS_DIR/data  --grid_file $RAGIS_DIR/d
ata/grids/pism-bedmachine-greenland.nc -s electra_skylake -n 80 -q normal -w 1:00:00 --o_dir $odir -r $grid -e $RAGIS_DIR/uq/ensemble_gris_ragis_mode.csv $RAGIS_DIR/calibration/2024_11_grimp_tw/state/g1200m_id_BAYES-MEDIAN_1980-1-1_1984-12-31.nc
=======
python $RAGIS_DIR/hindcasts/hindcast.py --start 1978-01-01 --end 2020-12-31 --boot_file $RAGIS_DIR/data/dem/BedMachineGreenland-v5_0.nc --exstep monthly --spatial_ts ragis --data_dir $RAGIS_DIR/data  --grid_file $RAGIS_DIR/data/grids/pism-bedmachine-greenland.nc -s electra_skylake -n 80 -q normal -w 1:00:00 --o_dir $odir -r $grid -e $RAGIS_DIR/uq/ensemble_gris_ragis_mode.csv $RAGIS_DIR/calibration/2024_11_grimp_tw/state/g1200m_id_BAYES-MEDIAN_1980-1-1_1984-12-31.nc
>>>>>>> Stashed changes

export PISM_PREFIX=$HOME/local/pism
export RAGIS_DIR=/nobackup/aaschwan/pism-ragis/
export odir=${RAGIS_DIR}/hindcasts/2024_12_mode
export grid=3600

python $RAGIS_DIR/hindcasts/hindcast.py --start 1978-01-01 --end 2020-12-31 --boot_file $RAGIS_DIR/data/dem/BedMachineGreenland-v5_0.nc --exstep monthly --spatial_ts ragis --data_dir $RAGIS_DIR/data  --grid_file $RAGIS_DIR/data/grids/pism-bedmachine-greenland.nc -s electra_skylake -n 80 -q normal -w 2:00:00 --o_dir $odir -r $grid -e $RAGIS_DIR/uq/ensemble_gris_ragis_mode.csv $RAGIS_DIR/calibration/2024_11_grimp_tw/state/g1200m_id_BAYES-MEDIAN_1980-1-1_1984-12-31.nc

export PISM_PREFIX=$HOME/local/pism
export RAGIS_DIR=/nobackup/aaschwan/pism-ragis/
export odir=${RAGIS_DIR}/hindcasts/2024_12_mode
export grid=2400

python $RAGIS_DIR/hindcasts/hindcast.py --start 1978-01-01 --end 2020-12-31 --boot_file $RAGIS_DIR/data/dem/BedMachineGreenland-v5_0.nc --exstep monthly --spatial_ts ragis --data_dir $RAGIS_DIR/data  --grid_file $RAGIS_DIR/data/grids/pism-bedmachine-greenland.nc -s electra_skylake -n 80 -q normal -w 4:00:00 --o_dir $odir -r $grid -e $RAGIS_DIR/uq/ensemble_gris_ragis_mode.csv $RAGIS_DIR/calibration/2024_11_grimp_tw/state/g1200m_id_BAYES-MEDIAN_1980-1-1_1984-12-31.nc

export PISM_PREFIX=$HOME/local/pism
export RAGIS_DIR=/nobackup/aaschwan/pism-ragis/
export odir=${RAGIS_DIR}/hindcasts/2024_12_mode
export grid=1800

python $RAGIS_DIR/hindcasts/hindcast.py --start 1978-01-01 --end 2020-12-31 --boot_file $RAGIS_DIR/data/dem/BedMachineGreenland-v5_0.nc --exstep monthly --spatial_ts ragis --data_dir $RAGIS_DIR/data  --grid_file $RAGIS_DIR/data/grids/pism-bedmachine-greenland.nc -s electra_skylake -n 80 -q normal -w 8:00:00 --o_dir $odir -r $grid -e $RAGIS_DIR/uq/ensemble_gris_ragis_mode.csv $RAGIS_DIR/calibration/2024_11_grimp_tw/state/g1200m_id_BAYES-MEDIAN_1980-1-1_1984-12-31.nc

<<<<<<< Updated upstream

=======
>>>>>>> Stashed changes
export PISM_PREFIX=$HOME/local/pism
export RAGIS_DIR=/nobackup/aaschwan/pism-ragis/
export odir=${RAGIS_DIR}/hindcasts/2024_12_mode
export grid=1200


python $RAGIS_DIR/hindcasts/hindcast.py --start 1978-01-01 --end 2020-12-31 --boot_file $RAGIS_DIR/data/dem/BedMachineGreenland-v5_0.nc --exstep monthly --spatial_ts ragis --data_dir $RAGIS_DIR/data  --grid_file $RAGIS_DIR/data/grids/pism-bedmachine-greenland.nc -s electra_skylake -n 160 -q long -w 36:00:00 --o_dir $odir -r $grid -e $RAGIS_DIR/uq/ensemble_gris_ragis_mode.csv $RAGIS_DIR/calibration/2024_11_grimp_tw/state/g1200m_id_BAYES-MEDIAN_1980-1-1_1984-12-31.nc

export PISM_PREFIX=$HOME/local/pism
export RAGIS_DIR=/nobackup/aaschwan/pism-ragis/
export odir=${RAGIS_DIR}/hindcasts/2024_12_mode
export grid=900

python $RAGIS_DIR/hindcasts/hindcast.py --start 1978-01-01 --end 2020-12-31 --boot_file $RAGIS_DIR/data/dem/BedMachineGreenland-v5_0.nc --exstep monthly --spatial_ts ragis --data_dir $RAGIS_DIR/data  --grid_file $RAGIS_DIR/data/grids/pism-bedmachine-greenland.nc -s electra_skylake -n 240 -q long -w 72:00:00 --o_dir $odir -r $grid -e $RAGIS_DIR/uq/ensemble_gris_ragis_mode.csv $RAGIS_DIR/calibration/2024_11_grimp_tw/state/g1200m_id_BAYES-MEDIAN_1980-1-1_1984-12-31.nc




<<<<<<< Updated upstream
=======

>>>>>>> Stashed changes
export RAGIS_DIR=/anvil/projects/x-ees240003/pism-ragis
export PISM_PREFIX=$HOME/local/pism
export grid=1200
export odir=${RAGIS_DIR}/hindcasts/2024_11_dem

python $RAGIS_DIR/hindcasts/hindcast.py --start 1940-01-01 --end 1979-12-31 --boot_file $RAGIS_DIR/data/dem/BedMachineGreenland-v5_0.nc --exstep monthly --spatial_ts dem --data_dir $RAGIS_DIR/data  --grid_file $RAGIS_DIR/data/grids/pism-bedmachine-greenland.nc -s anvil -n 128 -q wholenode -w 48:00:00 --o_dir $odir -r $grid -e $RAGIS_DIR/uq/ensemble_gris_ragis_dem_w_posterior_lhs_10.csv $RAGIS_DIR/calibration/2024_11_grimp_tw/state/g1200m_id_BAYES-MEDIAN_1980-1-1_1984-12-31.nc


export RAGIS_DIR=/anvil/projects/x-ees240003/pism-ragis
export PISM_PREFIX=$HOME/local/pism
export grid=1200
export odir=${RAGIS_DIR}/hindcasts/2024_11_dem_training

python $RAGIS_DIR/hindcasts/hindcast.py --start 1940-01-01 --end 1979-12-31 --boot_file $RAGIS_DIR/data/dem/BedMachineGreenland-v5_0.nc --exstep yearly --spatial_ts dem --data_dir $RAGIS_DIR/data  --grid_file $RAGIS_DIR/data/grids/pism-bedmachine-greenland.nc -s anvil -n 128 -q wholenode -w 24:00:00 --o_dir $odir -r $grid -e $RAGIS_DIR/uq/ensemble_gris_ragis_dem_w_posterior_lhs_10.csv $RAGIS_DIR/calibration/2024_11_grimp_tw/state/g1200m_id_BAYES-MEDIAN_1980-1-1_1984-12-31.nc





for file in hindcasts/2024_07_ragis_tw/spatial/ex_gris_g1800m_v2024-02_RAGIS_id_*; do python analysis/compute_basins_stats_single.py --ensemble_dir hindcasts/ --result_dir analysis/2024_07_ragis_uq --n_jobs 12 $file; done

grid=900
PISM_PREFIX=$HOME/local-intel/pism python hindcast.py --o_size none --exstep monthly --spatial_ts ragis --data_dir /import/c1/ICESHEET/ICESHEET/pism-greenland/data_sets/ --dataset_version 2024-02_RAGIS -s chinook-rl8-24 -n 240 -q t2standard -w 24:00:00 --o_dir 2024_08_ragis_lhs_100 --start 1975-1-1 --end 2021-1-1 -g $grid -e ../uq/ensemble_gris_ragis_w_posterior_lhs_100.csv /import/c1/ICESHEET/ICESHEET/pism-greenland/calibration/2024_02_init_ragis_tw/state/gris_g900m_v2024-02_RAGIS_id_BAYES-MEDIAN_0_20.nc


grid=900
PISM_PREFIX=$HOME/local/pism-dev python hindcast.py --o_size none --exstep monthly --spatial_ts ragis --data_dir /nobackup/aaschwan/pism-greenland/data_sets/ --dataset_version 2024-02_RAGIS -s pleiades_broadwell -n 280 -q debug -w 00:30:00 --o_dir 2024_08_dbg --start 1978-1-1 --end 2021-1-1 -g $grid -e ../uq/ensemble_gris_ragis_w_posterior_lhs_100.csv /nobackup/aaschwan/pism-greenland/calibration/2024_02_init_ragis_tw/state/gris_g900m_v2024-02_RAGIS_id_BAYES-MEDIAN_0_20.nc


grid=900
PISM_PREFIX=$HOME/local/pism-dev python hindcast.py --exstep monthly --spatial_ts ragis --data_dir /nobackup/aaschwan/pism-greenland/data_sets/ --dataset_version 2024-02_RAGIS -s pleiades_broadwell -n 280 -q long -w 48:00:00 --o_dir 2024_08_lhs_100 --start 1978-1-1 --end 2021-1-1 -g $grid -e ../uq/ensemble_gris_ragis_w_posterior_lhs_100.csv /nobackup/aaschwan/pism-greenland/calibration/2024_02_init_ragis_tw/state/gris_g900m_v2024-02_RAGIS_id_BAYES-MEDIAN_0_20.nc

grid=900
PISM_PREFIX=$HOME/local-intel/pism python hindcast.py --exstep monthly --spatial_ts ragis --data_dir /import/c1/ICESHEET/ICESHEET/pism-greenland/data_sets/ --dataset_version 2024-02_RAGIS -s pleiades_broadwell -n 280 -q long -w 48:00:00 --o_dir 2024_08_ragis --start 1978-1-1 --end 2021-1-1 -g $grid -e ../uq/ensemble_gris_ragis_w_posterior_lhs_100.csv /import/c1/ICESHEET/ICESHEET/pism-greenland/calibration/2024_02_init_ragis_tw/state/gris_g900m_v2024-02_RAGIS_id_BAYES-MEDIAN_0_20.nc

grid=1800
PISM_PREFIX=$HOME/local/pism-dev python hindcast.py --exstep monthly --spatial_ts ragis --data_dir /nobackup/aaschwan/pism-greenland/data_sets/ --dataset_version 2024-02_RAGIS -s pleiades_ivy -n 80 -q normal -w 24:00:00 --o_dir 2024_07_ragis_uq --start 1978-1-1 --end 2020-1-1 -g $grid -e ../uq/ensemble_gris_ragis_ctrl.csv /nobackup/aaschwan/pism-greenland/calibration/2024_02_init_ragis_tw/state/gris_g900m_v2024-02_RAGIS_id_BAYES-MEDIAN_0_20.nc


grid=1800
PISM_PREFIX=$HOME/local-intel/pism python hindcast.py --exstep monthly --spatial_ts ragis --data_dir /import/c1/ICESHEET/ICESHEET/pism-greenland/data_sets/ --dataset_version 2024-02_RAGIS -s chinook-rl8-24 -n 72 -q t2standard -w 24:00:00 --o_dir 2024_07_ragis_tw --start 1975-1-1 --end 2021-1-1 -g $grid -e ../uq/ensemble_gris_ragis_w_posterior_lhs_50.csv /import/c1/ICESHEET/ICESHEET/pism-greenland/calibration/2024_02_init_ragis_tw/state/gris_g900m_v2024-02_RAGIS_id_BAYES-MEDIAN_0_20.nc

grid=1800
PISM_PREFIX=$HOME/local-intel/pism python hindcast.py --exstep monthly --spatial_ts ragis --data_dir /import/c1/ICESHEET/ICESHEET/pism-greenland/data_sets/ --dataset_version 2024-02_RAGIS -s chinook-rl8-24 -n 72 -q t2standard -w 24:00:00 --o_dir 2024_07_ragis_lhs_10 --start 1978-1-1 --end 2021-1-1 -g $grid -e ../uq/ensemble_gris_ragis_w_posterior_lhs_10.csv /import/c1/ICESHEET/ICESHEET/pism-greenland/calibration/2024_02_init_ragis_tw/state/gris_g900m_v2024-02_RAGIS_id_BAYES-MEDIAN_0_20.nc


grid=600
PISM_PREFIX=$HOME/local-intel/pism python hindcast.py --o_size none --exstep monthly --spatial_ts ragis --data_dir /import/c1/ICESHEET/ICESHEET/pism-greenland/data_sets/ --dataset_version 2024-02_RAGIS -s chinook-rl8-40 -n 400 -q t2standard -w 96:00:00 --o_dir 2024_08_ragis_lhs_100_600m --start 1975-1-1 --end 2021-1-1 -g $grid -e ../uq/ensemble_gris_ragis_w_posterior_lhs_100.csv /import/c1/ICESHEET/ICESHEET/pism-greenland/calibration/2024_02_init_ragis_tw/state/gris_g600m_v2024-02_RAGIS_id_BAYES-MEDIAN_0_20.nc

grid=900
PISM_PREFIX=$HOME/local-intel/pism python hindcast.py --o_size none --exstep monthly --spatial_ts ragis --data_dir /import/c1/ICESHEET/ICESHEET/pism-greenland/data_sets/ --dataset_version 2024-02_RAGIS -s chinook-rl8-40 -n 240 -q t2standard -w 28:00:00 --o_dir 2024_08_ragis_lhs_100_900m --start 1975-1-1 --end 2021-1-1 -g $grid -e ../uq/ensemble_gris_ragis_w_posterior_lhs_100.csv /import/c1/ICESHEET/ICESHEET/pism-greenland/calibration/2024_02_init_ragis_tw/state/gris_g900m_v2024-02_RAGIS_id_BAYES-MEDIAN_0_20.nc

grid=900
PISM_PREFIX=$HOME/local-intel/pism python hindcast.py --o_size none --exstep monthly --spatial_ts ragis --data_dir /import/c1/ICESHEET/ICESHEET/pism-greenland/data_sets/ --dataset_version 2024-02_RAGIS -s chinook-rl8-24 -n 240 -q t2standard -w 28:00:00 --o_dir 2024_08_ragis_lhs_100_900m --start 1975-1-1 --end 2021-1-1 -g $grid -e ../uq/ensemble_gris_ragis_w_posterior_lhs_100.csv /import/c1/ICESHEET/ICESHEET/pism-greenland/calibration/2024_02_init_ragis_tw/state/gris_g900m_v2024-02_RAGIS_id_BAYES-MEDIAN_0_20.nc


grid=900
PISM_PREFIX=$HOME/local-intel/pism python hindcast.py --o_size none --exstep monthly --spatial_ts ragis --data_dir /import/c1/ICESHEET/ICESHEET/pism-greenland/data_sets/ --dataset_version 2024-02_RAGIS -s chinook-rl8-24 -n 240 -q t2standard -w 28:00:00 --o_dir 2024_08_dbg --start 1975-1-1 --end 2021-1-1 -g $grid -e ../uq/ensemble_gris_ragis_w_posterior_lhs_100.csv /import/c1/ICESHEET/ICESHEET/pism-greenland/calibration/2024_02_init_ragis_tw/state/gris_g900m_v2024-02_RAGIS_id_BAYES-MEDIAN_0_20.nc



grid=900
PISM_PREFIX=$HOME/local-intel/pism python hindcast.py --exstep monthly --spatial_ts ragis --data_dir /import/c1/ICESHEET/ICESHEET/pism-greenland/data_sets/ --dataset_version 2024-02_RAGIS -s chinook-rl8-24 -n 240 -q t2standard -w 24:00:00 --o_dir 2024_07_ragis_900m --start 1978-1-1 --end 2021-1-1 -g $grid -e ../uq/ensemble_gris_ragis_w_posterior_lhs_50.csv /import/c1/ICESHEET/ICESHEET/pism-greenland/calibration/2024_02_init_ragis_tw/state/gris_g900m_v2024-02_RAGIS_id_BAYES-MEDIAN_0_20.nc

grid=900
PISM_PREFIX=$HOME/local-intel/pism python hindcast.py --exstep monthly --spatial_ts ragis --data_dir /import/c1/ICESHEET/ICESHEET/pism-greenland/data_sets/ --dataset_version 2024-02_RAGIS -s chinook-rl8 -n 160 -q t2standard -w 96:00:00 --o_dir 2024_06_ragis_tw --start 1975-1-1 --end 2020-1-1 -g $grid -e ../uq/ensemble_gris_ragis_ctrl.csv /import/c1/ICESHEET/ICESHEET/pism-greenland/calibration/2024_02_init_ragis_tw/state/gris_g900m_v2024-02_RAGIS_id_BAYES-MEDIAN_0_20.nc

grid=600
PISM_PREFIX=$HOME/local-intel/pism python hindcast.py --exstep monthly --spatial_ts ragis --data_dir /import/c1/ICESHEET/ICESHEET/pism-greenland/data_sets/ --dataset_version 2024-02_RAGIS -s chinook-rl8 -n 240 -q t2standard -w 96:00:00 --o_dir 2024_06_ragis_tw --start 1975-1-1 --end 2020-1-1 -g $grid -e ../uq/ensemble_gris_ragis_ctrl.csv /import/c1/ICESHEET/ICESHEET/pism-greenland/calibration/2024_02_init_ragis_tw/state/gris_g600m_v2024-02_RAGIS_id_BAYES-MEDIAN_0_20.nc


grid=900
PISM_PREFIX=$HOME/local/pism-dev python hindcast.py --exstep monthly --spatial_ts ragis --data_dir /nobackup/aaschwan/pism-greenland/data_sets/ --dataset_version 2024-02_RAGIS -s pleiades_ivy -n 160 -q normal -w 8:00:00 --o_dir 2024_07_ragis_ctrl_hy --start 1975-1-1 --end 1980-1-1 -g $grid -e ../uq/ensemble_gris_ragis_ctrl.csv /nobackup/aaschwan/pism-greenland/calibration/2024_02_init_ragis_tw/state/gris_g900m_v2024-02_RAGIS_id_BAYES-MEDIAN_0_20.nc


grid=900
PISM_PREFIX=$HOME/local/pism-dev python hindcast.py --exstep monthly --spatial_ts ragis --data_dir /nobackup/aaschwan/pism-greenland/data_sets/ --dataset_version 2024-02_RAGIS -s pleiades_ivy -n 160 -q normal -w 8:00:00 --o_dir 2024_07_ragis_ctrl_hy --start 1975-1-1 --end 1980-1-1 -g $grid -e ../uq/ensemble_gris_ragis_ctrl.csv /nobackup/aaschwan/pism-greenland/calibration/2024_02_init_ragis_tw/state/gris_g900m_v2024-02_RAGIS_id_BAYES-MEDIAN_0_20.nc


grid=900
PISM_PREFIX=$HOME/local/pism-dev python3 hindcast.py --stress_balance ssa+sia --exstep monthly --spatial_ts ragis --data_dir /work2/08523/aaschwa/frontera/pism-greenland/data_sets/ --dataset_version 2024-02_RAGIS -s frontera -n 224 -q normal -w 48:00:00 --o_dir 2024_07_ragis_ctrl --start 1975-1-1 --end 2020-1-1 -g $grid -e ../uq/ensemble_gris_ragis_ctrl.csv /work2/08523/aaschwa/frontera/pism-greenland/calibration/2024_02_init_ragis_tw/state/gris_g900m_v2024-02_RAGIS_id_BAYES-MEDIAN_0_20.nc

grid=900
PISM_PREFIX=$HOME/local/pism-dev python3 hindcast.py --stress_balance ssa+sia --exstep monthly --spatial_ts ragis --data_dir /work2/08523/aaschwa/frontera/pism-greenland/data_sets/ --dataset_version 2024-02_RAGIS -s frontera -n 224 -q normal -w 48:00:00 --o_dir 2024_07_ragis_notw --start 1975-1-1 --end 2020-1-1 -g $grid -e ../uq/ensemble_gris_ragis_ctrl.csv /work2/08523/aaschwa/frontera/pism-greenland/calibration/2024_02_init_ragis/state/gris_g900m_v2024-02_RAGIS_id_BAYES-MEDIAN_0_20.nc

grid=900
PISM_PREFIX=$HOME/local/pism-dev python3 hindcast.py --stress_balance ssa+sia --exstep monthly --spatial_ts ragis --data_dir /work2/08523/aaschwa/frontera/pism-greenland/data_sets/ --dataset_version 2024-02_RAGIS -s frontera -n 168 -q development -w 2:00:00 --o_dir 2024_07_ragis_ctrl_hybrid --start 1975-1-1 --end 1980-1-1 -g $grid -e ../uq/ensemble_gris_ragis_ctrl.csv /work2/08523/aaschwa/frontera/pism-greenland/calibration/2024_02_init_ragis_tw/state/gris_g900m_v2024-02_RAGIS_id_BAYES-MEDIAN_0_20.nc


grid=900
PISM_PREFIX=$HOME/local/pism-dev python3 hindcast.py --stress_balance blatter --exstep monthly --spatial_ts ragis --data_dir /work2/08523/aaschwa/frontera/pism-greenland/data_sets/ --dataset_version 2024-02_RAGIS -s frontera -n 168 -q development -w 2:00:00 --o_dir 2024_07_ragis_ctrl_blatter --start 1975-1-1 --end 1980-1-1 -g $grid -e ../uq/ensemble_gris_ragis_ctrl.csv /work2/08523/aaschwa/frontera/pism-greenland/calibration/2024_02_init_ragis_tw/state/gris_g900m_v2024-02_RAGIS_id_BAYES-MEDIAN_0_20.nc
