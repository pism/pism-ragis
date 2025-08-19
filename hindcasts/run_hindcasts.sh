export PISM_PREFIX=$HOME/local/pism
export RAGIS_DIR=/import/c1/ICESHEET/ICESHEET/pism-ragis/
export odir=${RAGIS_DIR}/hindcasts/2025_08_ml
export grid=4800

python $RAGIS_DIR/hindcasts/hindcast.py --start 1900-01-01 --end 2010-01-01 --o_size medium --boot_file $RAGIS_DIR/data/dem/BedMachineGreenland-v5_g450m_0.nc --exstep monthly --spatial_ts ml --data_dir $RAGIS_DIR/data  --grid_file $RAGIS_DIR/data/grids/pism-bedmachine-greenland.nc -s chinook-40 -n 20 -q t2small -w 8:00:00 --o_dir $odir -r $grid -e $RAGIS_DIR/uq/ensemble_gris_ragis_mode.csv $RAGIS_DIR/hindcasts/2025_08_ml_init/state/g1200m_id_MODE-20C-CALV_1900-01-01_1910-01-01.nc

export PISM_PREFIX=$HOME/local/pism
export RAGIS_DIR=/import/c1/ICESHEET/ICESHEET/pism-ragis/
export odir=${RAGIS_DIR}/hindcasts/2025_08_ml
export grid=3600

python $RAGIS_DIR/hindcasts/hindcast.py --start 1900-01-01 --end 2010-01-01 --o_size medium --boot_file $RAGIS_DIR/data/dem/BedMachineGreenland-v5_g450m_0.nc --exstep monthly --spatial_ts ml --data_dir $RAGIS_DIR/data  --grid_file $RAGIS_DIR/data/grids/pism-bedmachine-greenland.nc -s chinook-40 -n 40 -q t2small  -w 8:00:00 --o_dir $odir -r $grid -e $RAGIS_DIR/uq/ensemble_gris_ragis_mode.csv $RAGIS_DIR/hindcasts/2025_08_ml_init/state/g1200m_id_MODE-20C-CALV_1900-01-01_1910-01-01.nc

export PISM_PREFIX=$HOME/local/pism
export RAGIS_DIR=/import/c1/ICESHEET/ICESHEET/pism-ragis/
export odir=${RAGIS_DIR}/hindcasts/2025_08_ml
export grid=2400

python $RAGIS_DIR/hindcasts/hindcast.py --start 1900-01-01 --end 2010-01-01 --o_size medium --boot_file $RAGIS_DIR/data/dem/BedMachineGreenland-v5_g450m_0.nc --exstep monthly --spatial_ts ml --data_dir $RAGIS_DIR/data  --grid_file $RAGIS_DIR/data/grids/pism-bedmachine-greenland.nc -s chinook-40 -n 80 -q t2small -w 8:00:00 --o_dir $odir -r $grid -e $RAGIS_DIR/uq/ensemble_gris_ragis_mode.csv $RAGIS_DIR/hindcasts/2025_08_ml_init/state/g1200m_id_MODE-20C-CALV_1900-01-01_1910-01-01.nc

export PISM_PREFIX=$HOME/local/pism
export RAGIS_DIR=/import/c1/ICESHEET/ICESHEET/pism-ragis/
export odir=${RAGIS_DIR}/hindcasts/2025_08_ml
export grid=1800

python $RAGIS_DIR/hindcasts/hindcast.py --start 1900-01-01 --end 2010-01-01 --o_size medium --boot_file $RAGIS_DIR/data/dem/BedMachineGreenland-v5_g450m_0.nc --exstep monthly --spatial_ts ml --data_dir $RAGIS_DIR/data  --grid_file $RAGIS_DIR/data/grids/pism-bedmachine-greenland.nc -s chinook-40 -n 80 -q t2small -w 20:00:00 --o_dir $odir -r $grid -e $RAGIS_DIR/uq/ensemble_gris_ragis_mode.csv $RAGIS_DIR/hindcasts/2025_08_ml_init/state/g1200m_id_MODE-20C-CALV_1900-01-01_1910-01-01.nc

export PISM_PREFIX=$HOME/local/pism
export RAGIS_DIR=/import/c1/ICESHEET/ICESHEET/pism-ragis/
export odir=${RAGIS_DIR}/hindcasts/2025_08_ml
export grid=1200

python $RAGIS_DIR/hindcasts/hindcast.py --start 1900-01-01 --end 2010-01-01 --o_size medium --boot_file $RAGIS_DIR/data/dem/BedMachineGreenland-v5_g450m_0.nc --exstep monthly --spatial_ts ml --data_dir $RAGIS_DIR/data  --grid_file $RAGIS_DIR/data/grids/pism-bedmachine-greenland.nc -s chinook-40 -n 160 -q t2standard -w 96:00:00 --o_dir $odir -r $grid -e $RAGIS_DIR/uq/ensemble_gris_ragis_mode.csv $RAGIS_DIR/hindcasts/2025_08_ml_init/state/g1200m_id_MODE-20C-CALV_1900-01-01_1910-01-01.nc


export PISM_PREFIX=$HOME/local/pism
export RAGIS_DIR=/import/c1/ICESHEET/ICESHEET/pism-ragis/
export odir=${RAGIS_DIR}/hindcasts/2025_08_ml
export grid=900

python $RAGIS_DIR/hindcasts/hindcast.py --start 1900-01-01 --end 2010-01-01 --o_size medium --boot_file $RAGIS_DIR/data/dem/BedMachineGreenland-v5_g450m_0.nc --exstep monthly --spatial_ts ml --data_dir $RAGIS_DIR/data  --grid_file $RAGIS_DIR/data/grids/pism-bedmachine-greenland.nc -s chinook-40 -n 240 -q t2standard -w 96:00:00 --o_dir $odir -r $grid -e $RAGIS_DIR/uq/ensemble_gris_ragis_mode.csv $RAGIS_DIR/hindcasts/2025_08_ml_init/state/g1200m_id_MODE-20C-CALV_1900-01-01_1910-01-01.nc

export PISM_PREFIX=$HOME/local/pism
export RAGIS_DIR=/import/c1/ICESHEET/ICESHEET/pism-ragis/
export odir=${RAGIS_DIR}/hindcasts/2025_08_ml
export grid=600

python $RAGIS_DIR/hindcasts/hindcast.py --start 1900-01-01 --end 2010-01-01 --o_size medium --boot_file $RAGIS_DIR/data/dem/BedMachineGreenland-v5_g450m_0.nc --exstep monthly --spatial_ts ml --data_dir $RAGIS_DIR/data  --grid_file $RAGIS_DIR/data/grids/pism-bedmachine-greenland.nc -s chinook-40 -n 400 -q t2standard -w 96:00:00 --o_dir $odir -r $grid -e $RAGIS_DIR/uq/ensemble_gris_ragis_mode.csv $RAGIS_DIR/hindcasts/2025_08_ml_init/state/g1200m_id_MODE-20C-CALV_1900-01-01_1910-01-01.nc



export PISM_PREFIX=$HOME/local/pism
export RAGIS_DIR=/import/c1/ICESHEET/ICESHEET/pism-ragis/
export odir=${RAGIS_DIR}/hindcasts/2025_07_ml
export grid=4800

python $RAGIS_DIR/hindcasts/hindcast.py --start 1900-01-01 --end 2010-01-01 --o_size medium --boot_file $RAGIS_DIR/data/dem/BedMachineGreenland-v5_g450m_0.nc --exstep monthly --spatial_ts ml --data_dir $RAGIS_DIR/data  --grid_file $RAGIS_DIR/data/grids/pism-bedmachine-greenland.nc -s chinook-40 -n 28 -q analysis -w 8:00:00 --o_dir $odir -r $grid -e $RAGIS_DIR/uq/ensemble_gris_ragis_mode.csv $RAGIS_DIR/calibration/2024_11_grimp_tw/state/g1200m_id_BAYES-MEDIAN_1980-1-1_1984-12-31.nc

export PISM_PREFIX=$HOME/local/pism
export RAGIS_DIR=/import/c1/ICESHEET/ICESHEET/pism-ragis/
export odir=${RAGIS_DIR}/hindcasts/2025_07_ml
export grid=3600

python $RAGIS_DIR/hindcasts/hindcast.py --start 1900-01-01 --end 2010-01-01 --o_size medium --boot_file $RAGIS_DIR/data/dem/BedMachineGreenland-v5_g450m_0.nc --exstep monthly --spatial_ts ml --data_dir $RAGIS_DIR/data  --grid_file $RAGIS_DIR/data/grids/pism-bedmachine-greenland.nc -s chinook-40 -n 28 -q analysis  -w 8:00:00 --o_dir $odir -r $grid -e $RAGIS_DIR/uq/ensemble_gris_ragis_mode.csv $RAGIS_DIR/calibration/2024_11_grimp_tw/state/g1200m_id_BAYES-MEDIAN_1980-1-1_1984-12-31.nc

export PISM_PREFIX=$HOME/local/pism
export RAGIS_DIR=/import/c1/ICESHEET/ICESHEET/pism-ragis/
export odir=${RAGIS_DIR}/hindcasts/2025_07_ml
export grid=2400

python $RAGIS_DIR/hindcasts/hindcast.py --start 1900-01-01 --end 2010-01-01 --o_size medium --boot_file $RAGIS_DIR/data/dem/BedMachineGreenland-v5_g450m_0.nc --exstep monthly --spatial_ts ml --data_dir $RAGIS_DIR/data  --grid_file $RAGIS_DIR/data/grids/pism-bedmachine-greenland.nc -s chinook-40 -n 80 -q t2small -w 8:00:00 --o_dir $odir -r $grid -e $RAGIS_DIR/uq/ensemble_gris_ragis_mode.csv $RAGIS_DIR/calibration/2024_11_grimp_tw/state/g1200m_id_BAYES-MEDIAN_1980-1-1_1984-12-31.nc

export PISM_PREFIX=$HOME/local/pism
export RAGIS_DIR=/import/c1/ICESHEET/ICESHEET/pism-ragis/
export odir=${RAGIS_DIR}/hindcasts/2025_07_ml
export grid=1800

python $RAGIS_DIR/hindcasts/hindcast.py --start 1900-01-01 --end 2010-01-01 --o_size medium --boot_file $RAGIS_DIR/data/dem/BedMachineGreenland-v5_g450m_0.nc --exstep monthly --spatial_ts ml --data_dir $RAGIS_DIR/data  --grid_file $RAGIS_DIR/data/grids/pism-bedmachine-greenland.nc -s chinook-40 -n 80 -q t2small -w 20:00:00 --o_dir $odir -r $grid -e $RAGIS_DIR/uq/ensemble_gris_ragis_mode.csv $RAGIS_DIR/calibration/2024_11_grimp_tw/state/g1200m_id_BAYES-MEDIAN_1980-1-1_1984-12-31.nc


export PISM_PREFIX=$HOME/local/pism
export RAGIS_DIR=/import/c1/ICESHEET/ICESHEET/pism-ragis/
export odir=${RAGIS_DIR}/hindcasts/2025_07_ml
export grid=1200

python $RAGIS_DIR/hindcasts/hindcast.py --start 1900-01-01 --end 2010-01-01 --o_size medium --boot_file $RAGIS_DIR/data/dem/BedMachineGreenland-v5_g450m_0.nc --exstep monthly --spatial_ts ml --data_dir $RAGIS_DIR/data  --grid_file $RAGIS_DIR/data/grids/pism-bedmachine-greenland.nc -s chinook-40 -n 160 -q t2standard -w 20:00:00 --o_dir $odir -r $grid -e $RAGIS_DIR/uq/ensemble_gris_ragis_mode.csv $RAGIS_DIR/calibration/2024_11_grimp_tw/state/g1200m_id_BAYES-MEDIAN_1980-1-1_1984-12-31.nc

export PISM_PREFIX=$HOME/local/pism
export RAGIS_DIR=/import/c1/ICESHEET/ICESHEET/pism-ragis/
export odir=${RAGIS_DIR}/hindcasts/2025_07_ml
export grid=900

python $RAGIS_DIR/hindcasts/hindcast.py --start 1900-01-01 --end 2010-01-01 --o_size medium --boot_file $RAGIS_DIR/data/dem/BedMachineGreenland-v5_g450m_0.nc --exstep monthly --spatial_ts ml --data_dir $RAGIS_DIR/data  --grid_file $RAGIS_DIR/data/grids/pism-bedmachine-greenland.nc -s chinook-40 -n 240 -q t2standard -w 96:00:00 --o_dir $odir -r $grid -e $RAGIS_DIR/uq/ensemble_gris_ragis_mode.csv $RAGIS_DIR/calibration/2024_11_grimp_tw/state/g1200m_id_BAYES-MEDIAN_1980-1-1_1984-12-31.nc

export PISM_PREFIX=$HOME/local/pism
export RAGIS_DIR=/import/c1/ICESHEET/ICESHEET/pism-ragis/
export odir=${RAGIS_DIR}/hindcasts/2025_07_ml
export grid=600

python $RAGIS_DIR/hindcasts/hindcast.py --start 1900-01-01 --end 2010-01-01 --o_size medium --boot_file $RAGIS_DIR/data/dem/BedMachineGreenland-v5_g450m_0.nc --exstep yearly --spatial_ts ml --data_dir $RAGIS_DIR/data  --grid_file $RAGIS_DIR/data/grids/pism-bedmachine-greenland.nc -s chinook-40 -n 400 -q t2standard -w 96:00:00 --o_dir $odir -r $grid -e $RAGIS_DIR/uq/ensemble_gris_ragis_mode.csv $RAGIS_DIR/calibration/2024_11_grimp_tw/state/g1200m_id_BAYES-MEDIAN_1980-1-1_1984-12-31.nc


PISM_PREFIX=~/local/pism/ python hindcast.py --exstep monthly --spatial_ts ragis --data_dir /import/c1/ICESHEET/ICESHEET/pism-greenland/data_sets/ --dataset_version 2023_RAGIS -s chinook -n 48 -q t2small -w 48:00:00 --o_dir 2023_06_ragis_vcm --start 1980-1-1 --end 2020-1-1  -g 1200 -e ../uq/ensemble_gris_ragis_ctrl.csv /import/c1/ICESHEET/ICESHEET/pism-greenland/calibration/2023_06_init_ragis/state/gris_g1200m_v2023_RAGIS_id_CTRL_0_50.nc

PISM_PREFIX=~/local/pism/ python hindcast.py --exstep monthly --spatial_ts ragis --data_dir /import/c1/ICESHEET/ICESHEET/pism-greenland/data_sets/ --dataset_version 2023_GIMP -s chinook -n 48 -q t2small -w 48:00:00 --o_dir 2023_06_gimp_vcm --start 1980-1-1 --end 2020-1-1  -g 1200 -e ../uq/ensemble_gris_ragis_ctrl.csv /import/c1/ICESHEET/ICESHEET/pism-greenland/calibration/2023_06_init_ragis/state/gris_g1200m_v2023_RAGIS_id_CTRL_0_50.nc

python calibrate-v2022.py --o_dir 2023_06_init_gimp --dataset_version 2023_GIMP --b wc --step 250 --duration 250 -s chinook -q t2small -n 48 -g 1200 -w 72:00:00 --ensemble_file ../uncertainty_quantification/ensemble_gris_ctrl.csv ../../best_v1/g1200m_const_ctrl_e_1.25_ppq_0.6_tefo_0.02_ssa_n_3.25_philow_5.0_hydro_null_100a.nc

python calibrate-v2022.py --o_dir 2023_06_init_ragis --dataset_version 2023_RAGIS --b wc --step 250 --duration 250 -s chinook -q t2small -n 48 -g 1200 -w 72:00:00 --ensemble_file ../uncertainty_quantification/ensemble_gris_ctrl.csv ../../best_v1/g1200m_const_ctrl_e_1.25_ppq_0.6_tefo_0.02_ssa_n_3.25_philow_5.0_hydro_null_100a.nc

PISM_PREFIX=~/local/pism/ python hindcast.py --exstep monthly --spatial_ts ragis --data_dir /import/c1/ICESHEET/ICESHEET/pism-greenland/data_sets/ --dataset_version 2023_RAGIS -s chinook -n 48 -q t2small -w 48:00:00 --o_dir 2023_06_ragis_250 --start 1980-1-1 --end 1986-1-1  -g 1200 -e ../uq/ensemble_gris_ragis_ctrl.csv /import/c1/ICESHEET/ICESHEET/pism-greenland/calibration/2023_06_init_ragis/state/gris_g1200m_v2023_RAGIS_id_CTRL_0_250.nc

PISM_PREFIX=~/local/pism/ python hindcast.py --exstep monthly --spatial_ts ragis --data_dir /import/c1/ICESHEET/ICESHEET/pism-greenland/data_sets/ --dataset_version 2023_RAGIS -s chinook -n 48 -q t2small -w 8:00:00 --o_dir 2023_06_ragis --start 1980-1-1 --end 1986-1-1  -g 1200 -e ../uq/ensemble_gris_ragis_ctrl.csv /import/c1/ICESHEET/ICESHEET/pism-greenland/calibration/2023_06_init_ragis/state/gris_g1200m_v2023_RAGIS_id_CTRL_0_250.nc

PISM_PREFIX=~/local/pism/ python hindcast.py --exstep monthly --spatial_ts ragis --data_dir /import/c1/ICESHEET/ICESHEET/pism-greenland/data_sets/ --dataset_version 2023_GIMP -s chinook -n 48 -q t2small -w 8:00:00 --o_dir 2023_06_gimp --start 1980-1-1 --end 2020-1-1  -g 1200 -e ../uq/ensemble_gris_ragis_ctrl.csv /import/c1/ICESHEET/ICESHEET/pism-greenland/calibration/2023_06_init_gimp/state/gris_g1200m_v2023_GIMP_id_CTRL_0_250.nc

PISM_PREFIX=~/local/pism/ python hindcast.py --exstep monthly --spatial_ts ragis --data_dir /import/c1/ICESHEET/ICESHEET/pism-greenland/data_sets/ --dataset_version 2023_RAGIS -s chinook -n 48 -q t2small -w 8:00:00 --o_dir 2023_06_ragis --start 1980-1-1 --end 2020-1-1  -g 1200 -e ../uq/ensemble_gris_ragis_ctrl.csv /import/c1/ICESHEET/ICESHEET/pism-greenland/calibration/2023_06_init_ragis/state/gris_g1200m_v2023_RAGIS_id_CTRL_0_250.nc

PISM_PREFIX=~/local/pism/ python hindcast.py --exstep monthly --spatial_ts ragis --data_dir /import/c1/ICESHEET/ICESHEET/pism-greenland/data_sets/ --dataset_version 2023_GIMP -s chinook -n 48 -q t2small -w 8:00:00 --o_dir 2023_06_gimp --start 1980-1-1 --end 1986-1-1  -g 1200 -e ../uq/ensemble_gris_ragis_ctrl.csv /import/c1/ICESHEET/ICESHEET/pism-greenland/calibration/2023_06_init_gimp/state/gris_g1200m_v2023_GIMP_id_CTRL_0_250.nc

PISM_PREFIX=~/local/pism/ python hindcast.py --exstep monthly --spatial_ts ragis --data_dir /import/c1/ICESHEET/ICESHEET/pism-greenland/data_sets/ --dataset_version 2023_RAGIS -s chinook -n 192 -q t2standard -w 48:00:00 --o_dir 2023_07_ragis --start 1980-1-1 --end 1986-1-1  -g 600 -e ../uq/ensemble_gris_ragis_ctrl.csv /import/c1/ICESHEET/ICESHEET/pism-greenland/calibration/2023_07_init_gimp/state/gris_g600m_v2023_GIMP_id_CTRL_0_25.nc

PISM_PREFIX=~/local/pism/ python hindcast.py --exstep monthly --spatial_ts ragis --data_dir /import/c1/ICESHEET/ICESHEET/pism-greenland/data_sets/ --dataset_version 2023_GIMP -s chinook -n 192 -q t2standard -w 48:00:00 --o_dir 2023_07_gimp --start 1980-1-1 --end 1986-1-1  -g 600 -e ../uq/ensemble_gris_ragis_ctrl.csv /import/c1/ICESHEET/ICESHEET/pism-greenland/calibration/2023_07_init_ragis/state/gris_g600m_v2023_RAGIS_id_CTRL_0_25.nc

PISM_PREFIX=~/local/pism/ python hindcast.py --exstep monthly --spatial_ts ragis --data_dir /import/c1/ICESHEET/ICESHEET/pism-greenland/data_sets/ --dataset_version 2023_GIMP -s chinook -n 48 -q t2standard -w 48:00:00 --o_dir 2023_07_gimp --start 1980-1-1 --end 1986-1-1 -g 1200 -e ../uq/ensemble_gris_ragis_ctrl.csv /import/c1/ICESHEET/ICESHEET/pism-greenland/calibration/2023_07_init_gimp/state/gris_g1200m_v2023_GIMP_id_CTRL_0_25.nc

PISM_PREFIX=~/local/pism/ python hindcast.py --exstep monthly --spatial_ts ragis --data_dir /import/c1/ICESHEET/ICESHEET/pism-greenland/data_sets/ --dataset_version 2023_RAGIS -s chinook -n 48 -q t2standard -w 48:00:00 --o_dir 2023_07_ragis --start 1980-1-1 --end 1986-1-1 -g 1200 -e ../uq/ensemble_gris_ragis_ctrl.csv /import/c1/ICESHEET/ICESHEET/pism-greenland/calibration/2023_07_init_ragis/state/gris_g1200m_v2023_RAGIS_id_CTRL_0_25.nc

PISM_PREFIX=~/local/pism/ python hindcast.py --exstep monthly --spatial_ts ragis --data_dir /import/c1/ICESHEET/ICESHEET/pism-greenland/data_sets/ --dataset_version 2023_GIMP -s chinook -n 192 -q t2standard -w 48:00:00 --o_dir 2023_07_gimp --start 1980-1-1 --end 1986-1-1  -g 600 -e ../uq/ensemble_gris_ragis_ctrl.csv /import/c1/ICESHEET/ICESHEET/pism-greenland/calibration/2023_07_init_gimp/state/gris_g600m_v2023_GIMP_id_CTRL_0_25.nc

iPISM_PREFIX=~/local/pism/ python hindcast.py --exstep yearly --spatial_ts ragis --data_dir /import/c1/ICESHEET/ICESHEET/pism-greenland/data_sets/ --dataset_version 2023_RAGIS -s chinook -n 192 -q t2standard -w 120:00:00 --o_dir 2023_07_ragis --start 1986-1-1 --end 2020-1-1  -g 600 -e ../uq/ensemble_gris_ragis_ctrl.csv /import/c1/ICESHEET/ICESHEET/pism-ragis/hindcasts/2023_07_ragis/state/gris_g600m_v2023_RAGIS_id_FM-VCM-0.45-200_1980-1-1_1986-1-1.nc

PISM_PREFIX=~/local/pism/ python hindcast.py --exstep yearly --spatial_ts ragis --data_dir /import/c1/ICESHEET/ICESHEET/pism-greenland/data_sets/ --dataset_version 2023_RAGIS -s chinook -n 144 -q t2standard -w 24:00:00 --o_dir 2023_08_ragis_tillwat --start 1980-1-1 --end 2020-1-1  -g 900 -e ../uq/ensemble_gris_ragis_ctrl.csv /import/c1/ICESHEET/ICESHEET/pism-greenland/calibration/2023_07_init_ragis_tillwat/state/gris_g900m_v2023_RAGIS_id_CTRL_0_25.nc

PISM_PREFIX=~/local-rl8/pism/ python hindcast.py --exstep yearly --spatial_ts ragis --data_dir /import/c1/ICESHEET/ICESHEET/pism-greenland/data_sets/ --dataset_version 2023_RAGIS -s chinook-rl8 -n 160 -q t2standard -w 24:00:00 --o_dir 2023_08_ragis_tillwat --start 1980-1-1 --end 2020-1-1  -g 900 -e ../uq/ensemble_gris_ragis_ctrl.csv /import/c1/ICESHEET/ICESHEET/pism-greenland/calibration/2023_07_init_ragis_tillwat/state/gris_g900m_v2023_RAGIS_id_CTRL_0_25.nc


PISM_PREFIX=~/local/pism/ python hindcast.py --exstep yearly --spatial_ts ragis --data_dir /import/c1/ICESHEET/ICESHEET/pism-greenland/data_sets/ --dataset_version 2023_GIMP -s chinook -n 144 -q t2standard -w 100:00:00 --o_dir 2023_08_gimp_tillwat --start 1980-1-1 --end 2020-1-1  -g 900 -e ../uq/ensemble_gris_ragis_ctrl.csv /import/c1/ICESHEET/ICESHEET/pism-greenland/calibration/2023_07_init_gimp_tillwat/state/gris_g900m_v2023_GIMP_id_CTRL_0_25.nc


odir=2023_08_ragis_tillwat
grid=900
mkdir -p ${odir}/processed
for year in 1985 2007; do
for id in FM-0-VCM-0.45-S-0 FM-0-VCM-0.45-S-0 FM-0-VCM-0.50-S-0 FM-1-VCM-0.45-S-0 FM-0-VCM-0.45-S-1.5 FM-1-VCM-0.45-S-1.5 FM-0-VCM-0.45-S-3.0 FM-1-VCM-0.45-S-3.0; do
cdo -L -f nc4 -z zip_2 -O ifthen -timmean -selvar,velsurf_mag -selyear,$year $odir/spatial/ex_gris_g${grid}m_v2023_RAGIS_id_${id}_1980-1-1_2020-1-1.nc -timmean -selvar,velsurf_mag,usurf -selyear,$year $odir/spatial/ex_gris_g${grid}m_v2023_RAGIS_id_${id}_1980-1-1_2020-1-1.nc $odir/processed/velsurf_mag_ex_gris_g${grid}m_v2023_RAGIS_id_${id}_${year}.nc
done
done

odir=2023_08_gimp_tillwat
grid=900
mkdir -p ${odir}/processed
for year in 1985 2007; do
for id in FM-0-VCM-0.45-S-0 FM-0-VCM-0.45-S-0 FM-0-VCM-0.50-S-0 FM-1-VCM-0.45-S-0 FM-0-VCM-0.45-S-1.5 FM-1-VCM-0.45-S-1.5 FM-0-VCM-0.45-S-3.0 FM-1-VCM-0.45-S-3.0; do
cdo -L -f nc4 -z zip_2 -O ifthen -timmean -selvar,velsurf_mag -selyear,$year $odir/spatial/ex_gris_g${grid}m_v2023_GIMP_id_${id}_1980-1-1_2020-1-1.nc -timmean -selvar,velsurf_mag,usurf -selyear,$year $odir/spatial/ex_gris_g${grid}m_v2023_GIMP_id_${id}_1980-1-1_2020-1-1.nc $odir/processed/velsurf_mag_ex_gris_g${grid}m_v2023_GIMP_id_${id}_${year}.nc
done
don

PISM_PREFIX=~/local-rl8/pism/ python hindcast.py --exstep yearly --spatial_ts ragis --data_dir /import/c1/ICESHEET/ICESHEET/pism-greenland/data_sets/ --dataset_version 2023_GIMP -s chinook-rl8 -n 120 -q t2standard -w 72:00:00 --o_dir 2023_08_gimp_tw_50 --start 1980-1-1 --end 2020-1-1 -g 900 -e ../uq/ensemble_gris_ragis_ocean_calving_w_posterior_lhs_50.csv /import/c1/ICESHEET/ICESHEET/pism-greenland/calibration/2023_07_init_gimp_tillwat/state/gris_g900m_v2023_GIMP_id_CTRL_0_25.nc

 PISM_PREFIX=~/local-rl8/pism/ python hindcast.py --exstep yearly --spatial_ts ragis --data_dir /import/c1/ICESHEET/ICESHEET/pism-greenland/data_sets/ --dataset_version 2023_GIMP -s chinook-rl8 -n 120 -q t2standard -w 72:00:00 --o_dir 2023_08_gimp_tw_50 --start 1980-1-1 --end 2020-1-1 -g 900 -e ../uq/ensemble_gris_ragis_ocean_calving_w_posterior_lhs_50.csv /import/c1/ICESHEET/ICESHEET/pism-greenland/calibration/2023_07_init_gimp_tillwat/state/gris_g900m_v2023_GIMP_id_CTRL_0_25.nc

 :e
