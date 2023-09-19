;; This buffer is for text that is not saved, and for Lisp evaluation.
;; To create a file, visit it with C-x C-f and enter text in its buffer.

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
done
