RAGIS_DIR=/import/c1/ICESHEET/ICESHEET/pism-ragis/
resolution=2400
n=40
odir=${RAGIS_DIR}/calibration/2026_02_krig_init

python calibrate.py --end 1984-12-31 --boot_file $RAGIS_DIR/data/dem/boot_g600m_GreenlandObsISMIP7-v1.3.nc --force_to_thickness_file /import/c1/ICESHEET/ICESHEET/pism-ragis/data/dem/boot_g600m_GreenlandObsISMIP7-v1.3.nc --exstep yearly --spatial_ts calibrate --data_dir $RAGIS_DIR/data --grid_file $RAGIS_DIR/data/grids/pism-bedmachine-greenland.nc -s chinook-40 -n $n -q t2small -w 28:00:00 --o_dir $odir -r $resolution -e $RAGIS_DIR/uq/ensemble_gris_calibrate.csv $RAGIS_DIR/calibration/2017_06_vc/state/gris_g${resolution}m_flux_v3a_no_bath_sia_e_1.25_sia_n_3_ssa_n_3.25_ppq_0.6_tefo_0.02_calving_vonmises_calving_0_100.nc

python adjust_tillwat.py --speed_file ../data/itslive/GRE_G0240_0000.nc ${odir}/state/g${resolution}m_id_BAYES-MEDIAN_1980-1-1_1984-12-31.nc ${odir}/state/g${resolution}m_id_BAYES-MEDIAN_1980-1-1_1984-12-31_tw.nc

idir=$odir
odir=2026_02_krig_tw

python calibrate.py --end 1984-12-31 --boot_file $RAGIS_DIR/data/dem/boot_g600m_GreenlandObsISMIP7-v1.3.nc --force_to_thickness_file /import/c1/ICESHEET/ICESHEET/pism-ragis/data/dem/boot_g600m_GreenlandObsISMIP7-v1.3.nc --exstep yearly --spatial_ts calibrate --data_dir $RAGIS_DIR/data --grid_file $RAGIS_DIR/data/grids/pism-bedmachine-greenland.nc -s chinook-40 -n $n -q t2small -w 28:00:00 --o_dir $odir -r $resolution -e $RAGIS_DIR/uq/ensemble_gris_calibrate.csv ${idir}/state/g${resolution}m_id_BAYES-MEDIAN_1980-1-1_1984-12-31_tw.nc

RAGIS_DIR=/import/c1/ICESHEET/ICESHEET/pism-ragis/
resolution=1200
n=80
odir=${RAGIS_DIR}/calibration/2026_02_krig_init

python calibrate.py --end 1984-12-31 --boot_file $RAGIS_DIR/data/dem/boot_g600m_GreenlandObsISMIP7-v1.3.nc --force_to_thickness_file /import/c1/ICESHEET/ICESHEET/pism-ragis/data/dem/boot_g600m_GreenlandObsISMIP7-v1.3.nc --exstep yearly --spatial_ts calibrate --data_dir $RAGIS_DIR/data --grid_file $RAGIS_DIR/data/grids/pism-bedmachine-greenland.nc -s chinook-40 -n $n -q t2small -w 28:00:00 --o_dir $odir -r $resolution -e $RAGIS_DIR/uq/ensemble_gris_calibrate.csv $RAGIS_DIR/calibration/2017_06_vc/state/gris_g${resolution}m_flux_v3a_no_bath_sia_e_1.25_sia_n_3_ssa_n_3.25_ppq_0.6_tefo_0.02_calving_vonmises_calving_0_100.nc

python adjust_tillwat.py --speed_file ../data/itslive/GRE_G0240_0000.nc ${odir}/state/g${resolution}m_id_BAYES-MEDIAN_1980-1-1_1984-12-31.nc ${odir}/state/g${resolution}m_id_BAYES-MEDIAN_1980-1-1_1984-12-31_tw.nc

idir=$odir
odir=2026_02_krig_tw

python calibrate.py --end 1984-12-31 --boot_file $RAGIS_DIR/data/dem/boot_g600m_GreenlandObsISMIP7-v1.3.nc --force_to_thickness_file /import/c1/ICESHEET/ICESHEET/pism-ragis/data/dem/boot_g600m_GreenlandObsISMIP7-v1.3.nc --exstep yearly --spatial_ts calibrate --data_dir $RAGIS_DIR/data --grid_file $RAGIS_DIR/data/grids/pism-bedmachine-greenland.nc -s chinook-40 -n $n -q t2small -w 28:00:00 --o_dir $odir -r $resolution -e $RAGIS_DIR/uq/ensemble_gris_calibrate.csv ${idir}/state/g${resolution}m_id_BAYES-MEDIAN_1980-1-1_1984-12-31_tw.nc


RAGIS_DIR=/import/c1/ICESHEET/ICESHEET/pism-ragis/
resolution=600
n=360
odir=${RAGIS_DIR}/calibration/2026_02_krig_init

python calibrate.py --end 1984-12-31 --boot_file $RAGIS_DIR/data/dem/boot_g600m_GreenlandObsISMIP7-v1.3.nc --force_to_thickness_file /import/c1/ICESHEET/ICESHEET/pism-ragis/data/dem/boot_g600m_GreenlandObsISMIP7-v1.3.nc --exstep yearly --spatial_ts calibrate --data_dir $RAGIS_DIR/data --grid_file $RAGIS_DIR/data/grids/pism-bedmachine-greenland.nc -s chinook-40 -n $n -q t2standard -w 18:00:00 --o_dir $odir -r $resolution -e $RAGIS_DIR/uq/ensemble_gris_calibrate.csv $RAGIS_DIR/calibration/2017_06_vc/state/gris_g${resolution}m_flux_v3a_no_bath_sia_e_1.25_sia_n_3_ssa_n_3.25_ppq_0.6_tefo_0.02_calving_vonmises_calving_0_100.nc

python adjust_tillwat.py --speed_file ../data/itslive/GRE_G0240_0000.nc ${odir}/state/g${resolution}m_id_BAYES-MEDIAN_1980-1-1_1984-12-31.nc ${odir}/state/g${resolution}m_id_BAYES-MEDIAN_1980-1-1_1984-12-31_tw.nc

idir=$odir
odir=2026_02_krig_tw

python calibrate.py --end 1984-12-31 --boot_file $RAGIS_DIR/data/dem/boot_g600m_GreenlandObsISMIP7-v1.3.nc --force_to_thickness_file /import/c1/ICESHEET/ICESHEET/pism-ragis/data/dem/boot_g600m_GreenlandObsISMIP7-v1.3.nc --exstep yearly --spatial_ts calibrate --data_dir $RAGIS_DIR/data --grid_file $RAGIS_DIR/data/grids/pism-bedmachine-greenland.nc -s chinook-40 -n $n -q t2standard -w 18:00:00 --o_dir $odir -r $resolution -e $RAGIS_DIR/uq/ensemble_gris_calibrate.csv ${idir}/state/g${resolution}m_id_BAYES-MEDIAN_1980-1-1_1984-12-31_tw.nc





python /import/c1/ICESHEET/ICESHEET/pism-ragis//calibration/calibrate.py --no-regrid-thickness --end 1984-12-31 --boot_file /import/c1/ICESHEET/ICESHEET/pism-ragis//data/dem/BedMachineGreenland-v5_g450m_0.nc --force_to_thickness_file /import/c1/ICESHEET/ICESHEET/pism-ragis//data/dem/BedMachineGreenland-v5_0.nc --exstep yearly --spatial_ts calibrate --data_dir /import/c1/ICESHEET/ICESHEET/pism-ragis//data --grid_file /import/c1/ICESHEET/ICESHEET/pism-ragis//data/grids/pism-bedmachine-greenland.nc -s chinook-40 -n 360 -q t2standard -w 36:00:00 --o_dir /import/c1/ICESHEET/ICESHEET/pism-ragis//calibration/2026_01_grimp_tw -r 600 -e /import/c1/ICESHEET/ICESHEET/pism-ragis//uq/ensemble_gris_calibrate.csv /import/c1/ICESHEET/ICESHEET/pism-ragis//calibration/2025_08_grimp_init/state/g600m_id_BAYES-MEDIAN_1980-1-1_1984-12-31_tw.nc

python /import/c1/ICESHEET/ICESHEET/pism-ragis//calibration/calibrate.py --no-regrid-thickness --end 1984-12-31 --boot_file /import/c1/ICESHEET/ICESHEET/pism-ragis//data/dem/bedmachine1980_reconstructed_g600_kriging_0.nc --force_to_thickness_file /import/c1/ICESHEET/ICESHEET/pism-ragis//data/dem/bedmachine1980_reconstructed_g600_kriging_0.nc --exstep yearly --spatial_ts calibrate --data_dir /import/c1/ICESHEET/ICESHEET/pism-ragis//data --grid_file /import/c1/ICESHEET/ICESHEET/pism-ragis//data/grids/pism-bedmachine-greenland.nc -s chinook-40 -n 360 -q t2standard -w 36:00:00 --o_dir /import/c1/ICESHEET/ICESHEET/pism-ragis//calibration/2026_01_krig_tw -r 600 -e /import/c1/ICESHEET/ICESHEET/pism-ragis//uq/ensemble_gris_calibrate.csv /import/c1/ICESHEET/ICESHEET/pism-ragis//calibration/2025_08_grimp_init/state/g600m_id_BAYES-MEDIAN_1980-1-1_1984-12-31_tw.nc

python /import/c1/ICESHEET/ICESHEET/pism-ragis//calibration/calibrate.py --no-regrid-thickness --end 1984-12-31 --boot_file /import/c1/ICESHEET/ICESHEET/pism-ragis//data/dem/bedmachine1980_reconstructed_g600_SVD_0.nc --force_to_thickness_file /import/c1/ICESHEET/ICESHEET/pism-ragis//data/dem/bedmachine1980_reconstructed_g600_SVD_0.nc --exstep yearly --spatial_ts calibrate --data_dir /import/c1/ICESHEET/ICESHEET/pism-ragis//data --grid_file /import/c1/ICESHEET/ICESHEET/pism-ragis//data/grids/pism-bedmachine-greenland.nc -s chinook-40 -n 360 -q t2standard -w 36:00:00 --o_dir /import/c1/ICESHEET/ICESHEET/pism-ragis//calibration/2026_01_svd_tw -r 600 -e /import/c1/ICESHEET/ICESHEET/pism-ragis//uq/ensemble_gris_calibrate.csv /import/c1/ICESHEET/ICESHEET/pism-ragis//calibration/2025_08_grimp_init/state/g600m_id_BAYES-MEDIAN_1980-1-1_1984-12-31_tw.nc



RAGIS_DIR=/import/c1/ICESHEET/ICESHEET/pism-ragis/
resolution=4500
n=14
odir=${RAGIS_DIR}/calibration/2025_08_grimp_init

python calibrate.py --end 1984-12-31 --boot_file $RAGIS_DIR/data/dem/BedMachineGreenland-v5_g450m_0.nc --force_to_thickness_file /import/c1/ICESHEET/ICESHEET/pism-ragis/data/dem/BedMachineGreenland-v5_0.nc --exstep yearly --spatial_ts calibrate --data_dir $RAGIS_DIR/data --grid_file $RAGIS_DIR/data/grids/pism-bedmachine-greenland.nc -s chinook-40 -n $n -q analysis -w 8:00:00 --o_dir $odir -r $resolution -e $RAGIS_DIR/uq/ensemble_gris_calibrate.csv $RAGIS_DIR/calibration/2017_06_vc/state/gris_g${resolution}m_flux_v3a_no_bath_sia_e_1.25_sia_n_3_ssa_n_3.25_ppq_0.6_tefo_0.02_calving_vonmises_calving_0_100.nc

python adjust_tillwat.py --speed_file ../data/itslive/GRE_G0240_0000.nc 2025_08_grimp_init/state/g${resolution}m_id_BAYES-MEDIAN_1980-1-1_1984-12-31.nc 2025_08_grimp_init/state/g${resolution}m_id_BAYES-MEDIAN_1980-1-1_1984-12-31_tw.nc

RAGIS_DIR=/import/c1/ICESHEET/ICESHEET/pism-ragis/
resolution=3600
n=28
odir=${RAGIS_DIR}/calibration/2025_08_grimp_init

python calibrate.py --end 1984-12-31 --boot_file $RAGIS_DIR/data/dem/BedMachineGreenland-v5_g450m_0.nc --force_to_thickness_file /import/c1/ICESHEET/ICESHEET/pism-ragis/data/dem/BedMachineGreenland-v5_0.nc --exstep yearly --spatial_ts calibrate --data_dir $RAGIS_DIR/data --grid_file $RAGIS_DIR/data/grids/pism-bedmachine-greenland.nc -s chinook-40 -n $n -q analysis -w 8:00:00 --o_dir $odir -r $resolution -e $RAGIS_DIR/uq/ensemble_gris_calibrate.csv $RAGIS_DIR/calibration/2017_06_vc/state/gris_g${resolution}m_flux_v3a_no_bath_sia_e_1.25_sia_n_3_ssa_n_3.25_ppq_0.6_tefo_0.02_calving_vonmises_calving_0_100.nc

python adjust_tillwat.py --speed_file ../data/itslive/GRE_G0240_0000.nc 2025_08_grimp_init/state/g${resolution}m_id_BAYES-MEDIAN_1980-1-1_1984-12-31.nc 2025_08_grimp_init/state/g${resolution}m_id_BAYES-MEDIAN_1980-1-1_1984-12-31_tw.nc

RAGIS_DIR=/import/c1/ICESHEET/ICESHEET/pism-ragis/
resolution=2400
n=40
odir=${RAGIS_DIR}/calibration/2025_08_grimp_init

python calibrate.py --end 1984-12-31 --boot_file $RAGIS_DIR/data/dem/BedMachineGreenland-v5_g450m_0.nc --force_to_thickness_file /import/c1/ICESHEET/ICESHEET/pism-ragis/data/dem/BedMachineGreenland-v5_0.nc --exstep yearly --spatial_ts calibrate --data_dir $RAGIS_DIR/data --grid_file $RAGIS_DIR/data/grids/pism-bedmachine-greenland.nc -s chinook-40 -n $n -q t2small -w 8:00:00 --o_dir $odir -r $resolution -e $RAGIS_DIR/uq/ensemble_gris_calibrate.csv $RAGIS_DIR/calibration/2017_06_vc/state/gris_g${resolution}m_flux_v3a_no_bath_sia_e_1.25_sia_n_3_ssa_n_3.25_ppq_0.6_tefo_0.02_calving_vonmises_calving_0_100.nc

python adjust_tillwat.py --speed_file ../data/itslive/GRE_G0240_0000.nc 2025_08_grimp_init/state/g${resolution}m_id_BAYES-MEDIAN_1980-1-1_1984-12-31.nc 2025_08_grimp_init/state/g${resolution}m_id_BAYES-MEDIAN_1980-1-1_1984-12-31_tw.nc

RAGIS_DIR=/import/c1/ICESHEET/ICESHEET/pism-ragis/
resolution=1800
n=80
odir=${RAGIS_DIR}/calibration/2025_08_grimp_init

python calibrate.py --end 1984-12-31 --boot_file $RAGIS_DIR/data/dem/BedMachineGreenland-v5_g450m_0.nc --force_to_thickness_file /import/c1/ICESHEET/ICESHEET/pism-ragis/data/dem/BedMachineGreenland-v5_0.nc --exstep yearly --spatial_ts calibrate --data_dir $RAGIS_DIR/data --grid_file $RAGIS_DIR/data/grids/pism-bedmachine-greenland.nc -s chinook-40 -n $n -q t2small -w 28:00:00 --o_dir $odir -r $resolution -e $RAGIS_DIR/uq/ensemble_gris_calibrate.csv $RAGIS_DIR/calibration/2017_06_vc/state/gris_g${resolution}m_flux_v3a_no_bath_sia_e_1.25_sia_n_3_ssa_n_3.25_ppq_0.6_tefo_0.02_calving_vonmises_calving_0_100.nc

python adjust_tillwat.py --speed_file ../data/itslive/GRE_G0240_0000.nc 2025_08_grimp_init/state/g${resolution}m_id_BAYES-MEDIAN_1980-1-1_1984-12-31.nc 2025_08_grimp_init/state/g${resolution}m_id_BAYES-MEDIAN_1980-1-1_1984-12-31_tw.nc

RAGIS_DIR=/import/c1/ICESHEET/ICESHEET/pism-ragis/
resolution=1500
n=80
odir=${RAGIS_DIR}/calibration/2025_08_grimp_init

python calibrate.py --end 1984-12-31 --boot_file $RAGIS_DIR/data/dem/BedMachineGreenland-v5_g450m_0.nc --force_to_thickness_file /import/c1/ICESHEET/ICESHEET/pism-ragis/data/dem/BedMachineGreenland-v5_0.nc --exstep yearly --spatial_ts calibrate --data_dir $RAGIS_DIR/data --grid_file $RAGIS_DIR/data/grids/pism-bedmachine-greenland.nc -s chinook-40 -n $n -q t2small -w 28:00:00 --o_dir $odir -r $resolution -e $RAGIS_DIR/uq/ensemble_gris_calibrate.csv $RAGIS_DIR/calibration/2017_06_vc/state/gris_g${resolution}m_flux_v3a_no_bath_sia_e_1.25_sia_n_3_ssa_n_3.25_ppq_0.6_tefo_0.02_calving_vonmises_calving_0_100.nc

python adjust_tillwat.py --speed_file ../data/itslive/GRE_G0240_0000.nc 2025_08_grimp_init/state/g${resolution}m_id_BAYES-MEDIAN_1980-1-1_1984-12-31.nc 2025_08_grimp_init/state/g${resolution}m_id_BAYES-MEDIAN_1980-1-1_1984-12-31_tw.nc

RAGIS_DIR=/import/c1/ICESHEET/ICESHEET/pism-ragis/
resolution=1200
n=80
odir=${RAGIS_DIR}/calibration/2025_08_grimp_init

python calibrate.py --end 1984-12-31 --boot_file $RAGIS_DIR/data/dem/BedMachineGreenland-v5_g450m_0.nc --force_to_thickness_file /import/c1/ICESHEET/ICESHEET/pism-ragis/data/dem/BedMachineGreenland-v5_0.nc --exstep yearly --spatial_ts calibrate --data_dir $RAGIS_DIR/data --grid_file $RAGIS_DIR/data/grids/pism-bedmachine-greenland.nc -s chinook-40 -n $n -q t2small -w 28:00:00 --o_dir $odir -r $resolution -e $RAGIS_DIR/uq/ensemble_gris_calibrate.csv $RAGIS_DIR/calibration/2017_06_vc/state/gris_g${resolution}m_flux_v3a_no_bath_sia_e_1.25_sia_n_3_ssa_n_3.25_ppq_0.6_tefo_0.02_calving_vonmises_calving_0_100.nc

python adjust_tillwat.py --speed_file ../data/itslive/GRE_G0240_0000.nc 2025_08_grimp_init/state/g${resolution}m_id_BAYES-MEDIAN_1980-1-1_1984-12-31.nc 2025_08_grimp_init/state/g${resolution}m_id_BAYES-MEDIAN_1980-1-1_1984-12-31_tw.nc

RAGIS_DIR=/import/c1/ICESHEET/ICESHEET/pism-ragis/
resolution=900
n=120
odir=${RAGIS_DIR}/calibration/2025_08_grimp_init

python adjust_tillwat.py --speed_file ../data/itslive/GRE_G0240_0000.nc 2025_08_grimp_init/state/g${resolution}m_id_BAYES-MEDIAN_1980-1-1_1984-12-31.nc 2025_08_grimp_init/state/g${resolution}m_id_BAYES-MEDIAN_1980-1-1_1984-12-31_tw.nc

python calibrate.py --end 1984-12-31 --boot_file $RAGIS_DIR/data/dem/BedMachineGreenland-v5_g450m_0.nc --force_to_thickness_file /import/c1/ICESHEET/ICESHEET/pism-ragis/data/dem/BedMachineGreenland-v5_0.nc --exstep yearly --spatial_ts calibrate --data_dir $RAGIS_DIR/data --grid_file $RAGIS_DIR/data/grids/pism-bedmachine-greenland.nc -s chinook-40 -n $n -q t2standard -w 28:00:00 --o_dir $odir -r $resolution -e $RAGIS_DIR/uq/ensemble_gris_calibrate.csv $RAGIS_DIR/calibration/2017_06_vc/state/gris_g${resolution}m_flux_v3a_no_bath_sia_e_1.25_sia_n_3_ssa_n_3.25_ppq_0.6_tefo_0.02_calving_vonmises_calving_0_100.nc

python adjust_tillwat.py --speed_file ../data/itslive/GRE_G0240_0000.nc 2025_08_grimp_init/state/g${resolution}m_id_BAYES-MEDIAN_1980-1-1_1984-12-31.nc 2025_08_grimp_init/state/g${resolution}m_id_BAYES-MEDIAN_1980-1-1_1984-12-31_tw.nc

RAGIS_DIR=/import/c1/ICESHEET/ICESHEET/pism-ragis/
resolution=600
n=200
odir=${RAGIS_DIR}/calibration/2025_08_grimp_init

python calibrate.py --end 1984-12-31 --boot_file $RAGIS_DIR/data/dem/BedMachineGreenland-v5_g450m_0.nc --force_to_thickness_file /import/c1/ICESHEET/ICESHEET/pism-ragis/data/dem/BedMachineGreenland-v5_0.nc --exstep yearly --spatial_ts calibrate --data_dir $RAGIS_DIR/data --grid_file $RAGIS_DIR/data/grids/pism-bedmachine-greenland.nc -s chinook-40 -n $n -q t2standard -w 28:00:00 --o_dir $odir -r $resolution -e $RAGIS_DIR/uq/ensemble_gris_calibrate.csv $RAGIS_DIR/calibration/2017_06_vc/state/gris_g${resolution}m_flux_v3a_no_bath_sia_e_1.25_sia_n_3_ssa_n_3.25_ppq_0.6_tefo_0.02_calving_vonmises_calving_0_100.nc

python adjust_tillwat.py --speed_file ../data/itslive/GRE_G0240_0000.nc 2025_08_grimp_init/state/g${resolution}m_id_BAYES-MEDIAN_1980-1-1_1984-12-31.nc 2025_08_grimp_init/state/g${resolution}m_id_BAYES-MEDIAN_1980-1-1_1984-12-31_tw.nc


RAGIS_DIR=/import/c1/ICESHEET/ICESHEET/pism-ragis/
resolution=4500
n=14
odir=${RAGIS_DIR}/calibration/2025_08_grimp_tw

python calibrate.py --end 1984-12-31 --boot_file $RAGIS_DIR/data/dem/BedMachineGreenland-v5_g450m_0.nc --force_to_thickness_file /import/c1/ICESHEET/ICESHEET/pism-ragis/data/dem/BedMachineGreenland-v5_0.nc --exstep yearly --spatial_ts calibrate --data_dir $RAGIS_DIR/data --grid_file $RAGIS_DIR/data/grids/pism-bedmachine-greenland.nc -s chinook-40 -n $n -q analysis -w 8:00:00 --o_dir $odir -r $resolution -e $RAGIS_DIR/uq/ensemble_gris_calibrate.csv $RAGIS_DIR/calibration/2025_08_grimp_init/state/g${resolution}m_id_BAYES-MEDIAN_1980-1-1_1984-12-31_tw.nc


RAGIS_DIR=/import/c1/ICESHEET/ICESHEET/pism-ragis/
resolution=3600
n=28
odir=${RAGIS_DIR}/calibration/2025_08_grimp_tw

python calibrate.py --end 1984-12-31 --boot_file $RAGIS_DIR/data/dem/BedMachineGreenland-v5_g450m_0.nc --force_to_thickness_file /import/c1/ICESHEET/ICESHEET/pism-ragis/data/dem/BedMachineGreenland-v5_0.nc --exstep yearly --spatial_ts calibrate --data_dir $RAGIS_DIR/data --grid_file $RAGIS_DIR/data/grids/pism-bedmachine-greenland.nc -s chinook-40 -n $n -q analysis -w 8:00:00 --o_dir $odir -r $resolution -e $RAGIS_DIR/uq/ensemble_gris_calibrate.csv $RAGIS_DIR/calibration/2025_08_grimp_init/state/g${resolution}m_id_BAYES-MEDIAN_1980-1-1_1984-12-31_tw.nc


RAGIS_DIR=/import/c1/ICESHEET/ICESHEET/pism-ragis/
resolution=2400
n=40
odir=${RAGIS_DIR}/calibration/2025_08_grimp_tw

python calibrate.py --end 1984-12-31 --boot_file $RAGIS_DIR/data/dem/BedMachineGreenland-v5_g450m_0.nc --force_to_thickness_file /import/c1/ICESHEET/ICESHEET/pism-ragis/data/dem/BedMachineGreenland-v5_0.nc --exstep yearly --spatial_ts calibrate --data_dir $RAGIS_DIR/data --grid_file $RAGIS_DIR/data/grids/pism-bedmachine-greenland.nc -s chinook-40 -n $n -q t2small -w 8:00:00 --o_dir $odir -r $resolution -e $RAGIS_DIR/uq/ensemble_gris_calibrate.csv $RAGIS_DIR/calibration/2025_08_grimp_init/state/g${resolution}m_id_BAYES-MEDIAN_1980-1-1_1984-12-31_tw.nc


RAGIS_DIR=/import/c1/ICESHEET/ICESHEET/pism-ragis/
resolution=1800
n=80
odir=${RAGIS_DIR}/calibration/2025_08_grimp_tw

python calibrate.py --end 1984-12-31 --boot_file $RAGIS_DIR/data/dem/BedMachineGreenland-v5_g450m_0.nc --force_to_thickness_file /import/c1/ICESHEET/ICESHEET/pism-ragis/data/dem/BedMachineGreenland-v5_0.nc --exstep yearly --spatial_ts calibrate --data_dir $RAGIS_DIR/data --grid_file $RAGIS_DIR/data/grids/pism-bedmachine-greenland.nc -s chinook-40 -n $n -q t2small -w 28:00:00 --o_dir $odir -r $resolution -e $RAGIS_DIR/uq/ensemble_gris_calibrate.csv $RAGIS_DIR/calibration/2025_08_grimp_init/state/g${resolution}m_id_BAYES-MEDIAN_1980-1-1_1984-12-31_tw.nc


RAGIS_DIR=/import/c1/ICESHEET/ICESHEET/pism-ragis/
resolution=1500
n=80
odir=${RAGIS_DIR}/calibration/2025_08_grimp_tw

python calibrate.py --end 1984-12-31 --boot_file $RAGIS_DIR/data/dem/BedMachineGreenland-v5_g450m_0.nc --force_to_thickness_file /import/c1/ICESHEET/ICESHEET/pism-ragis/data/dem/BedMachineGreenland-v5_0.nc --exstep yearly --spatial_ts calibrate --data_dir $RAGIS_DIR/data --grid_file $RAGIS_DIR/data/grids/pism-bedmachine-greenland.nc -s chinook-40 -n $n -q t2small -w 28:00:00 --o_dir $odir -r $resolution -e $RAGIS_DIR/uq/ensemble_gris_calibrate.csv $RAGIS_DIR/calibration/2025_08_grimp_init/state/g${resolution}m_id_BAYES-MEDIAN_1980-1-1_1984-12-31_tw.nc


RAGIS_DIR=/import/c1/ICESHEET/ICESHEET/pism-ragis/
resolution=1200
n=80
odir=${RAGIS_DIR}/calibration/2025_08_grimp_tw

python calibrate.py --end 1984-12-31 --boot_file $RAGIS_DIR/data/dem/BedMachineGreenland-v5_g450m_0.nc --force_to_thickness_file /import/c1/ICESHEET/ICESHEET/pism-ragis/data/dem/BedMachineGreenland-v5_0.nc --exstep yearly --spatial_ts calibrate --data_dir $RAGIS_DIR/data --grid_file $RAGIS_DIR/data/grids/pism-bedmachine-greenland.nc -s chinook-40 -n $n -q t2small -w 28:00:00 --o_dir $odir -r $resolution -e $RAGIS_DIR/uq/ensemble_gris_calibrate.csv $RAGIS_DIR/calibration/2025_08_grimp_init/state/g${resolution}m_id_BAYES-MEDIAN_1980-1-1_1984-12-31_tw.nc


RAGIS_DIR=/import/c1/ICESHEET/ICESHEET/pism-ragis/
resolution=900
n=120
odir=${RAGIS_DIR}/calibration/2025_08_grimp_tw


python calibrate.py --end 1984-12-31 --boot_file $RAGIS_DIR/data/dem/BedMachineGreenland-v5_g450m_0.nc --force_to_thickness_file /import/c1/ICESHEET/ICESHEET/pism-ragis/data/dem/BedMachineGreenland-v5_0.nc --exstep yearly --spatial_ts calibrate --data_dir $RAGIS_DIR/data --grid_file $RAGIS_DIR/data/grids/pism-bedmachine-greenland.nc -s chinook-40 -n $n -q t2standard -w 28:00:00 --o_dir $odir -r $resolution -e $RAGIS_DIR/uq/ensemble_gris_calibrate.csv $RAGIS_DIR/calibration/2025_08_grimp_init/state/g${resolution}m_id_BAYES-MEDIAN_1980-1-1_1984-12-31_tw.nc


RAGIS_DIR=/import/c1/ICESHEET/ICESHEET/pism-ragis/
resolution=600
n=200
odir=${RAGIS_DIR}/calibration/2025_08_grimp_tw

python calibrate.py --end 1984-12-31 --boot_file $RAGIS_DIR/data/dem/BedMachineGreenland-v5_g450m_0.nc --force_to_thickness_file /import/c1/ICESHEET/ICESHEET/pism-ragis/data/dem/BedMachineGreenland-v5_0.nc --exstep yearly --spatial_ts calibrate --data_dir $RAGIS_DIR/data --grid_file $RAGIS_DIR/data/grids/pism-bedmachine-greenland.nc -s chinook-40 -n $n -q t2standard -w 28:00:00 --o_dir $odir -r $resolution -e $RAGIS_DIR/uq/ensemble_gris_calibrate.csv $RAGIS_DIR/calibration/2025_08_grimp_init/state/g${resolution}m_id_BAYES-MEDIAN_1980-1-1_1984-12-31_tw.nc


