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
