#!/usr/bin/env bash
# plot statistic over a given dataset, plot distribution of triangle areas and first zero-crossing in the real part of the scattering amplitude
set -xe

DATASET="t2d_min_fov_exp"
ID_SERIES="min_fov_run_"
PARAMS="
--series_dir \
    models/dataset_${DATASET}/${ID_SERIES}0 \
    models/dataset_${DATASET}/${ID_SERIES}1 \
    models/dataset_${DATASET}/${ID_SERIES}2 \
    models/dataset_${DATASET}/${ID_SERIES}3 \
    models/dataset_${DATASET}/${ID_SERIES}4 \
"


CUDA_VISIBLE_DEVICES="" PYTHONPATH=/home/$USER/devel/projects/projectneiss2d/tf_neiss:$PYTHONPATH python -u ./tf_neiss/util/tools/plot_series_json.py ${PARAMS} "$@"
