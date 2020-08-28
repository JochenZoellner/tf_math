#!/usr/bin/env bash
# plot statistic over a given dataset, plot distribution of triangle areas and first zero-crossing in the real part of the scattering amplitude
set -xe

DATASET="t2d_fov_exp"
ID_SERIES=$1
shift
PARAMS="
--series_dir \
    models/dataset_${DATASET}/${ID_SERIES}_0 \
    models/dataset_${DATASET}/${ID_SERIES}_1 \
    models/dataset_${DATASET}/${ID_SERIES}_2 \
    models/dataset_${DATASET}/${ID_SERIES}_3 \
    models/dataset_${DATASET}/${ID_SERIES}_4 \
"


CUDA_VISIBLE_DEVICES="" PYTHONPATH=/home/$USER/devel/projects/projectneiss2d/tf_neiss:$PYTHONPATH python -u ./tf_neiss/util/tools/plot_series_json.py ${PARAMS} "$@"

# example call: $> sh ./tf_neiss/scripts/experiments/test_fov/analyze_series_part1.sh min_fov_series
# example call: $> sh ./tf_neiss/scripts/experiments/test_fov/analyze_series_part1.sh max_fov_series --parameter_name max_fov
# example call: $> sh ./tf_neiss/scripts/experiments/test_fov/analyze_series_part1.sh max_fov_series --series_dir models/dataset_t2d_fov_exp/min_fov_seriesB_0 models/dataset_t2d_fov_exp/min_fov_seriesB_1
# example call: $> sh ./tf_neiss/scripts/experiments/test_fov/analyze_series_part1.sh max_fov_series --parameter_name max_fov --series_dir models/dataset_t2d_fov_exp/max_fov_seriesB_0 models/dataset_t2d_fov_exp/max_fov_seriesB_1 models/dataset_t2d_fov_exp/max_fov_seriesB_2 models/dataset_t2d_fov_exp/max_fov_seriesB_3 models/dataset_t2d_fov_exp/max_fov_seriesB_4
# example call: $> sh ./tf_neiss/scripts/experiments/test_fov/analyze_series_part1.sh min_fov_series --series_dir models/dataset_t2d_fov_exp/min_fov_seriesB_0 models/dataset_t2d_fov_exp/min_fov_seriesB_1 models/dataset_t2d_fov_exp/min_fov_seriesB_2 models/dataset_t2d_fov_exp/min_fov_seriesB_3 models/dataset_t2d_fov_exp/min_fov_seriesB_4