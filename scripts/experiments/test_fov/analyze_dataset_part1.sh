#!/usr/bin/env bash
# plot statistic over a given dataset, plot distribution of triangle areas and first zero-crossing in the real part of the scattering amplitude
set -xe

DATASET="t2d_fov_exp"

PARAMS="
--val_list lists/${DATASET}_val.lst \
--samples 10000 \
--plot_prefix val_ \
"

CUDA_VISIBLE_DEVICES="" PYTHONPATH=/home/$USER/devel/projects/projectneiss2d/tf_neiss:$PYTHONPATH python -u ./tf_neiss/util/tools/dataset_stats.py ${PARAMS} "$@"

PARAMS="
--val_list lists/${DATASET}_train.lst \
--samples 1000000 \
--plot_prefix train_ \
"

CUDA_VISIBLE_DEVICES="" PYTHONPATH=/home/$USER/devel/projects/projectneiss2d/tf_neiss:$PYTHONPATH python -u ./tf_neiss/util/tools/dataset_stats.py ${PARAMS} "$@"

# example call: $> sh ./tf_neiss/scripts/experiments/test_fov/analyze_dataset_part1.sh