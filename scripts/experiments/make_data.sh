#!/usr/bin/env bash
set -xe
DATASET="t2d_min_fov_exp"

PARAMS="\
    --print_to both \
    --mode "val" \
    --data_id t2d_312symetric_unbatchedB \
    --centered False
    "
# make val data
CUDA_VISIBLE_DEVICES="" PYTHONPATH=/home/$USER/devel/projects/projectneiss2d/tf_neiss:$PYTHONPATH python -u ./tf_neiss/input_fn/input_fn_2d/data_gen_2dt/data_generator_t2d.py ${PARAMS} "$@"
# make train data
CUDA_VISIBLE_DEVICES="" PYTHONPATH=/home/$USER/devel/projects/projectneiss2d/tf_neiss:$PYTHONPATH python -u ./tf_neiss/input_fn/input_fn_2d/data_gen_2dt/data_generator_t2d.py "${PARAMS} --mode train $@"
