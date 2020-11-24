#!/usr/bin/env bash
set -xe
DATASET="t2d_find_ambiguous_FC"

PARAMS="\
    --print_to both \
    --mode "val" \
    --data_id ${DATASET} \
    --centered False
    --phi_range 2pi \
    "
# make val data
CUDA_VISIBLE_DEVICES="" TS_SOCKET="cpu" PYTHONPATH=/home/$USER/devel/projects/projectneiss2d/tf_neiss:$PYTHONPATH tsp python -u ./tf_neiss/input_fn/input_fn_2d/data_gen_2dt/data_generator_t2d.py ${PARAMS} "$@"
# make train data
PARAMS="\
    --print_to both \
    --mode "train" \
    --data_id ${DATASET} \
    --phi_range 2pi \
    --centered False \
    "
CUDA_VISIBLE_DEVICES="" TS_SOCKET="cpu" PYTHONPATH=/home/$USER/devel/projects/projectneiss2d/tf_neiss:$PYTHONPATH tsp python -u ./tf_neiss/input_fn/input_fn_2d/data_gen_2dt/data_generator_t2d.py ${PARAMS} "$@"

# example call: $> sh ./tf_neiss/scripts/experiments/find_ambiguous_triangle/make_data_full_circle.sh
