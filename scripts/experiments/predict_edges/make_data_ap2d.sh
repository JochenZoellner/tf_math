#!/usr/bin/env bash
# v0.1
set -e
DATASET=$1
shift

PARAMS="
--print_to file \
--mode "val" \
--data_id ${DATASET} \
--centered False \
--min_edges 3 \
--max_edges 4 \
"

CUDA_VISIBLE_DEVICES="" TS_SOCKET="cpu" PYTHONPATH=/home/$USER/devel/projects/projectneiss2d/tf_neiss:$PYTHONPATH tsp python -u ./tf_neiss/input_fn/input_fn_2d/data_gen_2dt/data_generator_ap2d.py ${PARAMS} "$@"
PARAMS="
--print_to file \
--mode "train" \
--data_id ${DATASET} \
--centered False \
--min_edges 3 \
--max_edges 4 \
"

CUDA_VISIBLE_DEVICES="" TS_SOCKET="cpu" PYTHONPATH=/home/$USER/devel/projects/projectneiss2d/tf_neiss:$PYTHONPATH tsp python -u ./tf_neiss/input_fn/input_fn_2d/data_gen_2dt/data_generator_ap2d.py ${PARAMS} "$@"

# example call: $> sh ./tf_neiss/scripts/experiments/predict_edges/make_data_ap2d.sh ap2d_3to4edge --max_edges 4
# example call: $> sh ./tf_neiss/scripts/experiments/predict_edges/make_data_ap2d.sh ap2d_3to7edge --max_edges 7
# example call: $> sh ./tf_neiss/scripts/experiments/predict_edges/make_data_ap2d.sh ap2d_3to12edge --max_edges 12
