#!/usr/bin/env bash
# v0.1
ID=$1
shift
set -xe
PARAMS="
--print_to file \
--data_id $ID \
--centered False \
"

PYTHONPATH=/home/$USER/devel/projects/projectneiss2d/tf_neiss:$PYTHONPATH python -u ./tf_neiss/input_fn/input_fn_2d/data_gen_2dt/data_generator_t2d.py ${PARAMS} "$@"

# example call: $> sh ./tf_bic/train_bic_queue.s