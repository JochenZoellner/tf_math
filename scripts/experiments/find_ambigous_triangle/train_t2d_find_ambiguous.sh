#!/usr/bin/env bash
# v0.1
ID=$1
shift
GPU=$1
shift
QUEUE="gpu${GPU}"
EPOCH=200
set -e

DATASET="t2d_find_ambiguous"
DATALEN=312

mkdir -p models/dataset_${DATASET}

PARAMS="
  --train_lists lists/${DATASET}_train.lst
  --val_list lists/${DATASET}_val.lst
  --checkpoint_dir models/dataset_${DATASET}/${ID}
  --epochs $EPOCH
  --model_type ModelTriangle
  --graph GraphConv1MultiFF
  --loss_mode input_diff
  --print_to file
  --data_len ${DATALEN}
  --calc_ema True
  --optimizer FinalDecayOptimizer
  --input_params max_fov=180.0 min_fov=0.0
  --optimizer_params learning_rate=0.001 lr_decay_rate=0.95
  --gpu_devices ${GPU}
"
CUDA_VISIBLE_DEVICES="" TS_SOCKET=${QUEUE} PYTHONPATH=$(pwd)/tf_neiss:$PYTHONPATH python -u ./tf_neiss/trainer/trainer_types/trainer_2dt/trainer_triangle2d_area.py ${PARAMS} "$@"

<<////
# example call:
sh ./tf_neiss/scripts/experiments/t2d_find_ambiguous/train_t2d_find_ambiguous.sh find_ambiguous_1 0
////
