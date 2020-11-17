#!/usr/bin/env bash
# v0.1
ID=$1
shift
GPU=$1
shift
QUEUE="gpu${GPU}"
EPOCH=200
set -e

DATASET="t2d_find_ambiguous_FC"
DATALEN=614

mkdir -p models/dataset_${DATASET}
# ToDo replace val with train
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
CUDA_VISIBLE_DEVICES="" TS_SOCKET=${QUEUE} PYTHONPATH=$(pwd)/tf_neiss:$PYTHONPATH tsp python -u ./tf_neiss/trainer/trainer_types/trainer_2dt/trainer_triangle2d.py ${PARAMS} "$@"

<<////
# example call:
sh ./tf_neiss/scripts/experiments/find_ambiguous_triangle/train_t2d_find_ambiguous_full_circle.sh find_ambiguous_FC_reference 0
sh ./tf_neiss/scripts/experiments/find_ambiguous_triangle/train_t2d_find_ambiguous_full_circle..sh find_ambiguous_FC_reference_lr0.002 1 --optimizer_params learning_rate=0.002 lr_decay_rate=0.95
sh ./tf_neiss/scripts/experiments/find_ambiguous_triangle/train_t2d_find_ambiguous_full_circle..sh find_ambiguous_FC_reference_lrd0.98 0 --optimizer_params learning_rate=0.001 lr_decay_rate=0.98
sh ./tf_neiss/scripts/experiments/find_ambiguous_triangle/train_t2d_find_ambiguous_full_circle..sh find_ambiguous_FC_reference_lr0.0005_lrd0.98 1 --optimizer_params learning_rate=0.0005 lr_decay_rate=0.98
sh ./tf_neiss/scripts/experiments/find_ambiguous_triangle/train_t2d_find_ambiguous_full_circle..sh find_ambiguous_FC_reference_swish 0  --graph_prams ff_activation=swish
sh ./tf_neiss/scripts/experiments/find_ambiguous_triangle/train_t2d_find_ambiguous_full_circle..sh find_ambiguous_FC_reference_lr0.0005_lrd0.98_swish 1 --optimizer_params learning_rate=0.0005 lr_decay_rate=0.98 --graph_prams ff_activation=swish
sh ./tf_neiss/scripts/experiments/find_ambiguous_triangle/train_t2d_find_ambiguous_full_circle..sh find_ambiguous_FC_reference_lr0.0005_lrd0.98_xlayer256 0 --optimizer_params learning_rate=0.0005 lr_decay_rate=0.98 --graph_prams=[512,1024,1024,512,256,256,256]

////
