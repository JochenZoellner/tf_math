#!/usr/bin/env bash

ID=$1
shift
EPOCH=100
set -xe
DATASET="t2d_tf_314_1M_po"
DATALEN=316
PARAMS="
  --train_lists lists/${DATASET}_train.lst
  --val_list lists/${DATASET}_val.lst
  --checkpoint_dir models/dataset_${DATASET}/${ID}
  --epochs $EPOCH
  --graph GraphMultiFF
  --print_to file
  --data_len ${DATALEN}
  --optimizer FinalDecayOptimizer
  --loss_mode point_diff
  --input_fn_params min_fov=1.0 max_fov=180.0
  --gpu_devices 0
  --optimizer_params learning_rate=0.001
"

PYTHONPATH=/home/$USER/devel/projects/projectneiss2d/tf_neiss:$PYTHONPATH python -u ./tf_neiss/trainer/trainer_types/trainer_2dt/trainer_triangle2d.py ${PARAMS} "$@"
LAV_PARAMS="
  --val_list lists/${DATASET}_val.lst
  --model_dir models/dataset_${DATASET}/${ID}
  --input_fn_params min_fov=1.0 max_fov=180.0
  --data_len ${DATALEN}
  --batch_limiter 1
  "

CUDA_VISIBLE_DEVICES="" PYTHONPATH=/home/$USER/devel/projects/projectneiss2d/tf_neiss:$PYTHONPATH python -u ./tf_neiss/trainer/lav_types/lav_triangle2d.py ${LAV_PARAMS} >> "models/dataset_${DATASET}/${ID}/lav-${ID}.log" 2>&1

# example call: $> sh ./tf_bic/train_bic_queue.s