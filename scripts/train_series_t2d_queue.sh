#!/usr/bin/env bash
# v0.1
ID_SERIES=$1
shift
GPU=$1
shift
QUEUE="gpu${GPU}"
EPOCH=40
set -xe

DATASET="t2d_tf_314_1M_po_B"
DATALEN=314
mkdir -p models/dataset_${DATASET}/${ID_SERIES}
for parameter in 0.0 5.0 10.0 #20.0 30.0 40.0 50.0 70.0 100.0 130.0 160.0
do
    ID="${ID_SERIES}_${parameter}"
    PARAMS="
      --train_lists lists/${DATASET}_train.lst
      --val_list lists/${DATASET}_val.lst
      --checkpoint_dir models/dataset_${DATASET}/${ID_SERIES}/${ID}
      --epochs $EPOCH
      --model_type ModelTriangleArea
      --graph GraphConv1MultiFF
      --graph_params area_pred=True pre_points_out=False
      --loss_mode relativeError
      --print_to file
      --data_len ${DATALEN}
      --calc_ema True
      --optimizer FinalDecayOptimizer
      --input_fn_params min_fov=${parameter} max_fov=180.0
      --optimizer_params learning_rate=0.001 lr_decay_rate=0.9
"

    TS_SOCKET=${QUEUE} PYTHONPATH=/home/$USER/devel/projects/projectneiss2d/tf_neiss:$PYTHONPATH tsp  python -u ./tf_neiss/trainer/trainer_types/trainer_2dt/trainer_triangle2d_area.py ${PARAMS} "$@"
#    LAV_PARAMS="
#      --val_list lists/${DATASET}_val.lst
#      --model_dir models/dataset_${DATASET}/${ID}
#      --input_fn_params min_fov=1.0 max_fov=180.0
#      --data_len ${DATALEN}
#      --batch_limiter 1
#      "
#    CUDA_VISIBLE_DEVICES="" PYTHONPATH=/home/$USER/devel/projects/projectneiss2d/tf_neiss:$PYTHONPATH python -u ./tf_neiss/trainer/lav_types/lav_triangle2d.py ${LAV_PARAMS} >> "models/dataset_${DATASET}/${ID}/lav-${ID}.log" 2>&1
done
# example call: $> sh ./tf_bic/train_bic_queue.sh