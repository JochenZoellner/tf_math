#!/usr/bin/env bash
# v0.1
ID_SERIES=$1
shift
GPU=$1
shift
QUEUE="gpu${GPU}"
EPOCH=40
set -xe

DATASET="t2d_tf_314_1M_po"
DATALEN=314
mkdir -p models/dataset_${DATASET}/${ID_SERIES}
for parameter in 26.0 27.0 28.0 29.0 31.0 32.0 33.0 34.0 36.0 37.0 38.0 39.0 41.0 42.0 43.0\
 44.0 46.0 47.0 48.0 49.0 55.0 60.0 65.0 80.0 90.0 110.0 120.0 140.0 150.0 170.0 175.0\
 0.0 5.0 10.0 20.0 30.0 40.0 50.0 70.0 100.0 130.0 160.0 12.5 15.0 17.5 22.5 25.0 27.5 32.5 35.0 37.5 42.5 45.0 
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
      --input_params min_fov=${parameter} max_fov=180.0
      --optimizer_params learning_rate=0.001 lr_decay_rate=0.9
      --gpu_devices ${GPU}
"

    CUDA_VISIBLE_DEVICES="" TS_SOCKET=${QUEUE} PYTHONPATH=/home/$USER/devel/projects/projectneiss2d/tf_neiss:$PYTHONPATH tsp  python -u ./tf_neiss/trainer/trainer_types/trainer_2dt/trainer_triangle2d_area.py ${PARAMS} "$@"
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
