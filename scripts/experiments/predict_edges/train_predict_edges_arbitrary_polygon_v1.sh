#!/usr/bin/env bash
# v0.1
set -e
ID=$1
shift
DATASET=$1
shift
GPU=$1
shift
QUEUE="gpu${GPU}"
EPOCH=100

mkdir -p models/${DATASET}

PARAMS="
    --train_lists lists/${DATASET}_train.lst \
    --val_list lists/${DATASET}_val.lst \
    --checkpoint_dir models/${DATASET}/${ID} \
    --optimizer_params learning_rate=0.0001 \
    --val_batch_size 1000\
    --calc_ema True \
    --data_len 312 \
    --epochs ${EPOCH} \
    --delete_event_files False \
    --model_type ModelPolygonClassifier \
    --input_type InputFnArbitraryPolygon2D \
    --graph GraphConv1MultiFF \
    --graph_params pre_radius=False pre_rotation=False pre_translation=False \
    --max_edges 12 \
    --loss_mode softmax_crossentropy \
    --print_to both \
    --gpu_devices ${GPU}
"

CUDA_VISIBLE_DEVICES="" TS_SOCKET=${QUEUE} PYTHONPATH=/home/$USER/devel/projects/projectneiss2d/tf_neiss:$PYTHONPATH tsp python -u ./tf_neiss/trainer/trainer_types/trainer_2dt/trainer_polygon2d_classifier.py ${PARAMS} "$@"

<< ////
# example call:
sh ./tf_neiss/scripts/experiments/predict_edges/train_predict_edges_regular_polygon_v1.sh test_A ap2d_3to4edge 0 --max_edges 4
sh ./tf_neiss/scripts/experiments/predict_edges/train_predict_edges_regular_polygon_v1.sh test_A ap2d_3to7edge 0 --max_edges 7
sh ./tf_neiss/scripts/experiments/predict_edges/train_predict_edges_regular_polygon_v1.sh test_A ap2d_3to12edge 0 --max_edges 12
////