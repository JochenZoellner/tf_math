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

    LAV_PARAMS="
      --val_list lists/${DATASET}_val.lst
      --model_dir models/dataset_${DATASET}/${ID}
      --input_fn_params min_fov=0.0 max_fov=180.0
      --plot True
      --plot_params select=all select_counter=200 filename=plot_summary.pdf
      --data_len ${DATALEN}
      --batch_limiter -1
      --graph GraphConv1MultiFF
      --loss_mode best_point_diff best_point_diff_invariant
      "
    # shellcheck disable=SC2068
    CUDA_VISIBLE_DEVICES="" PYTHONPATH=/home/$USER/devel/projects/projectneiss2d/tf_neiss:$PYTHONPATH python -u ./tf_neiss/trainer/lav_types/lav_triangle2d.py ${LAV_PARAMS} ${@} >> "models/dataset_${DATASET}/${ID}/lav-${ID}.log" 2>&1

<< ////
# example call:
sh ./tf_neiss/scripts/experiments/find_ambiguous_triangle/eval_t2d_find_ambiguous.sh find_ambiguous_1 0
sh ./tf_neiss/scripts/experiments/find_ambiguous_triangle/eval_t2d_find_ambiguous.sh find_ambiguous_1 0 --plot_params select=1 select_counter=200 filename=select1.pdf


////<< ////
# example call:
sh ./tf_neiss/scripts/experiments/find_ambiguous_triangle/eval_t2d_find_ambiguous.sh FA_FC_reference_lr0.0002_lrd0.98_swish 1 --plot_params select=1 select_counter=200 filename=select1.pdf &
sh ./tf_neiss/scripts/experiments/find_ambiguous_triangle/eval_t2d_find_ambiguous.sh FA_FC_reference_lr0.0002_lrd0.98 1 --plot_params select=1 select_counter=200 filename=select1.pdf &
sh ./tf_neiss/scripts/experiments/find_ambiguous_triangle/eval_t2d_find_ambiguous.sh FA_absolute_FC_reference_lr0.0002_lrd0.98_swish 1 --plot_params select=1 select_counter=200 filename=select1.pdf &
sh ./tf_neiss/scripts/experiments/find_ambiguous_triangle/eval_t2d_find_ambiguous.sh FA_absolute_reference_lr0.0001 1 --plot_params select=1 select_counter=200 filename=select1.pdf &
sh ./tf_neiss/scripts/experiments/find_ambiguous_triangle/eval_t2d_find_ambiguous.sh FA_absolute_reference_lr0.0002_lrd0.98 1 --plot_params select=1 select_counter=200 filename=select1.pdf &
sh ./tf_neiss/scripts/experiments/find_ambiguous_triangle/eval_t2d_find_ambiguous.sh FA_absolute_reference_lr0.0002_lrd0.98_swish 1 --plot_params select=1 select_counter=200 filename=select1.pdf &
sh ./tf_neiss/scripts/experiments/find_ambiguous_triangle/eval_t2d_find_ambiguous.sh FA_absolute_s_norm_reference_lr0.0002_lrd0.98 1 --plot_params select=1 select_counter=200 filename=select1.pdf &
sh ./tf_neiss/scripts/experiments/find_ambiguous_triangle/eval_t2d_find_ambiguous.sh FA_absolute_s_norm_reference_lr0.0002_lrd0.98_swish 1 --plot_params select=1 select_counter=200 filename=select1.pdf &
sh ./tf_neiss/scripts/experiments/find_ambiguous_triangle/eval_t2d_find_ambiguous.sh FA_reference_lr0.0002_lrd0.98_swish 1 --plot_params select=1 select_counter=200 filename=select1.pdf &
sh ./tf_neiss/scripts/experiments/find_ambiguous_triangle/eval_t2d_find_ambiguous.sh FA_s_norm_reference_lr0.0002_lrd0.98 1 --plot_params select=1 select_counter=200 filename=select1.pdf &
sh ./tf_neiss/scripts/experiments/find_ambiguous_triangle/eval_t2d_find_ambiguous.sh FA_s_norm_reference_lr0.0002_lrd0.98_swish 1 --plot_params select=1 select_counter=200 filename=select1.pdf &
FA_absolute_reference_lr0.0002_lrd0.98
sh ./tf_neiss/scripts/experiments/find_ambiguous_triangle/eval_t2d_find_ambiguous.sh find_ambiguous_1 1
sh ./tf_neiss/scripts/experiments/find_ambiguous_triangle/eval_t2d_find_ambiguous.sh find_ambiguous_1 1
sh ./tf_neiss/scripts/experiments/find_ambiguous_triangle/eval_t2d_find_ambiguous.sh find_ambiguous_1 1
sh ./tf_neiss/scripts/experiments/find_ambiguous_triangle/eval_t2d_find_ambiguous.sh find_ambiguous_1 1



sh ./tf_neiss/scripts/experiments/find_ambiguous_triangle/eval_t2d_find_ambiguous.sh find_ambiguous_1 0 --plot_params select=1 select_counter=200 filename=select1.pdf
////
