#!/bin/bash

dataset=${1}
shot=${2}
for fold in 0 1 2;
do
    python main_oss.py  \
        --benchmark ${dataset} \
        --nshot ${shot} \
        --max_sample_iterations 64 \
        --box_nms_thresh 0.65 \
        --sample-range "(1,6)" \
        --topk_scores_threshold 0.0 \
        --use_dense_mask 1 \
        --use_points_or_centers \
        --purity_filter 0.02 \
        --iou_filter 0.85 \
        --multimask_output 1 \
        --sel_stability_score_thresh 0.90 \
        --use_score_filter \
        --alpha 1.0 --beta 0. --exp 0. \
        --num_merging_mask 9  \
        --fold ${fold} --log-root "output/${dataset}_${shot}shot/fold${fold}" &
done
wait
