#!/bin/bash
python3 python/plot_anchor_poses.py \
  --gt_poses1 example/model_test/parsed_dataset/anchor.txt \
  --gt_poses2 example/anchor_self_calibration1/parsed_dataset/anchor.txt \
  --poses2 example/anchor_self_calibration1/02-7_full_batch_optimization_C/aligned_anchor_poses.txt \
  --gt_poses3 example/anchor_self_calibration2/parsed_dataset/anchor.txt \
  --poses3 example/anchor_self_calibration2/02-7_full_batch_optimization_C/aligned_anchor_poses.txt
