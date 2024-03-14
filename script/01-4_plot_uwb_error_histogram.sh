#!/bin/bash
python3 python/plot_error_histogram.py \
  --uwb example/model_test/parsed_dataset/uwb.txt \
  --tag_pose example/model_test/parsed_dataset/tag_pose.txt \
  --anchor example/model_test/parsed_dataset/anchor.txt \
  --a_dir example/model_calibration/01-1_model_calibration_A \
  --b_dir example/model_calibration/01-2_model_calibration_B \
  --c_dir example/model_calibration/01-3_model_calibration_C
