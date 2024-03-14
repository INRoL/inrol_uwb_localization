#!/bin/bash
python3 python/analyze_trajectories.py \
  --gt example/filtering_state_estimation2/parsed_dataset/imu_pose.txt \
  --traj1 example/filtering_state_estimation2/03-1_kalman_filter_A/estimated_traj.txt \
  --traj2 example/filtering_state_estimation2/03-2_kalman_filter_AH/estimated_traj.txt \
  --traj3 example/filtering_state_estimation2/03-3_kalman_filter_B/estimated_traj.txt \
  --traj4 example/filtering_state_estimation2/03-4_kalman_filter_BH/estimated_traj.txt \
  --traj5 example/filtering_state_estimation2/03-5_kalman_filter_C/estimated_traj.txt \
  --start 140 \
  --end 180
