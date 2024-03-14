#pragma once

#include <eigen3/Eigen/Dense>

#include "glog/logging.h"

namespace inrol {
  using Eigen::Matrix3d;
  using Eigen::Matrix4d;
  using Eigen::Quaterniond;
  using Eigen::Vector3d;
  using std::vector;

  struct pose_t {
    Vector3d p;
    Quaterniond q;

    pose_t product(pose_t pose);
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
  };

  struct posestamped_t {
    pose_t pose;
    double timestamp;
  };

  posestamped_t pose_interpolation(
    posestamped_t ps_a, posestamped_t ps_b, double ts);

  vector<posestamped_t> get_interpolated_knots(
    const vector<posestamped_t>& input_traj, double dt);

  pose_t pose_b_spline(
    pose_t pose0, pose_t pose1, pose_t pose2, pose_t pose3, double u);

  pose_t convert_eigen_to_pose_t(Matrix4d m);
}  // namespace inrol
