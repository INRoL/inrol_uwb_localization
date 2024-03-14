#pragma once

#include "libinroluwb/utils/binary.hpp"

#include <vector>
#include <glog/logging.h>
#include <eigen3/Eigen/Sparse>

#include <memory>
#include <variant>

namespace inrol {
  using std::shared_ptr;
  using std::string;
  using std::vector;

  struct sensor_noise_param_t {
    double accel;
    double gyro;
    double bias_a;
    double bias_w;
    double pixel;
    double anchor;
  };

  struct camera_param_t {
    double fx;
    double fy;
    double cx;
    double cy;
    double k1;
    double k2;
    double p1;
    double p2;
    double timeshift;
  };

  struct spline_param_t {
    double spline_pose_dt;
    double spline_bias_dt;
  };

  struct harmonics_param_t {
    size_t degree;
  };

  struct anchor_param_t {
    vector<size_t> used_anchor;
  };

  struct extrinsic_param_t {
    Eigen::Matrix<double, 4, 4> T_mocap_tag;
    Eigen::Matrix<double, 4, 4> T_mocap_cam;
    Eigen::Matrix<double, 4, 4> T_cam_imu;
  };

  struct param_t {
    sensor_noise_param_t sensor_noise;
    camera_param_t camera;
    spline_param_t spline;
    harmonics_param_t harmonics;
    anchor_param_t anchor;
    extrinsic_param_t extrinsics;

    void print();
  };

  shared_ptr<param_t> get_parameters(const string& filename);

  struct gaussian_param_t {
    double sigma;
  };

  struct asymmetric_param_t {
    double sigma;
    double gamma;
  };
  double get_alpha(asymmetric_param_t param);

  using uncertainty_param_t =
    std::variant<gaussian_param_t, asymmetric_param_t>;

  aligned_unordered_map<size_t, Eigen::Matrix<double, 3, 3>> get_orientations(
    const string& filename);

}  // namespace inrol
