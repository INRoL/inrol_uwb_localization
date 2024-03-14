#pragma once

#include <stddef.h>
#include <Eigen/Dense>

#include <variant>

namespace inrol {
  struct imu_data_t {
    double timestamp;
    double wx;
    double wy;
    double wz;
    double ax;
    double ay;
    double az;
  };

  struct uwb_data_t {
    double timestamp;
    size_t anchor_idx;
    double distance;
  };

  struct vision_data_t {
    double timestamp;
    size_t point3d_id;
    Eigen::Vector2d uv;
  };

  struct uwb_error_t {
    double timestamp;
    size_t anchor_idx;
    double error;
  };

  using data_t = std::variant<imu_data_t, uwb_data_t, vision_data_t>;
}  // namespace inrol
