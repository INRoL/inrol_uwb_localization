#pragma once

#include "libinroluwb/geometry/measurements.hpp"
#include "libinroluwb/geometry/pose.hpp"

namespace inrol {
  using feature_measurement_t = std::pair<size_t, Eigen::Vector2d>;

  struct feature_point_t {
    size_t id;
    Vector3d p;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
  };

  class Frame {
  public:
    Frame(uint32_t id): _id(id - 1) {
    }

    void add_feature_measurement(feature_measurement_t fm);

    void add_vision_data(
      const vector<double> timestamps, vector<vision_data_t>& data);

    void set_pose(
      double tx, double ty, double tz, double qx, double qy, double qz,
      double qw);

    pose_t get_pose();

  private:
    vector<feature_measurement_t> _feature_measurements;
    uint32_t _id;
    pose_t _pose;
  };
}  // namespace inrol
