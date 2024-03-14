#include "libinroluwb/geometry/feature.hpp"
#include "libinroluwb/geometry/pose.hpp"

namespace inrol {
  void Frame::add_feature_measurement(feature_measurement_t fm) {
    _feature_measurements.push_back(fm);
  }

  void Frame::add_vision_data(
    const vector<double> timestamps, vector<vision_data_t>& data) {
    double ts = timestamps.at(_id);
    for (auto const& fm : _feature_measurements) {
      data.push_back(vision_data_t {
        .timestamp = ts, .point3d_id = fm.first, .uv = fm.second});
    }
  }

  void Frame::set_pose(
    double tx, double ty, double tz, double qx, double qy, double qz,
    double qw) {
    Vector3d p = Vector3d(tx, ty, tz);
    Quaterniond q = Quaterniond(qw, qx, qy, qz);
    _pose.p = -q.conjugate().toRotationMatrix() * p;
    _pose.q = q.conjugate();
  }

  pose_t Frame::get_pose() {
    return _pose;
  }
}  // namespace inrol
