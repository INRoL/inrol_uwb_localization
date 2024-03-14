#pragma once

#include "libinroluwb/geometry/measurements.hpp"
#include "libinroluwb/geometry/pose.hpp"
#include "libinroluwb/optimizer/logging.hpp"
#include "libinroluwb/optimizer/batch.hpp"
#include "libinroluwb/utils/param.hpp"
#include "libinroluwb/utils/binary.hpp"

#include <map>
#include <optional>

namespace inrol {
  using Eigen::Matrix;
  using Eigen::Matrix3d;
  using std::map;
  using std::multimap;
  using std::optional;

  struct filter_input_t {
    const vector<imu_data_t> imu_dataset;
    const vector<uwb_data_t> uwb_dataset;
    const aligned_unordered_map<size_t, pose_t> anchor_poses;
    const vector<posestamped_t> gt_poses;
    const ModelType type;
    const uncertainty_param_t uncertainty_param;
    const vector<double> tag_coeffs;
    const vector<double> anchor_coeffs;
    const shared_ptr<param_t> param;
  };

  class KalmanFilter {
  public:
    KalmanFilter(filter_input_t input);

    void run();
    vector<posestamped_t> get_trajectory();
    double get_RMSE();

  private:
    void propagation(const imu_data_t imu, const double dt);
    void correction(const uwb_data_t uwb);
    double get_directional_bias(pose_t tag_pose, pose_t anchor_pose);

    Matrix<double, 15, 15> get_error_Phi(
      const Vector3d am, const Vector3d wm, const double dt);
    Matrix<double, 1, 15> get_measurement_jacobian(
      Vector3d p, Matrix3d R, pose_t anchor_pose);

    const aligned_unordered_map<size_t, pose_t> _anchor_poses;
    Vector3d _gravity_vector;
    const vector<double> _tag_coeffs;
    const vector<double> _anchor_coeffs;
    const shared_ptr<param_t> _param;
    ModelType _type;
    uncertainty_param_t _uncertainty_param;
    pose_t _T_imu_tag;
    int _degree;
    vector<int> _degree_container;
    vector<int> _order_container;

    multimap<double, data_t> _data;

    Matrix3d _R;
    Vector3d _v;
    Vector3d _p;
    Vector3d _wb;
    Vector3d _ab;

    Matrix<double, 15, 15> _P;
    Matrix<double, 15, 15> _Q;
    double _su;
    double _gu;

    double _last_timestamp = 0;
    optional<imu_data_t> _last_imu = std::nullopt;
    vector<posestamped_t> _estimated_traj;
    vector<posestamped_t> _gt_poses;
    map<size_t, optional<double>> _last_uwb_data;
  };
}  // namespace inrol
