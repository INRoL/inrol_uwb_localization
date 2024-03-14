#pragma once

#include "libinroluwb/geometry/measurements.hpp"
#include "libinroluwb/geometry/pose.hpp"
#include "libinroluwb/optimizer/math/spline.hpp"
#include "libinroluwb/optimizer/calibration.hpp"
#include "libinroluwb/optimizer/logging.hpp"
#include "libinroluwb/utils/binary.hpp"
#include "libinroluwb/utils/param.hpp"

#include <Eigen/src/Core/Matrix.h>

namespace inrol {
  using inttuple_t = std::tuple<int, int>;

  struct batch_input_t {
    const vector<posestamped_t> pose_knots_initial_guess;
    const aligned_unordered_map<size_t, Vector3d> feature_points_initial_guess;
    const aligned_unordered_map<size_t, Vector3d> anchor_position_initial_guess;
    const Vector3d gravity_vector;
    const vector<double> tag_coeffs;
    const vector<double> anchor_coeffs;
    const vector<imu_data_t> imu_dataset;
    const vector<uwb_data_t> uwb_dataset;
    const vector<vision_data_t> vision_dataset;
    const Camera camera;
    const shared_ptr<param_t> param;
    const ModelType type;
    const uncertainty_param_t uncertainty_param;
    const aligned_unordered_map<size_t, pose_t> anchor_pose_prior;
  };

  class FullBatchOptimizer {
  public:
    FullBatchOptimizer(batch_input_t batch_input);

    bool solve();
    vector<posestamped_t> get_pose_knots();
    vector<Vector3d> get_accel_bias_knots();
    vector<Vector3d> get_gyro_bias_knots();
    aligned_unordered_map<size_t, Vector3d> get_feature_points();
    aligned_unordered_map<size_t, pose_t> get_anchor_poses();
    Vector3d get_gravity_vector();

  private:
    void print_problem();
    void add_residuals();

    void add_vision_residuals();
    void add_imu_residuals();
    void add_bias_residuals();
    void add_uwb_residuals();
    void add_anchor_prior_residuals();

    vector<posestamped_t> _pose_knots;
    vector<Vector3d> _accel_bias_knots;
    vector<Vector3d> _gyro_bias_knots;
    aligned_unordered_map<size_t, Vector3d> _feature_points;
    aligned_unordered_map<size_t, pose_t> _anchor_poses;
    Vector3d _gravity_vector;

    const ModelType _type;
    const uncertainty_param_t _uncertainty_param;
    const size_t _degree;
    const vector<double> _tag_coeffs;
    const vector<double> _anchor_coeffs;

    const vector<imu_data_t> _imu_dataset;
    const vector<uwb_data_t> _uwb_dataset;
    const vector<vision_data_t> _vision_dataset;

    const shared_ptr<Camera> _camera;
    const shared_ptr<param_t> _param;
    const double _duration;

    ceres::Problem _problem;
  };
}  // namespace inrol
