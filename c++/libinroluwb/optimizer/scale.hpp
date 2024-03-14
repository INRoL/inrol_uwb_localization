#pragma once

#include "libinroluwb/geometry/measurements.hpp"
#include "libinroluwb/geometry/pose.hpp"
#include "libinroluwb/optimizer/math/spline.hpp"
#include "libinroluwb/utils/binary.hpp"
#include "libinroluwb/utils/param.hpp"

#include <Eigen/src/Core/Matrix.h>
#include <Eigen/src/Core/util/Memory.h>
#include <ceres/autodiff_cost_function.h>
#include <ceres/ceres.h>

namespace inrol {
  class ScaleFindingOptimizer {
  public:
    ScaleFindingOptimizer(
      const vector<posestamped_t>& scaleless_knots,
      const vector<uwb_data_t>& uwb_dataset, shared_ptr<param_t> param);

    bool solve();
    double get_scale_variable();
    aligned_unordered_map<size_t, Vector3d> get_anchorposition_variable();
    vector<posestamped_t> get_scaled_trajectory();

  private:
    void add_residuals();

    vector<posestamped_t> _knots;
    vector<uwb_data_t> _uwb_dataset;

    shared_ptr<param_t> _param;
    vector<size_t> _used_anchor;

    size_t _uwb_dataset_number;
    double _dt;
    pose_t _T_cam_tag;
    int _degree;

    aligned_unordered_map<size_t, Vector3d> _anchor_position;
    double _scale;

    ceres::Problem _problem;
  };
}  // namespace inrol
