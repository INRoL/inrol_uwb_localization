#include "libinroluwb/optimizer/scale.hpp"
#include "libinroluwb/optimizer/logging.hpp"
#include "libinroluwb/optimizer/functor/uwb.hpp"
#include "libinroluwb/utils/algorithm.hpp"
#include "libinroluwb/utils/param.hpp"

#include <ceres/cost_function.h>
#include <ceres/loss_function.h>
#include <ceres/manifold.h>
#include <ceres/types.h>

namespace inrol {
  using std::floor;

  ScaleFindingOptimizer::ScaleFindingOptimizer(
    const vector<posestamped_t>& scaleless_knots,
    const vector<uwb_data_t>& uwb_dataset, shared_ptr<param_t> param)
      : _knots(scaleless_knots), _uwb_dataset(uwb_dataset), _param(param) {
    _used_anchor = _param->anchor.used_anchor;
    _uwb_dataset_number = _uwb_dataset.size();
    _dt = _param->spline.spline_pose_dt;

    _scale = 1.0;
    _degree = param->harmonics.degree;

    for (auto const anchor_idx : _used_anchor) {
      _anchor_position[anchor_idx] = Vector3d(0.0, 0.0, 0.0);
    }

    auto extrinsic = param->extrinsics;
    auto T_c2t = extrinsic.T_mocap_cam.inverse() * extrinsic.T_mocap_tag;
    _T_cam_tag = convert_eigen_to_pose_t(T_c2t);

    add_residuals();
  }

  vector<posestamped_t> ScaleFindingOptimizer::get_scaled_trajectory() {
    vector<posestamped_t> result;
    for (auto const& stamped : _knots) {
      pose_t pose = stamped.pose;
      pose.p = _scale * stamped.pose.p;
      result.push_back(
        posestamped_t {.pose = pose, .timestamp = stamped.timestamp});
    }

    return result;
  }

  void ScaleFindingOptimizer::add_residuals() {
    ceres::LossFunction* loss_function = nullptr;

    for (auto const& uwb : _uwb_dataset) {
      double ts = uwb.timestamp;
      size_t idx = floor(ts / _dt);
      if (ts < 0)
        continue;
      if (ts > _knots.back().timestamp - _dt)
        continue;

      double t1 = idx * _dt;
      CHECK_NEAR(t1, _knots.at(idx + 1).timestamp, 1e-6);

      double u = (ts - t1) / _dt;
      pose_t p0 = _knots.at(idx).pose;
      pose_t p1 = _knots.at(idx + 1).pose;
      pose_t p2 = _knots.at(idx + 2).pose;
      pose_t p3 = _knots.at(idx + 3).pose;

      auto cam_pose = pose_b_spline(p0, p1, p2, p3, u);
      auto tag_pose = cam_pose.product(_T_cam_tag);
      ceres::CostFunction* cost_function =
        ScaleUWBErrorTerm::Create(uwb, tag_pose);

      auto maybe_j = find_index_in_vector<size_t>(_used_anchor, uwb.anchor_idx);
      CHECK(maybe_j) << "unknown anchor index of " << uwb.anchor_idx
                     << " is measured";

      _problem.AddResidualBlock(
        cost_function, loss_function, &_scale,
        _anchor_position.at(uwb.anchor_idx).data());
    }
    _problem.SetParameterLowerBound(&_scale, 0, 0.01);
  }

  bool ScaleFindingOptimizer::solve() {
    ceres::Solver::Options options;
    options.max_num_iterations = 200;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.update_state_every_iteration = true;
    LoggingCallback logging_callback(true);
    options.callbacks.push_back(&logging_callback);
    ceres::Solver::Summary summary;
    ceres::Solve(options, &_problem, &summary);

    LOG(INFO) << summary.FullReport();
    return summary.IsSolutionUsable();
  }

  double ScaleFindingOptimizer::get_scale_variable() {
    return _scale;
  }

  aligned_unordered_map<size_t, Vector3d>
  ScaleFindingOptimizer::get_anchorposition_variable() {
    return _anchor_position;
  }
}  // namespace inrol
