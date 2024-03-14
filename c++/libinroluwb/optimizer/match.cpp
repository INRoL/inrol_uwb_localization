#include "libinroluwb/optimizer/match.hpp"
#include "libinroluwb/geometry/pose.hpp"

#include <ceres/manifold.h>
#include <ceres/types.h>

namespace inrol {
  using std::floor;

  SplineMatchingOptimizer::SplineMatchingOptimizer(
    const vector<posestamped_t>& initial_knots,
    const vector<posestamped_t>& colmap_poses, double dt)
      : _knots(initial_knots), _colmap_poses(colmap_poses), _dt(dt) {
    _optimized_poses_number = _knots.size();

    add_residuals();
  }

  void SplineMatchingOptimizer::add_residuals() {
    ceres::LossFunction* loss_function = nullptr;
    ceres::Manifold* quaternion_manifold = new ceres::EigenQuaternionManifold;

    for (auto const& colmap_pose : _colmap_poses) {
      pose_t pose = colmap_pose.pose;

      double ts = colmap_pose.timestamp;
      if (ts < 0) {
        continue;
      }
      size_t idx = floor(ts / _dt);
      double t1 = idx * _dt;
      double u = (ts - t1) / _dt;

      CHECK(t1 == _knots.at(idx + 1).timestamp)
        << "timestamps are not matched (" << t1 << ", "
        << _knots.at(idx + 1).timestamp << ")";

      pose_t* pose0_iter = &_knots.at(idx).pose;
      pose_t* pose1_iter = &_knots.at(idx + 1).pose;
      pose_t* pose2_iter = &_knots.at(idx + 2).pose;
      pose_t* pose3_iter = &_knots.at(idx + 3).pose;

      ceres::CostFunction* cost_function =
        PoseMatchingErrorTerm::Create(pose, u);

      _problem.AddResidualBlock(
        cost_function, loss_function, pose0_iter->p.data(),
        pose0_iter->q.coeffs().data(), pose1_iter->p.data(),
        pose1_iter->q.coeffs().data(), pose2_iter->p.data(),
        pose2_iter->q.coeffs().data(), pose3_iter->p.data(),
        pose3_iter->q.coeffs().data());

      _problem.SetManifold(pose0_iter->q.coeffs().data(), quaternion_manifold);
      _problem.SetManifold(pose1_iter->q.coeffs().data(), quaternion_manifold);
      _problem.SetManifold(pose2_iter->q.coeffs().data(), quaternion_manifold);
      _problem.SetManifold(pose3_iter->q.coeffs().data(), quaternion_manifold);
    }
  }

  bool SplineMatchingOptimizer::solve() {
    ceres::Solver::Options options;
    options.max_num_iterations = 200;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &_problem, &summary);

    LOG(INFO) << summary.FullReport();
    return summary.IsSolutionUsable();
  }

  vector<posestamped_t> SplineMatchingOptimizer::get_pose_knots() {
    return _knots;
  }
}  // namespace inrol
