#include "libinroluwb/optimizer/align.hpp"
#include "libinroluwb/optimizer/math/spline.hpp"

namespace inrol {
  FrameTransformationOptimizer::FrameTransformationOptimizer(
    vector<posestamped_t> poses, vector<posestamped_t> gt_poses,
    aligned_unordered_map<size_t, Vector3d> feature_points,
    aligned_unordered_map<size_t, pose_t> anchor_poses, Vector3d gravity_vector,
    double dt)
      : _poses(poses),
        _gt_poses(gt_poses),
        _feature_points(feature_points),
        _anchor_poses(anchor_poses),
        _gravity_vector(gravity_vector),
        _dt(dt) {
    _poses.pop_back();
    add_residuals();
  }

  void FrameTransformationOptimizer::add_residuals() {
    ceres::LossFunction* loss_function = nullptr;
    ceres::Manifold* quaternion_manifold = new ceres::EigenQuaternionManifold;

    for (auto const& gt_pose : _gt_poses) {
      pose_t gt = gt_pose.pose;

      double ts = gt_pose.timestamp;
      if (ts < 0) {
        continue;
      }
      size_t idx = floor(ts / _dt);
      double t1 = idx * _dt;
      double u = (ts - t1) / _dt;

      //   << "timestamps are not matched (" << t1 << ", "
      //   << _poses.at(idx + 1).timestamp << ")";
      if (idx + 5 > _poses.size())
        continue;
      CHECK_NEAR(t1, _poses.at(idx + 1).timestamp, 1e-6);

      pose_t pose0 = _poses.at(idx).pose;
      pose_t pose1 = _poses.at(idx + 1).pose;
      pose_t pose2 = _poses.at(idx + 2).pose;
      pose_t pose3 = _poses.at(idx + 3).pose;
      Quaterniond q =
        quaternion_b_spline_(pose0.q, pose1.q, pose2.q, pose3.q, u);
      Vector3d p = position_b_spline_(pose0.p, pose1.p, pose2.p, pose3.p, u);
      pose_t estm = {
        .p = p,
        .q = q,
      };
      ceres::CostFunction* cost_function =
        TransformationErrorTerm::Create(gt, estm);

      _problem.AddResidualBlock(
        cost_function, loss_function, _p.data(), _q.coeffs().data());
    }
    _problem.SetManifold(_q.coeffs().data(), quaternion_manifold);
  }

  bool FrameTransformationOptimizer::solve() {
    ceres::Solver::Options options;
    options.max_num_iterations = 200;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &_problem, &summary);

    LOG(INFO) << summary.FullReport();
    LOG(INFO) << "optimized p: " << _p.transpose();
    LOG(INFO) << "optimized q: " << _q.x() << ", " << _q.y() << ", " << _q.z()
              << ", " << _q.w();

    for (auto& pose : _poses) {
      pose.pose.p = _q.toRotationMatrix() * pose.pose.p + _p;
      pose.pose.q = _q * pose.pose.q;
    }
    for (auto& anchor_pose : _anchor_poses) {
      anchor_pose.second.p = _q.toRotationMatrix() * anchor_pose.second.p + _p;
      anchor_pose.second.q = _q * anchor_pose.second.q;
    }
    for (auto& point : _feature_points) {
      point.second = _q.toRotationMatrix() * point.second + _p;
    }
    return summary.IsSolutionUsable();
  }

  vector<posestamped_t> FrameTransformationOptimizer::get_poses() {
    return _poses;
  }

  aligned_unordered_map<size_t, Vector3d>
  FrameTransformationOptimizer::get_feature_points() {
    return _feature_points;
  }

  aligned_unordered_map<size_t, pose_t>
  FrameTransformationOptimizer::get_anchor_poses() {
    return _anchor_poses;
  }

  pose_t FrameTransformationOptimizer::get_aligning_transformation() {
    return pose_t {.p = _p, .q = _q};
  }
}  // namespace inrol
