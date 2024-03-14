#pragma once

#include "libinroluwb/geometry/pose.hpp"
#include "libinroluwb/optimizer/math/spline.hpp"
#include "libinroluwb/utils/binary.hpp"

#include <Eigen/src/Core/util/Memory.h>
#include <ceres/autodiff_cost_function.h>
#include <ceres/ceres.h>

namespace inrol {
  class FrameTransformationOptimizer {
  public:
    FrameTransformationOptimizer(
      vector<posestamped_t> poses, vector<posestamped_t> gt_poses,
      aligned_unordered_map<size_t, Vector3d> feature_points,
      aligned_unordered_map<size_t, pose_t> anchor_poses,
      Vector3d gravity_vector, double dt);

    bool solve();

    vector<posestamped_t> get_poses();
    aligned_unordered_map<size_t, Vector3d> get_feature_points();
    aligned_unordered_map<size_t, pose_t> get_anchor_poses();
    pose_t get_aligning_transformation();

  private:
    void add_residuals();
    Quaterniond _q = Quaterniond(1, 0, 0, 0);
    Vector3d _p = Vector3d(0, 0, 0);

    ceres::Problem _problem;

    vector<posestamped_t> _poses;
    vector<posestamped_t> _gt_poses;
    aligned_unordered_map<size_t, Vector3d> _feature_points;
    aligned_unordered_map<size_t, pose_t> _anchor_poses;
    Vector3d _gravity_vector;
    double _dt;
  };

  class TransformationErrorTerm {
  public:
    TransformationErrorTerm(pose_t gt, pose_t estm)
        : _gt(std::move(gt)), _estm(estm) {
    }

    template <typename T>
    bool operator()(
      const T* const p_ptr, const T* q_ptr, T* residuals_ptr) const {
      Eigen::Map<const Eigen::Matrix<T, 3, 1>> p(p_ptr);
      Eigen::Map<const Eigen::Quaternion<T>> q(q_ptr);
      Eigen::Map<Eigen::Matrix<T, 3, 1>> residuals(residuals_ptr);

      Eigen::Matrix<T, 3, 1> transformed_p =
        q.toRotationMatrix() * _estm.p.template cast<T>() + p;
      Eigen::Quaternion<T> transformed_q = q * _estm.q.template cast<T>();

      Eigen::Quaternion<T> delta_q =
        transformed_q.conjugate() * _gt.q.template cast<T>();
      Eigen::Matrix<T, 3, 1> delta_p = transformed_p - _gt.p.template cast<T>();

      residuals.template block<3, 1>(0, 0) = delta_p;
      // residuals.template block<3, 1>(3, 0) = quaterniond_logarithm(delta_q);
      return true;
    }

    static ceres::CostFunction* Create(pose_t gt, pose_t estm) {
      return new ceres::AutoDiffCostFunction<TransformationErrorTerm, 3, 3, 4>(
        new TransformationErrorTerm(gt, estm));
    }

  private:
    pose_t _gt;
    pose_t _estm;
  };
}  // namespace inrol
