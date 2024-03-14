#pragma once

#include "libinroluwb/geometry/pose.hpp"
#include "libinroluwb/optimizer/math/spline.hpp"

#include <Eigen/src/Core/util/Memory.h>
#include <ceres/autodiff_cost_function.h>
#include <ceres/ceres.h>

namespace inrol {
  using std::vector;

  class SplineMatchingOptimizer {
  public:
    SplineMatchingOptimizer(
      const vector<posestamped_t>& initial_knots,
      const vector<posestamped_t>& colmap_poses, double dt);

    bool solve();
    vector<posestamped_t> get_pose_knots();

  private:
    void add_residuals();

    vector<posestamped_t> _colmap_poses;

    size_t _optimized_poses_number;
    double _dt;

    vector<posestamped_t> _knots;
    ceres::Problem _problem;
  };

  class PoseMatchingErrorTerm {
  public:
    PoseMatchingErrorTerm(pose_t pose, double u)
        : _pose(std::move(pose)), _u(u) {
    }

    template <typename T>
    bool operator()(
      const T* const p0_ptr, const T* q0_ptr, const T* const p1_ptr,
      const T* q1_ptr, const T* const p2_ptr, const T* q2_ptr,
      const T* const p3_ptr, const T* q3_ptr, T* residuals_ptr) const {
      Eigen::Map<const Eigen::Matrix<T, 3, 1>> p0(p0_ptr);
      Eigen::Map<const Eigen::Quaternion<T>> q0(q0_ptr);
      Eigen::Map<const Eigen::Matrix<T, 3, 1>> p1(p1_ptr);
      Eigen::Map<const Eigen::Quaternion<T>> q1(q1_ptr);
      Eigen::Map<const Eigen::Matrix<T, 3, 1>> p2(p2_ptr);
      Eigen::Map<const Eigen::Quaternion<T>> q2(q2_ptr);
      Eigen::Map<const Eigen::Matrix<T, 3, 1>> p3(p3_ptr);
      Eigen::Map<const Eigen::Quaternion<T>> q3(q3_ptr);
      Eigen::Map<Eigen::Matrix<T, 6, 1>> residuals(residuals_ptr);

      Eigen::Quaternion<T> q = quaternion_b_spline(q0, q1, q2, q3, _u);
      Eigen::Matrix<T, 3, 1> p = position_b_spline(p0, p1, p2, p3, _u);

      Eigen::Quaternion<T> delta_q = q.conjugate() * _pose.q.template cast<T>();
      Eigen::Matrix<T, 3, 1> delta_p = p - _pose.p.template cast<T>();

      residuals.template block<3, 1>(0, 0) = delta_p;
      residuals.template block<3, 1>(3, 0) = quaterniond_logarithm(delta_q);
      return true;
    }

    static ceres::CostFunction* Create(pose_t c, double u) {
      return new ceres::AutoDiffCostFunction<
        PoseMatchingErrorTerm, 6, 3, 4, 3, 4, 3, 4, 3, 4>(
        new PoseMatchingErrorTerm(c, u));
    }

  private:
    pose_t _pose;
    double _u;
  };
}  // namespace inrol
