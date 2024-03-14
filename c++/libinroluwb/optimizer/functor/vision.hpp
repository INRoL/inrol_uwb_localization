#pragma once

#include "libinroluwb/geometry/camera.hpp"
#include "libinroluwb/geometry/pose.hpp"
#include "libinroluwb/geometry/measurements.hpp"

#include <ceres/ceres.h>

namespace inrol {
  using std::shared_ptr;

  class VisionResidualTerm {
  public:
    VisionResidualTerm(
      const vision_data_t data, const double sigma, const double u,
      const shared_ptr<Camera> camera)
        : _data(data), _sigma(sigma), _u(u), _camera(camera) {
    }

    template <typename T>
    bool operator()(
      const T* const p0_ptr, const T* q0_ptr, const T* const p1_ptr,
      const T* q1_ptr, const T* const p2_ptr, const T* q2_ptr,
      const T* const p3_ptr, const T* q3_ptr, const T* const l_ptr,
      T* residuals_ptr) const {
      Eigen::Map<const Eigen::Matrix<T, 3, 1>> p0(p0_ptr);
      Eigen::Map<const Eigen::Quaternion<T>> q0(q0_ptr);
      Eigen::Map<const Eigen::Matrix<T, 3, 1>> p1(p1_ptr);
      Eigen::Map<const Eigen::Quaternion<T>> q1(q1_ptr);
      Eigen::Map<const Eigen::Matrix<T, 3, 1>> p2(p2_ptr);
      Eigen::Map<const Eigen::Quaternion<T>> q2(q2_ptr);
      Eigen::Map<const Eigen::Matrix<T, 3, 1>> p3(p3_ptr);
      Eigen::Map<const Eigen::Quaternion<T>> q3(q3_ptr);
      Eigen::Map<const Eigen::Matrix<T, 3, 1>> l(l_ptr);
      Eigen::Map<Eigen::Matrix<T, 2, 1>> residuals(residuals_ptr);

      Eigen::Quaternion<T> q = quaternion_b_spline(q0, q1, q2, q3, _u);
      Eigen::Matrix<T, 3, 1> p = position_b_spline(p0, p1, p2, p3, _u);

      Eigen::Matrix<T, 3, 1> l_B = q.toRotationMatrix().transpose() * (l - p);
      Eigen::Matrix<T, 2, 1> projected_uv = _camera->project(l_B);
      Eigen::Matrix<T, 2, 1> measured_uv = _data.uv.template cast<T>();
      residuals = (projected_uv - measured_uv) / T(_sigma);
      return true;
    }

    static ceres::CostFunction* Create(
      const vision_data_t data, const double sigma, const double u,
      const shared_ptr<Camera> camera) {
      return new ceres::AutoDiffCostFunction<
        VisionResidualTerm, 2, 3, 4, 3, 4, 3, 4, 3, 4, 3>(
        new VisionResidualTerm(data, sigma, u, camera));
    }

  private:
    const vision_data_t _data;
    const double _sigma;
    const double _u;
    const shared_ptr<Camera> _camera;
  };
}  // namespace inrol
