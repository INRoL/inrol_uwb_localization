#pragma once

#include "libinroluwb/geometry/pose.hpp"

#include <ceres/ceres.h>

namespace inrol {
  class BiasResidualTerm {
  public:
    BiasResidualTerm(const double sigma, const double u, const double dt)
        : _sigma(sigma), _u(u), _dt(dt) {
    }

    template <typename T>
    bool operator()(
      const T* const b0_ptr, const T* b1_ptr, const T* const b2_ptr,
      const T* const b3_ptr, T* residuals_ptr) const {
      Eigen::Map<const Eigen::Matrix<T, 3, 1>> b0(b0_ptr);
      Eigen::Map<const Eigen::Matrix<T, 3, 1>> b1(b1_ptr);
      Eigen::Map<const Eigen::Matrix<T, 3, 1>> b2(b2_ptr);
      Eigen::Map<const Eigen::Matrix<T, 3, 1>> b3(b3_ptr);
      Eigen::Map<Eigen::Matrix<T, 3, 1>> residuals(residuals_ptr);

      Eigen::Matrix<T, 3, 1> db = velocity_b_spline(b0, b1, b2, b3, _u);
      residuals = db * T(std::sqrt(_dt)) / T(_sigma);
      return true;
    }

    static ceres::CostFunction* Create(
      const double sigma, const double u, const double dt) {
      return new ceres::AutoDiffCostFunction<BiasResidualTerm, 3, 3, 3, 3, 3>(
        new BiasResidualTerm(sigma, u, dt));
    }

  private:
    const double _sigma;
    const double _u;
    const double _dt;
  };
}  // namespace inrol
