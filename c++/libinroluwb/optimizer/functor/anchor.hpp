#pragma once

#include "libinroluwb/geometry/pose.hpp"

#include <ceres/ceres.h>

namespace inrol {
  class AnchorPriorErrorTerm {
  public:
    AnchorPriorErrorTerm(const Quaterniond q_prior, const double sigma)
        : _q_prior(q_prior), _sigma(sigma) {
    }

    template <typename T>
    bool operator()(const T* const q_ptr, T* residuals_ptr) const {
      Eigen::Map<Eigen::Matrix<T, 3, 1>> residuals(residuals_ptr);

      Eigen::Map<const Eigen::Quaternion<T>> q(q_ptr);
      Eigen::Quaternion<T> delta_q =
        _q_prior.template cast<T>() * q.conjugate();
      Eigen::Matrix<T, 3, 3> R = Eigen::Matrix<T, 3, 3>::Identity();
      R(0, 0) = T(1.0 / _sigma);
      R(1, 1) = T(1.0 / _sigma);
      R(2, 2) = T(1.0 / _sigma);
      residuals = R * quaterniond_logarithm(delta_q);
      return true;
    }

    static ceres::CostFunction* Create(
      const Quaterniond q_prior, const double s) {
      return new ceres::AutoDiffCostFunction<AnchorPriorErrorTerm, 3, 4>(
        new AnchorPriorErrorTerm(q_prior, s));
    }

  private:
    const Quaterniond _q_prior;
    const double _sigma;
  };
}  // namespace inrol
