#pragma once

#include "libinroluwb/geometry/pose.hpp"

#include <Eigen/Dense>

namespace inrol {
  class Camera {
  public:
    Camera(
      uint64_t w, uint64_t h, double fx, double fy, double cx, double cy,
      vector<double> dist_coeffs)
        : _w(w),
          _h(h),
          _fx(fx),
          _fy(fy),
          _cx(cx),
          _cy(cy),
          _dist_coeffs(dist_coeffs) {};

    template <typename T>
    Eigen::Matrix<T, 2, 1> distortRadTan(const Eigen::Matrix<T, 2, 1>& p) {
      Eigen::Matrix<T, 2, 1> result;

      T mx2_u = p[0] * p[0];
      T my2_u = p[1] * p[1];
      T mxy_u = p[0] * p[1];
      T rho2_u = mx2_u + my2_u;
      T rad_dist_u =
        _dist_coeffs[0] * rho2_u + _dist_coeffs[1] * rho2_u * rho2_u;

      result[0] = p[0] + p[0] * rad_dist_u + 2.0 * _dist_coeffs[2] * mxy_u +
        _dist_coeffs[3] * (rho2_u + 2.0 * mx2_u);
      result[1] = p[1] + p[1] * rad_dist_u + 2.0 * _dist_coeffs[3] * mxy_u +
        _dist_coeffs[2] * (rho2_u + 2.0 * my2_u);

      return result;
    }

    template <typename T>
    Eigen::Matrix<T, 2, 1> project(const Eigen::Matrix<T, 3, 1>& p_c) {
      Eigen::Matrix<T, 2, 1> result;
      result[0] = p_c[0] / p_c[2];
      result[1] = p_c[1] / p_c[2];
      result = distortRadTan(result);

      result[0] = _fx * result[0] + _cx;
      result[1] = _fy * result[1] + _cy;
      return result;
    }

  private:
    uint64_t _w, _h;
    double _fx, _fy;
    double _cx, _cy;
    vector<double> _dist_coeffs;
  };
}  // namespace inrol
