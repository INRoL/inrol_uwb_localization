#pragma once

#include <eigen3/Eigen/Dense>
#include <ceres/ceres.h>

namespace inrol {
  template <typename T>
  Eigen::Quaternion<T> quaterniond_exponential(Eigen::Matrix<T, 3, 1> v) {
    T phi = ceres::sqrt(v(0) * v(0) + v(1) * v(1) + v(2) * v(2));
    if (phi == T(0)) {
      return Eigen::Quaternion<T>(T(1.0), T(0.0), T(0.0), T(0.0));
    }
    T sine = ceres::sin(phi * T(0.5));
    T x = sine * v(0) / phi;
    T y = sine * v(1) / phi;
    T z = sine * v(2) / phi;
    T w = ceres::cos(phi * T(0.5));
    return Eigen::Quaternion<T>(w, x, y, z);
  }

  template <typename T>
  Eigen::Matrix<T, 3, 1> quaterniond_logarithm(Eigen::Quaternion<T> q) {
    if (q.w() < 0) {
      q = Eigen::Quaternion<T>(-q.w(), -q.x(), -q.y(), -q.z());
    }
    T n = ceres::sqrt(q.x() * q.x() + q.y() * q.y() + q.z() * q.z());
    if (ceres::abs(n) < 1.0e-4) {
      T a = T(2.0) / q.w();
      return Eigen::Matrix<T, 3, 1>(q.x() * a, q.y() * a, q.z() * a);
    }
    T phi = T(2.0) * ceres::atan2(n, q.w());
    return Eigen::Matrix<T, 3, 1>(
      q.x() * phi / n, q.y() * phi / n, q.z() * phi / n);
  }

  template <typename T>
  Eigen::Quaternion<T> quaternion_b_spline(
    Eigen::Map<const Eigen::Quaternion<T>> q0,
    Eigen::Map<const Eigen::Quaternion<T>> q1,
    Eigen::Map<const Eigen::Quaternion<T>> q2,
    Eigen::Map<const Eigen::Quaternion<T>> q3, const double u) {
    double u2 = u * u;
    double u3 = u2 * u;
    double b1 = 5.0 / 6.0 + u / 2.0 - u2 / 2.0 + u3 / 6.0;
    double b2 = 1.0 / 6.0 + u / 2.0 + u2 / 2.0 - u3 / 3.0;
    double b3 = u3 / 6.0;
    Eigen::Matrix<T, 3, 1> qd01 =
      quaterniond_logarithm(q0.conjugate() * q1) * T(b1);
    Eigen::Matrix<T, 3, 1> qd12 =
      quaterniond_logarithm(q1.conjugate() * q2) * T(b2);
    Eigen::Matrix<T, 3, 1> qd23 =
      quaterniond_logarithm(q2.conjugate() * q3) * T(b3);
    Eigen::Quaternion<T> A1 = quaterniond_exponential(qd01);
    Eigen::Quaternion<T> A2 = quaterniond_exponential(qd12);
    Eigen::Quaternion<T> A3 = quaterniond_exponential(qd23);
    return q0 * A1 * A2 * A3;
  }

  template <typename T>
  Eigen::Matrix<T, 3, 1> position_b_spline(
    Eigen::Map<const Eigen::Matrix<T, 3, 1>> p0,
    Eigen::Map<const Eigen::Matrix<T, 3, 1>> p1,
    Eigen::Map<const Eigen::Matrix<T, 3, 1>> p2,
    Eigen::Map<const Eigen::Matrix<T, 3, 1>> p3, const double u) {
    double u2 = u * u;
    double u3 = u2 * u;
    double b1 = 5.0 / 6.0 + u / 2.0 - u2 / 2.0 + u3 / 6.0;
    double b2 = 1.0 / 6.0 + u / 2.0 + u2 / 2.0 - u3 / 3.0;
    double b3 = u3 / 6.0;
    Eigen::Matrix<T, 3, 1> pd01 = (p1 - p0) * T(b1);
    Eigen::Matrix<T, 3, 1> pd12 = (p2 - p1) * T(b2);
    Eigen::Matrix<T, 3, 1> pd23 = (p3 - p2) * T(b3);
    return p0 + pd01 + pd12 + pd23;
  }

  template <typename T>
  Eigen::Matrix<T, 3, 1> velocity_b_spline(
    Eigen::Map<const Eigen::Matrix<T, 3, 1>> p0,
    Eigen::Map<const Eigen::Matrix<T, 3, 1>> p1,
    Eigen::Map<const Eigen::Matrix<T, 3, 1>> p2,
    Eigen::Map<const Eigen::Matrix<T, 3, 1>> p3, const double u) {
    double u2 = u * u;
    double db1 = 1 / 2.0 - u + u2 / 2.0;
    double db2 = 1 / 2.0 + u - u2;
    double db3 = u2 / 2.0;
    Eigen::Matrix<T, 3, 1> vd01 = (p1 - p0) * T(db1);
    Eigen::Matrix<T, 3, 1> vd12 = (p2 - p1) * T(db2);
    Eigen::Matrix<T, 3, 1> vd23 = (p3 - p2) * T(db3);
    return vd01 + vd12 + vd23;
  }

  template <typename T>
  Eigen::Quaternion<T> quaternion_b_spline_(
    Eigen::Quaternion<T> q0, Eigen::Quaternion<T> q1, Eigen::Quaternion<T> q2,
    Eigen::Quaternion<T> q3, const double u) {
    double u2 = u * u;
    double u3 = u2 * u;
    double b1 = 5.0 / 6.0 + u / 2.0 - u2 / 2.0 + u3 / 6.0;
    double b2 = 1.0 / 6.0 + u / 2.0 + u2 / 2.0 - u3 / 3.0;
    double b3 = u3 / 6.0;
    Eigen::Matrix<T, 3, 1> qd01 =
      quaterniond_logarithm(q0.conjugate() * q1) * T(b1);
    Eigen::Matrix<T, 3, 1> qd12 =
      quaterniond_logarithm(q1.conjugate() * q2) * T(b2);
    Eigen::Matrix<T, 3, 1> qd23 =
      quaterniond_logarithm(q2.conjugate() * q3) * T(b3);
    Eigen::Quaternion<T> A1 = quaterniond_exponential(qd01);
    Eigen::Quaternion<T> A2 = quaterniond_exponential(qd12);
    Eigen::Quaternion<T> A3 = quaterniond_exponential(qd23);
    return q0 * A1 * A2 * A3;
  }

  template <typename T>
  Eigen::Matrix<T, 3, 1> position_b_spline_(
    Eigen::Matrix<T, 3, 1> p0, Eigen::Matrix<T, 3, 1> p1,
    Eigen::Matrix<T, 3, 1> p2, Eigen::Matrix<T, 3, 1> p3, const double u) {
    double u2 = u * u;
    double u3 = u2 * u;
    double b1 = 5.0 / 6.0 + u / 2.0 - u2 / 2.0 + u3 / 6.0;
    double b2 = 1.0 / 6.0 + u / 2.0 + u2 / 2.0 - u3 / 3.0;
    double b3 = u3 / 6.0;
    Eigen::Matrix<T, 3, 1> pd01 = (p1 - p0) * T(b1);
    Eigen::Matrix<T, 3, 1> pd12 = (p2 - p1) * T(b2);
    Eigen::Matrix<T, 3, 1> pd23 = (p3 - p2) * T(b3);
    return p0 + pd01 + pd12 + pd23;
  }

  template <typename T>
  Eigen::Matrix<T, 3, 1> acceleration_b_spline_(
    Eigen::Matrix<T, 3, 1> p0, Eigen::Matrix<T, 3, 1> p1,
    Eigen::Matrix<T, 3, 1> p2, Eigen::Matrix<T, 3, 1> p3, const double u,
    double dt) {
    double ddb1 = -1 + u;
    double ddb2 = 1 - 2 * u;
    double ddb3 = u;
    Eigen::Matrix<T, 3, 1> ad01 = (p1 - p0) * T(ddb1) / T(dt * dt);
    Eigen::Matrix<T, 3, 1> ad12 = (p2 - p1) * T(ddb2) / T(dt * dt);
    Eigen::Matrix<T, 3, 1> ad23 = (p3 - p2) * T(ddb3) / T(dt * dt);
    return ad01 + ad12 + ad23;
  }

  template <typename T>
  Eigen::Matrix<T, 3, 1> angular_velocity_b_spline_(
    Eigen::Quaternion<T> q0, Eigen::Quaternion<T> q1, Eigen::Quaternion<T> q2,
    Eigen::Quaternion<T> q3, const double u, double dt) {
    double u2 = u * u;
    double u3 = u2 * u;
    double b1 = 5.0 / 6.0 + u / 2.0 - u2 / 2.0 + u3 / 6.0;
    double b2 = 1.0 / 6.0 + u / 2.0 + u2 / 2.0 - u3 / 3.0;
    double b3 = u3 / 6.0;
    double db1 = 1 / 2.0 - u + u2 / 2.0;
    double db2 = 1 / 2.0 + u - u2;
    double db3 = u2 / 2.0;
    Eigen::Matrix<T, 3, 1> qd01 =
      quaterniond_logarithm(q0.conjugate() * q1) * T(b1);
    Eigen::Matrix<T, 3, 1> qd12 =
      quaterniond_logarithm(q1.conjugate() * q2) * T(b2);
    Eigen::Matrix<T, 3, 1> qd23 =
      quaterniond_logarithm(q2.conjugate() * q3) * T(b3);
    Eigen::Quaternion<T> qqd01(T(0), qd01.x(), qd01.y(), qd01.z());
    Eigen::Quaternion<T> qqd12(T(0), qd12.x(), qd12.y(), qd12.z());
    Eigen::Quaternion<T> qqd23(T(0), qd23.x(), qd23.y(), qd23.z());
    Eigen::Quaternion<T> A1 = quaterniond_exponential(qd01);
    Eigen::Quaternion<T> A2 = quaterniond_exponential(qd12);
    Eigen::Quaternion<T> A3 = quaterniond_exponential(qd23);
    Eigen::Quaternion<T> q = q0 * A1 * A2 * A3;

    Eigen::Quaternion<T> dA1 = A1 * qqd01;
    Eigen::Quaternion<T> dA2 = A2 * qqd12;
    Eigen::Quaternion<T> dA3 = A3 * qqd23;
    Eigen::Matrix<T, 4, 1> Ad123 = T(db1) * (dA1 * A2 * A3).coeffs() / T(dt);
    Eigen::Matrix<T, 4, 1> A1d23 = T(db2) * (A1 * dA2 * A3).coeffs() / T(dt);
    Eigen::Matrix<T, 4, 1> A12d3 = T(db3) * (A1 * A2 * dA3).coeffs() / T(dt);
    Eigen::Quaternion<T> dA((Ad123 + A1d23 + A12d3));
    Eigen::Quaternion<T> dq = q0 * dA;

    Eigen::Quaternion<T> w = q.conjugate() * dq;
    return Eigen::Matrix<T, 3, 1>(T(2.) * w.x(), T(2.) * w.y(), T(2.) * w.z());
  }
}  // namespace inrol
