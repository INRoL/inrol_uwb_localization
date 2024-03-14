#pragma once

#include "libinroluwb/geometry/measurements.hpp"
#include "libinroluwb/geometry/pose.hpp"

#include <ceres/ceres.h>

namespace inrol {
  class AccelerometerResidualTerm {
  public:
    AccelerometerResidualTerm(
      const imu_data_t data, const double sigma, const double u,
      const double bias_u, const double dt, const pose_t T_cam_imu)
        : _data(std::move(data)),
          _sigma(sigma),
          _u(u),
          _bias_u(bias_u),
          _dt(dt),
          _T_cam_imu(T_cam_imu) {
    }

    template <typename T>
    bool operator()(
      const T* const p0_ptr, const T* q0_ptr, const T* const p1_ptr,
      const T* q1_ptr, const T* const p2_ptr, const T* q2_ptr,
      const T* const p3_ptr, const T* q3_ptr, const T* const b0_ptr,
      const T* const b1_ptr, const T* const b2_ptr, const T* b3_ptr,
      const T* g_ptr, T* residuals_ptr) const {
      Eigen::Map<const Eigen::Matrix<T, 3, 1>> cam_p0(p0_ptr);
      Eigen::Map<const Eigen::Quaternion<T>> cam_q0(q0_ptr);
      Eigen::Map<const Eigen::Matrix<T, 3, 1>> cam_p1(p1_ptr);
      Eigen::Map<const Eigen::Quaternion<T>> cam_q1(q1_ptr);
      Eigen::Map<const Eigen::Matrix<T, 3, 1>> cam_p2(p2_ptr);
      Eigen::Map<const Eigen::Quaternion<T>> cam_q2(q2_ptr);
      Eigen::Map<const Eigen::Matrix<T, 3, 1>> cam_p3(p3_ptr);
      Eigen::Map<const Eigen::Quaternion<T>> cam_q3(q3_ptr);
      Eigen::Map<const Eigen::Matrix<T, 3, 1>> g(g_ptr);

      Eigen::Matrix<T, 3, 1> p_cam_imu = _T_cam_imu.p.template cast<T>();
      Eigen::Quaternion<T> q_cam_imu = _T_cam_imu.q.template cast<T>();

      Eigen::Matrix<T, 3, 1> p0 =
        cam_p0 + cam_q0.toRotationMatrix() * p_cam_imu;
      Eigen::Matrix<T, 3, 1> p1 =
        cam_p1 + cam_q1.toRotationMatrix() * p_cam_imu;
      Eigen::Matrix<T, 3, 1> p2 =
        cam_p2 + cam_q2.toRotationMatrix() * p_cam_imu;
      Eigen::Matrix<T, 3, 1> p3 =
        cam_p3 + cam_q3.toRotationMatrix() * p_cam_imu;
      Eigen::Quaternion<T> q0 = cam_q0 * q_cam_imu;
      Eigen::Quaternion<T> q1 = cam_q1 * q_cam_imu;
      Eigen::Quaternion<T> q2 = cam_q2 * q_cam_imu;
      Eigen::Quaternion<T> q3 = cam_q3 * q_cam_imu;

      Eigen::Map<const Eigen::Matrix<T, 3, 1>> b0(b0_ptr);
      Eigen::Map<const Eigen::Matrix<T, 3, 1>> b1(b1_ptr);
      Eigen::Map<const Eigen::Matrix<T, 3, 1>> b2(b2_ptr);
      Eigen::Map<const Eigen::Matrix<T, 3, 1>> b3(b3_ptr);
      Eigen::Map<Eigen::Matrix<T, 3, 1>> residuals(residuals_ptr);

      Eigen::Quaternion<T> q = quaternion_b_spline_(q0, q1, q2, q3, _u);
      Eigen::Matrix<T, 3, 1> b = position_b_spline(b0, b1, b2, b3, _bias_u);
      Eigen::Matrix<T, 3, 1> a_W =
        acceleration_b_spline_(p0, p1, p2, p3, _u, _dt);
      const Eigen::Matrix<T, 3, 1> a_B =
        q.toRotationMatrix().transpose() * (a_W + g) - b;
      Eigen::Matrix<T, 3, 1> imu_a;
      imu_a << T(_data.ax), T(_data.ay), T(_data.az);
      residuals = (a_B - imu_a) / T(_sigma);
      return true;
    }

    static ceres::CostFunction* Create(
      const imu_data_t data, const double sigma, const double u,
      const double bias_u, const double dt, const pose_t T_cam_imu) {
      return new ceres::AutoDiffCostFunction<
        AccelerometerResidualTerm, 3, 3, 4, 3, 4, 3, 4, 3, 4, 3, 3, 3, 3, 3>(
        new AccelerometerResidualTerm(data, sigma, u, bias_u, dt, T_cam_imu));
    }

  private:
    const imu_data_t _data;
    const double _u;
    const double _sigma;
    const double _bias_u;
    const double _dt;
    const pose_t _T_cam_imu;
  };

  class GyroscopeResidualTerm {
  public:
    GyroscopeResidualTerm(
      const imu_data_t data, const double sigma, const double u,
      const double bias_u, const double dt, const pose_t T_cam_imu)
        : _data(std::move(data)),
          _sigma(sigma),
          _u(u),
          _dt(dt),
          _bias_u(bias_u),
          _T_cam_imu(T_cam_imu) {
    }

    template <typename T>
    bool operator()(
      const T* q0_ptr, const T* q1_ptr, const T* q2_ptr, const T* q3_ptr,
      const T* const b0_ptr, const T* const b1_ptr, const T* const b2_ptr,
      const T* b3_ptr, T* residuals_ptr) const {
      Eigen::Map<const Eigen::Quaternion<T>> cam_q0(q0_ptr);
      Eigen::Map<const Eigen::Quaternion<T>> cam_q1(q1_ptr);
      Eigen::Map<const Eigen::Quaternion<T>> cam_q2(q2_ptr);
      Eigen::Map<const Eigen::Quaternion<T>> cam_q3(q3_ptr);
      Eigen::Map<const Eigen::Matrix<T, 3, 1>> b0(b0_ptr);
      Eigen::Map<const Eigen::Matrix<T, 3, 1>> b1(b1_ptr);
      Eigen::Map<const Eigen::Matrix<T, 3, 1>> b2(b2_ptr);
      Eigen::Map<const Eigen::Matrix<T, 3, 1>> b3(b3_ptr);
      Eigen::Map<Eigen::Matrix<T, 3, 1>> residuals(residuals_ptr);
      Eigen::Quaternion<T> q_cam_imu = _T_cam_imu.q.template cast<T>();

      Eigen::Quaternion<T> q0 = cam_q0 * q_cam_imu;
      Eigen::Quaternion<T> q1 = cam_q1 * q_cam_imu;
      Eigen::Quaternion<T> q2 = cam_q2 * q_cam_imu;
      Eigen::Quaternion<T> q3 = cam_q3 * q_cam_imu;
      Eigen::Matrix<T, 3, 1> b = position_b_spline(b0, b1, b2, b3, _bias_u);

      Eigen::Matrix<T, 3, 1> w =
        angular_velocity_b_spline_(q0, q1, q2, q3, _u, _dt);
      Eigen::Matrix<T, 3, 1> imu_w;
      imu_w << T(_data.wx), T(_data.wy), T(_data.wz);
      residuals = (w - imu_w - b) / T(_sigma);
      return true;
    }

    static ceres::CostFunction* Create(
      imu_data_t i, double s, double u, double bias_u, double dt,
      pose_t T_cam_imu) {
      return new ceres::AutoDiffCostFunction<
        GyroscopeResidualTerm, 3, 4, 4, 4, 4, 3, 3, 3, 3>(
        new GyroscopeResidualTerm(i, s, u, bias_u, dt, T_cam_imu));
    }

  private:
    imu_data_t _data;
    double _u;
    double _sigma;
    double _bias_u;
    double _dt;
    pose_t _T_cam_imu;
  };

}  // namespace inrol
