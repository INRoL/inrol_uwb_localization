#pragma once

#include "libinroluwb/geometry/pose.hpp"

#include <Eigen/Dense>

namespace inrol {
  using Eigen::Matrix3d;
  using Eigen::Vector3d;
  using Matrix9d = Eigen::Matrix<double, 9, 9>;
  using Vector9d = Eigen::Matrix<double, 9, 1>;
  using Matrix5d = Eigen::Matrix<double, 5, 5>;
  using Matrix15d = Eigen::Matrix<double, 15, 15>;
  using Vector15d = Eigen::Matrix<double, 15, 1>;

  Matrix3d get_Gamma0(Vector3d phi);
  Matrix3d get_Gamma1(Vector3d phi);
  Matrix3d get_Gamma2(Vector3d phi);
  Matrix3d get_Sigmma0(Vector3d w, double dt, Matrix3d Thth);
  Matrix3d get_Sigmma1(Vector3d w, double dt, Matrix3d Thth);
  Matrix3d get_Sigmma2(Vector3d w, double dt, Matrix3d Thth);
  Matrix3d get_Sigmma3(Vector3d w, double dt, Matrix3d Thth);

  Matrix3d hat(Vector3d phi);
  Matrix5d hat2(Vector9d phi);

  double get_psi_huber(double e, double sigma);
  double get_psi_asymm(double e, double sigma, double gamma);

  Vector3d get_init_velocity_from_traj(const vector<posestamped_t>& traj);
}  // namespace inrol
