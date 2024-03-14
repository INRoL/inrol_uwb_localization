#pragma once

#include <Eigen/Dense>

namespace inrol {
  using Eigen::Matrix3d;
  using Matrix9d = Eigen::Matrix<double, 9, 9>;
  using Eigen::Quaterniond;
  using Eigen::Vector3d;

  Matrix9d J_of_AB_wrt_A(Matrix3d B);
  Matrix9d J_of_AB_wrt_B(Matrix3d A);
  Eigen::Matrix<double, 3, 9> J_of_Ac_wrt_A(Vector3d c);
  Matrix9d J_of_transpose(Matrix3d A);
  Eigen::Matrix<double, 1, 3> J_of_norm(Vector3d r);
  Eigen::Matrix<double, 9, 4> J_of_R_wrt_q(Quaterniond q);
  Eigen::Matrix<double, 4, 3> J_of_q_wrt_dth(Quaterniond q);
  Eigen::Matrix<double, 1, 3> J_of_normalized_z(Vector3d r);
  Eigen::Matrix<double, 3, 4> J_of_rotation_action_wrt_q(
    Quaterniond q, Vector3d a);
  Eigen::Matrix<double, 1, 3> derivative_elevation(Vector3d r);
  Eigen::Matrix<double, 1, 3> derivative_azimuth(Vector3d r);
  Eigen::Matrix<double, 1, 2> J_of_basis_wrt_spherical_coord(
    int l, int m, double theta, double phi);
}  // namespace inrol
