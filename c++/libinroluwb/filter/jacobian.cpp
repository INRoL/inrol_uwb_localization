#include "libinroluwb/filter/jacobian.hpp"
#include "libinroluwb/filter/filter_math_utility.hpp"
#include "libinroluwb/optimizer/math/harmonics.hpp"

namespace inrol {
  using Eigen::Matrix3d;
  using Matrix9d = Eigen::Matrix<double, 9, 9>;
  using Eigen::Vector3d;

  Matrix9d J_of_AB_wrt_A(Matrix3d B) {
    Matrix9d result = Matrix9d::Zero();
    result.block(0, 0, 3, 3) = B.transpose();
    result.block(3, 3, 3, 3) = B.transpose();
    result.block(6, 6, 3, 3) = B.transpose();
    return result;
  }

  Matrix9d J_of_AB_wrt_B(Matrix3d A) {
    Matrix9d result = Matrix9d::Zero();
    Matrix3d I3 = Matrix3d::Identity();
    result.block(0, 0, 3, 3) = A(0, 0) * I3;
    result.block(0, 3, 3, 3) = A(0, 1) * I3;
    result.block(0, 6, 3, 3) = A(0, 2) * I3;
    result.block(3, 0, 3, 3) = A(1, 0) * I3;
    result.block(3, 3, 3, 3) = A(1, 1) * I3;
    result.block(3, 6, 3, 3) = A(1, 2) * I3;
    result.block(6, 0, 3, 3) = A(2, 0) * I3;
    result.block(6, 3, 3, 3) = A(2, 1) * I3;
    result.block(6, 6, 3, 3) = A(2, 2) * I3;
    return result;
  }

  Eigen::Matrix<double, 3, 9> J_of_Ac_wrt_A(Vector3d c) {
    Eigen::Matrix<double, 3, 9> result = Eigen::Matrix<double, 3, 9>::Zero();
    result.block(0, 0, 1, 3) = c.transpose();
    result.block(1, 3, 1, 3) = c.transpose();
    result.block(2, 6, 1, 3) = c.transpose();
    return result;
  }

  Matrix9d J_of_transpose(Matrix3d A) {
    Matrix9d result = Matrix9d::Zero();
    result(0, 0) = 1.0;
    result(1, 3) = 1.0;
    result(2, 6) = 1.0;
    result(3, 1) = 1.0;
    result(4, 4) = 1.0;
    result(5, 7) = 1.0;
    result(6, 2) = 1.0;
    result(7, 5) = 1.0;
    result(8, 8) = 1.0;
    return result;
  }

  Eigen::Matrix<double, 1, 3> J_of_norm(Vector3d r) {
    Eigen::Matrix<double, 1, 3> result = r.transpose() / r.norm();
    return result;
  }

  // clang-format off
  Eigen::Matrix<double, 9, 4> J_of_R_wrt_q(Quaterniond q) {
    Eigen::Matrix<double, 9, 4> result = Eigen::Matrix<double, 9, 4>::Zero();
    result << q.w(),  q.x(), -q.y(), -q.z(),
             -q.z(),  q.y(),  q.x(), -q.w(),
              q.y(),  q.z(),  q.w(),  q.x(),
              q.z(),  q.y(),  q.x(),  q.w(),
              q.w(), -q.x(),  q.y(), -q.z(),
             -q.x(), -q.w(),  q.z(),  q.y(),
             -q.y(),  q.z(), -q.w(),  q.x(),
              q.x(),  q.w(),  q.z(),  q.y(),
              q.w(), -q.x(), -q.y(),  q.z();
    result = 2 * result;
    return result;
  }

  Eigen::Matrix<double, 4, 3> J_of_q_wrt_dth(Quaterniond q) {
    Eigen::Matrix<double, 4, 3> result = Eigen::Matrix<double, 4, 3>::Zero();
    result << -q.x(), -q.y(), -q.z(),
               q.w(), -q.z(),  q.y(),
               q.z(),  q.w(), -q.x(),
              -q.y(),  q.x(),  q.w();
    result = 0.5 * result;
    return result;
  }

  Eigen::Matrix<double, 3, 4> J_of_rotation_action_wrt_q(
    Quaterniond q, Vector3d a) {
    Eigen::Matrix<double, 3, 4> result = Eigen::Matrix<double, 3, 4>::Zero();
    double w = q.w();
    Vector3d v = q.vec();
    Vector3d left_column = w * a + hat(v) * a;
    Matrix3d skew_a = hat(a);
    Matrix3d right_matrix = v.transpose() * a * Matrix3d::Identity() +
      v * a.transpose() - a * v.transpose() - w * skew_a;
    result << left_column , right_matrix;

    result = 2.0 * result;
    return result;
  }
  // clang-format on

  Eigen::Matrix<double, 1, 3> J_of_normalized_z(Vector3d r) {
    Eigen::Matrix<double, 1, 3> result;
    result << -r.x() * r.z(), -r.y() * r.z(), r.x() * r.x() + r.y() * r.y();
    result /= r.norm() * r.norm() * r.norm();
    return result;
  }

  Eigen::Matrix<double, 1, 3> derivative_elevation(Vector3d r) {
    Eigen::Matrix<double, 1, 3> result;
    double bar_z = r.z() / r.norm();
    double J_of_acos = -1.0 / std::sqrt(1.0 - bar_z * bar_z);
    result = J_of_normalized_z(r);

    return J_of_acos * result;
  }

  Eigen::Matrix<double, 1, 3> derivative_azimuth(Vector3d r) {
    Eigen::Matrix<double, 1, 3> result;
    result << -r.y(), r.x(), 0.0;
    result /= r.x() * r.x() + r.y() * r.y();
    return result;
  }

  Eigen::Matrix<double, 1, 2> J_of_basis_wrt_spherical_coord(
    int l, int m, double theta, double phi) {
    Eigen::Matrix<double, 1, 2> result;
    double J_of_basis_wrt_phi = dYdphi(l, m, theta, phi);
    double J_of_basis_wrt_theta = dYdtheta(l, m, theta, phi);

    result << J_of_basis_wrt_phi, J_of_basis_wrt_theta;
    return result;
  }
}  // namespace inrol
