#include "libinroluwb/filter/filter_math_utility.hpp"

#include <glog/logging.h>

#include <cmath>

namespace inrol {
  Matrix3d get_Gamma0(Vector3d phi) {
    double p = phi.norm();
    if (p < 1e-6)
      return Matrix3d::Identity();
    Vector3d u = phi / p;
    Matrix3d hat_u = hat(u);
    return Matrix3d::Identity() + std::sin(p) * hat_u +
      (1. - std::cos(p)) * hat_u * hat_u;
  }

  Matrix3d get_Gamma1(Vector3d phi) {
    double p = phi.norm();
    Vector3d u = phi / p;
    Matrix3d hat_u = hat(u);
    return Matrix3d::Identity() + (1. - std::cos(p)) / p * hat_u +
      (p - std::sin(p)) / p * hat_u * hat_u;
  }

  Matrix3d get_Gamma2(Vector3d phi) {
    double p = phi.norm();
    Vector3d u = phi / p;
    Matrix3d hat_u = hat(u);
    return Matrix3d::Identity() / 2. + (p - std::sin(p)) / (p * p) * hat_u +
      (p * p + 2 * std::cos(p) - 2.) / (2. * p * p) * hat_u * hat_u;
  }

  Matrix3d hat(Vector3d w) {
    Matrix3d result;
    result << 0, -w(2), w(1), w(2), 0, -w(0), -w(1), w(0), 0;
    return result;
  }

  Matrix5d hat2(Vector9d w) {
    Matrix5d result = Matrix5d::Zero();
    result.block(0, 0, 3, 3) = hat(w.segment(0, 3));
    result.block(0, 3, 3, 1) = w.segment(3, 3);
    result.block(0, 4, 3, 1) = w.segment(6, 3);
    return result;
  }

  Matrix3d get_Sigmma0(Vector3d w, double dt, Matrix3d Thth) {
    return get_Gamma0(w * dt).transpose();
  }

  Matrix3d get_Sigmma1(Vector3d w, double dt, Matrix3d Thth) {
    double nw = w.norm();
    if (nw < 1e-8) {
      return Matrix3d::Identity() * dt;
    }
    auto term1 = Matrix3d::Identity() * dt;
    auto term2 = -Thth / (nw * nw) *
      (get_Sigmma0(w, dt, Thth) - Matrix3d::Identity() - Thth * dt);
    return term1 + term2;
  }

  Matrix3d get_Sigmma2(Vector3d w, double dt, Matrix3d Thth) {
    double nw = w.norm();
    if (nw < 1e-8) {
      return 0.5 * Matrix3d::Identity() * dt * dt;
    }
    auto term1 = 0.5 * Matrix3d::Identity() * dt * dt;
    auto term2 = -1 / (nw * nw) *
      (get_Sigmma0(w, dt, Thth) - Matrix3d::Identity() - Thth * dt -
       0.5 * Thth * Thth * dt * dt);
    return term1 + term2;
  }

  Matrix3d get_Sigmma3(Vector3d w, double dt, Matrix3d Thth) {
    double nw = w.norm();
    if (nw < 1e-8) {
      return 1.0 / 6.0 * Matrix3d::Identity() * dt * dt * dt;
    }
    auto term1 = Matrix3d::Identity() / 6.0 * dt * dt * dt;
    auto term2 = Thth / (nw * nw * nw * nw) *
      (get_Sigmma0(w, dt, Thth) - Matrix3d::Identity() - Thth * dt -
       0.5 * Thth * Thth * dt * dt -
       1.0 / 6.0 * Thth * Thth * Thth * dt * dt * dt);
    return term1 + term2;
  }

  double get_psi_huber(double e, double sigma) {
    if (std::abs(e) < sigma) {
      return 1.0 / (sigma * sigma);
    } else {
      return 1.0 / (sigma * std::abs(e));
    }
  }

  double get_psi_asymm(double e, double sigma, double gamma) {
    if (e < 0) {
      return 1.0 / (sigma * sigma);
    } else {
      return 1.0 / (e * e + gamma * gamma);
    }
  }

  Vector3d get_init_velocity_from_traj(const vector<posestamped_t>& traj) {
    auto first_pose = traj[0].pose;
    auto second_pose = traj[1].pose;
    auto third_pose = traj[2].pose;
    auto fourth_pose = traj[3].pose;

    return 100.0 *
      (-11. / 6. * first_pose.p + 3. * second_pose.p - 3. / 2. * third_pose.p +
       1. / 3. * fourth_pose.p);
  }
}  // namespace inrol
