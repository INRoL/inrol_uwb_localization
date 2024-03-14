#pragma once

#include "libinroluwb/geometry/measurements.hpp"
#include "libinroluwb/geometry/pose.hpp"
#include "libinroluwb/optimizer/math/harmonics.hpp"
#include "libinroluwb/optimizer/math/spline.hpp"
#include "libinroluwb/utils/binary.hpp"
#include "libinroluwb/optimizer/calibration.hpp"
#include "libinroluwb/optimizer/batch.hpp"

#include <ceres/ceres.h>
#include <cmath>

namespace inrol {
  class ModelACaliCost {
  public:
    ModelACaliCost(vector<cali_data_t> dataset): _dataset(dataset) {
    }

    template <typename T>
    bool operator()(const T* const x_ptr, T* cost_ptr) const {
      T s = x_ptr[0];

      T cost = T(0.0);
      for (auto const& data : _dataset) {
        Vector3d tag_position = data.tag_pose.p;
        Vector3d anchor_position = data.anchor_pose.p;
        Vector3d ray_world_coord = anchor_position - tag_position;
        double r = ray_world_coord.norm();
        T error = T(data.uwb.distance - r);
        cost = cost + ceres::log(s);
        cost = cost + error * error / (T(2) * s * s);
      }

      cost_ptr[0] = cost;
      return true;
    }

    static ceres::FirstOrderFunction* Create(vector<cali_data_t> dataset) {
      return new ceres::AutoDiffFirstOrderFunction<ModelACaliCost, 2>(
        new ModelACaliCost(dataset));
    }

  private:
    vector<cali_data_t> _dataset;
  };

  class ModelBCaliCost {
  public:
    ModelBCaliCost(vector<cali_data_t> dataset, int degree)
        : _dataset(dataset), _degree(degree) {
      _degree_container.push_back(0);
      _order_container.push_back(0);
      for (int i = 1; i < (_degree + 1) * (_degree + 1); i++) {
        int l = index2degree(i);
        int m = index2order(i, l);
        _degree_container.push_back(l);
        _order_container.push_back(m);
      }
    }

    template <typename T>
    bool operator()(const T* const x_ptr, T* cost_ptr) const {
      T s = x_ptr[0];
      T tag_coeffs[100] = {T(0.0)};
      T anchor_coeffs[100] = {T(0.0)};
      tag_coeffs[0] = x_ptr[1];
      for (int i = 1; i < (_degree + 1) * (_degree + 1); i++) {
        tag_coeffs[i] = x_ptr[i + 1];
        anchor_coeffs[i] = x_ptr[i + (_degree + 1) * (_degree + 1)];
      }

      T cost = T(0.0);
      for (auto const& data : _dataset) {
        Vector3d tag_position = data.tag_pose.p;
        Matrix3d tag_rotation_matrix = data.tag_pose.q.toRotationMatrix();
        Vector3d anchor_position = data.anchor_pose.p;
        Matrix3d anchor_rotation_matrix = data.anchor_pose.q.toRotationMatrix();
        Vector3d ray_world_coord = anchor_position - tag_position;
        Vector3d ray_tag_coord =
          tag_rotation_matrix.transpose() * ray_world_coord;
        Vector3d ray_anchor_coord =
          -anchor_rotation_matrix.transpose() * ray_world_coord;

        double r = ray_world_coord.norm();

        double tag_theta = std::acos(ray_tag_coord(2) / r);
        double tag_phi = std::atan2(ray_tag_coord(1), ray_tag_coord(0));

        double anchor_theta = std::acos(ray_anchor_coord(2) / r);
        double anchor_phi =
          std::atan2(ray_anchor_coord(1), ray_anchor_coord(0));

        T bias =
          tag_coeffs[0] * T(real_spherical_harmonics(0, 0, tag_theta, tag_phi));
        for (int i = 1; i < (_degree + 1) * (_degree + 1); i++) {
          int l = _degree_container[i];
          int m = _order_container[i];

          double tag_value = real_spherical_harmonics(l, m, tag_theta, tag_phi);
          T tc = tag_coeffs[i];
          bias += tc * T(tag_value);

          double anchor_value =
            real_spherical_harmonics(l, m, anchor_theta, anchor_phi);
          T ac = anchor_coeffs[i];
          bias += ac * T(anchor_value);
        }
        T error = T(data.uwb.distance - r) - bias;

        cost = cost + ceres::log(s);
        cost = cost + (error * error) / (T(2) * s * s);
      }

      cost_ptr[0] = cost;
      return true;
    }

    static ceres::FirstOrderFunction* Create(
      vector<cali_data_t> dataset, int degree) {
      return new ceres::AutoDiffFirstOrderFunction<ModelBCaliCost, 201>(
        new ModelBCaliCost(dataset, degree));
    }

  private:
    vector<cali_data_t> _dataset;
    int _degree;

    vector<int> _degree_container;
    vector<int> _order_container;
  };

  class ModelCCaliCost {
  public:
    ModelCCaliCost(vector<cali_data_t> dataset, int degree)
        : _dataset(dataset), _degree(degree) {
      _degree_container.push_back(0);
      _order_container.push_back(0);
      for (int i = 1; i < (_degree + 1) * (_degree + 1); i++) {
        int l = index2degree(i);
        int m = index2order(i, l);
        _degree_container.push_back(l);
        _order_container.push_back(m);
      }
    }

    template <typename T>
    bool operator()(const T* const x_ptr, T* cost_ptr) const {
      T s = x_ptr[0];
      T g = x_ptr[1];
      T tag_coeffs[100] = {T(0.0)};
      T anchor_coeffs[100] = {T(0.0)};
      tag_coeffs[0] = x_ptr[2];
      for (int i = 1; i < (_degree + 1) * (_degree + 1); i++) {
        tag_coeffs[i] = x_ptr[i + 2];
        anchor_coeffs[i] = x_ptr[i + (_degree + 1) * (_degree + 1) + 1];
      }

      T cost = T(0.0);
      for (auto const& data : _dataset) {
        Vector3d tag_position = data.tag_pose.p;
        Matrix3d tag_rotation_matrix = data.tag_pose.q.toRotationMatrix();
        Vector3d anchor_position = data.anchor_pose.p;
        Matrix3d anchor_rotation_matrix = data.anchor_pose.q.toRotationMatrix();
        Vector3d ray_world_coord = anchor_position - tag_position;
        Vector3d ray_tag_coord =
          tag_rotation_matrix.transpose() * ray_world_coord;
        Vector3d ray_anchor_coord =
          -anchor_rotation_matrix.transpose() * ray_world_coord;

        double r = ray_world_coord.norm();

        double tag_theta = std::acos(ray_tag_coord(2) / r);
        double tag_phi = std::atan2(ray_tag_coord(1), ray_tag_coord(0));

        double anchor_theta = std::acos(ray_anchor_coord(2) / r);
        double anchor_phi =
          std::atan2(ray_anchor_coord(1), ray_anchor_coord(0));
        T bias =
          tag_coeffs[0] * T(real_spherical_harmonics(0, 0, tag_theta, tag_phi));
        for (int i = 1; i < (_degree + 1) * (_degree + 1); i++) {
          int l = _degree_container[i];
          int m = _order_container[i];

          double tag_value = real_spherical_harmonics(l, m, tag_theta, tag_phi);
          T tc = tag_coeffs[i];
          bias += tc * T(tag_value);

          double anchor_value =
            real_spherical_harmonics(l, m, anchor_theta, anchor_phi);
          T ac = anchor_coeffs[i];
          bias += ac * T(anchor_value);
        }
        T error = T(data.uwb.distance - r) - bias;

        T alpha =
          T(2.0 * M_PI) * g / (T(std::sqrt(2.0 * M_PI)) * s + T(M_PI) * g);
        if (error < 0) {
          cost = cost - ceres::log(T(2.0) - alpha);
          cost = cost + T(0.5) * ceres::log(T(2.0 * M_PI));
          cost = cost + ceres::log(s);
          cost = cost + error * error / (T(2.0) * s * s);
        } else {
          cost = cost - ceres::log(alpha);
          cost = cost + ceres::log(T(M_PI));
          cost = cost + ceres::log(g);
          cost = cost + ceres::log(T(1.0) + error * error / (g * g));
        }
      }

      cost_ptr[0] = cost;
      return true;
    }

    static ceres::FirstOrderFunction* Create(
      vector<cali_data_t> dataset, int degree) {
      return new ceres::AutoDiffFirstOrderFunction<ModelCCaliCost, 202>(
        new ModelCCaliCost(dataset, degree));
    }

  private:
    vector<cali_data_t> _dataset;
    int _degree;

    vector<int> _degree_container;
    vector<int> _order_container;
  };

  class ScaleUWBErrorTerm {
  public:
    ScaleUWBErrorTerm(uwb_data_t uwb, pose_t tag_pose)
        : _uwb(std::move(uwb)), _tag_pose(tag_pose) {
    }

    template <typename T>
    bool operator()(
      const T* const s_ptr, const T* const a_ptr, T* residuals_ptr) const {
      T s(*s_ptr);
      Eigen::Map<const Eigen::Matrix<T, 3, 1>> a(a_ptr);

      Eigen::Matrix<T, 3, 1> tag_position = _tag_pose.p.template cast<T>();
      Eigen::Matrix<T, 3, 1> ray_world_coord = a - tag_position * s;

      T r = ray_world_coord.norm();

      residuals_ptr[0] = T(_uwb.distance) - r;
      return true;
    }

    static ceres::CostFunction* Create(uwb_data_t uwb, pose_t tag_pose) {
      return new ceres::AutoDiffCostFunction<ScaleUWBErrorTerm, 1, 1, 3>(
        new ScaleUWBErrorTerm(uwb, tag_pose));
    }

  private:
    uwb_data_t _uwb;
    pose_t _tag_pose;
  };

  class PoseUWBErrorTerm {
  public:
    PoseUWBErrorTerm(
      const uwb_data_t uwb, const int degree, const double u,
      const pose_t T_cam_tag, const vector<double>& tag_coeffs,
      const vector<double>& anchor_coeffs, const ModelType type,
      const uncertainty_param_t uncertainty_param)
        : _uwb(std::move(uwb)),
          _degree(degree),
          _u(u),
          _T_cam_tag(T_cam_tag),
          _tag_coeffs(tag_coeffs),
          _anchor_coeffs(anchor_coeffs),
          _type(type),
          _uncertainty_param(uncertainty_param) {
      _degree_container.push_back(0);
      _order_container.push_back(0);
      for (int i = 1; i < (_degree + 1) * (_degree + 1); i++) {
        int l = index2degree(i);
        int m = index2order(i, l);
        _degree_container.push_back(l);
        _order_container.push_back(m);
      }
    }

    template <typename T>
    bool operator()(
      const T* const p0_ptr, const T* const q0_ptr, const T* const p1_ptr,
      const T* const q1_ptr, const T* const p2_ptr, const T* const q2_ptr,
      const T* const p3_ptr, const T* const q3_ptr, const T* const ap_ptr,
      const T* const aq_ptr, T* residuals_ptr) const {
      Eigen::Map<const Eigen::Matrix<T, 3, 1>> cam_p0(p0_ptr);
      Eigen::Map<const Eigen::Quaternion<T>> cam_q0(q0_ptr);
      Eigen::Map<const Eigen::Matrix<T, 3, 1>> cam_p1(p1_ptr);
      Eigen::Map<const Eigen::Quaternion<T>> cam_q1(q1_ptr);
      Eigen::Map<const Eigen::Matrix<T, 3, 1>> cam_p2(p2_ptr);
      Eigen::Map<const Eigen::Quaternion<T>> cam_q2(q2_ptr);
      Eigen::Map<const Eigen::Matrix<T, 3, 1>> cam_p3(p3_ptr);
      Eigen::Map<const Eigen::Quaternion<T>> cam_q3(q3_ptr);
      Eigen::Map<const Eigen::Matrix<T, 3, 1>> ap(ap_ptr);
      Eigen::Map<const Eigen::Quaternion<T>> aq(aq_ptr);

      Eigen::Quaternion<T> cam_q =
        quaternion_b_spline(cam_q0, cam_q1, cam_q2, cam_q3, _u);
      Eigen::Matrix<T, 3, 1> cam_p =
        position_b_spline(cam_p0, cam_p1, cam_p2, cam_p3, _u);

      Eigen::Matrix<T, 3, 1> p_cam_tag = _T_cam_tag.p.template cast<T>();
      Eigen::Quaternion<T> q_cam_tag = _T_cam_tag.q.template cast<T>();

      Eigen::Quaternion<T> q = cam_q * q_cam_tag;
      Eigen::Matrix<T, 3, 1> p = cam_p + cam_q.toRotationMatrix() * p_cam_tag;

      Eigen::Matrix<T, 3, 3> tag_rotation_matrix = q.toRotationMatrix();
      Eigen::Matrix<T, 3, 3> anchor_rotation_matrix = aq.toRotationMatrix();

      Eigen::Matrix<T, 3, 1> ray_world_coord = ap - p;
      Eigen::Matrix<T, 3, 1> ray_tag_coord =
        tag_rotation_matrix.transpose() * ray_world_coord;
      Eigen::Matrix<T, 3, 1> ray_anchor_coord =
        -anchor_rotation_matrix.transpose() * ray_world_coord;
      T r = ray_world_coord.norm();

      T tag_theta = ceres::acos(ray_tag_coord(2) / r);
      T tag_phi = ceres::atan2(ray_tag_coord(1), ray_tag_coord(0));

      T anchor_theta = ceres::acos(ray_anchor_coord(2) / r);
      T anchor_phi = ceres::atan2(ray_anchor_coord(1), ray_anchor_coord(0));

      T bias =
        _tag_coeffs[0] * T(real_spherical_harmonics(0, 0, tag_theta, tag_phi));
      for (int i = 1; i < (_degree + 1) * (_degree + 1); i++) {
        int l = _degree_container[i];
        int m = _order_container[i];

        T tag_value = real_spherical_harmonics(l, m, tag_theta, tag_phi);
        T tag_coeffs = T(_tag_coeffs[i]);
        bias += tag_coeffs * tag_value;

        T anchor_value =
          real_spherical_harmonics(l, m, anchor_theta, anchor_phi);
        T anchor_coeffs = T(_anchor_coeffs[i]);
        bias += anchor_coeffs * anchor_value;
      }
      gaussian_param_t g;
      asymmetric_param_t a;
      T error;
      switch (_type) {
      case ModelType::A:
      case ModelType::AH:
        g = std::get<gaussian_param_t>(_uncertainty_param);
        error = T(_uwb.distance) - r;
        residuals_ptr[0] = error / T(g.sigma);
        break;
      case ModelType::B:
      case ModelType::BH:
        g = std::get<gaussian_param_t>(_uncertainty_param);
        error = T(_uwb.distance) - r - bias;
        residuals_ptr[0] = error / T(g.sigma);
        break;
      case ModelType::C:
        error = (T(_uwb.distance) - r - bias);
        a = std::get<asymmetric_param_t>(_uncertainty_param);
        if (error < 0) {
          residuals_ptr[0] = error / T(a.sigma);
        } else {
          residuals_ptr[0] = ceres::sqrt(
            T(2.0) * ceres::log(T(1.0) + error * error / T(a.gamma * a.gamma)));
        }
        break;
      default:
        break;
      }
      return true;
    }

    static ceres::CostFunction* Create(
      uwb_data_t uwb, int degree, double u, pose_t T_cam_tag,
      const vector<double>& tag_coeffs, const vector<double>& anchor_coeffs,
      ModelType type, uncertainty_param_t uncertainty_param) {
      return new ceres::AutoDiffCostFunction<
        PoseUWBErrorTerm, 1, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4>(new PoseUWBErrorTerm(
        uwb, degree, u, T_cam_tag, tag_coeffs, anchor_coeffs, type,
        uncertainty_param));
    }

  private:
    const uwb_data_t _uwb;
    const int _degree;
    const double _u;
    vector<int> _degree_container;
    vector<int> _order_container;
    const pose_t _T_cam_tag;
    const vector<double> _tag_coeffs;
    const vector<double> _anchor_coeffs;
    const ModelType _type;
    const uncertainty_param_t _uncertainty_param;
  };
}  // namespace inrol
