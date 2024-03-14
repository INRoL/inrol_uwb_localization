#include "libinroluwb/filter/kalman_filter.hpp"
#include "libinroluwb/filter/filter_math_utility.hpp"
#include "libinroluwb/filter/jacobian.hpp"
#include "libinroluwb/optimizer/math/harmonics.hpp"

namespace inrol {
  KalmanFilter::KalmanFilter(filter_input_t input)
      : _anchor_poses(input.anchor_poses),
        _tag_coeffs(input.tag_coeffs),
        _anchor_coeffs(input.anchor_coeffs),
        _param(input.param),
        _type(input.type),
        _uncertainty_param(input.uncertainty_param),
        _gt_poses(input.gt_poses) {
    pose_t init_pose = _gt_poses.front().pose;
    _R << init_pose.q.toRotationMatrix();
    _v = get_init_velocity_from_traj(_gt_poses);
    _p = init_pose.p;
    _wb = Vector3d(0, 0, 0);
    _ab = Vector3d(0, 0, 0);

    _P = 0.2 * Matrix15d::Identity();
    _Q = Matrix15d::Zero();
    Matrix3d I3 = Matrix3d::Identity();
    _P.block(6, 6, 3, 3) = I3 * 0.02;
    _P.block(9, 9, 3, 3) = I3 * 0.5;
    _P.block(12, 12, 3, 3) = I3 * 0.5;
    double sa = _param->sensor_noise.accel;
    double sw = _param->sensor_noise.gyro;
    double sba = _param->sensor_noise.bias_a;
    double sbw = _param->sensor_noise.bias_w;
    _Q.block(3, 3, 3, 3) = I3 * sa * sa;
    _Q.block(6, 6, 3, 3) = I3 * sw * sw;
    _Q.block(9, 9, 3, 3) = I3 * sba * sba;
    _Q.block(12, 12, 3, 3) = I3 * sbw * sbw;

    if (_type == ModelType::A) {
      _su = std::get<gaussian_param_t>(_uncertainty_param).sigma;
      LOG(INFO) << "Model A - (sigma): (" << _su << ")";
    } else if (_type == ModelType::AH) {
      _su = std::get<gaussian_param_t>(_uncertainty_param).sigma;
      LOG(INFO) << "Model AH - (sigma): (" << _su << ")";
    } else if (_type == ModelType::B) {
      _su = std::get<gaussian_param_t>(_uncertainty_param).sigma;
      LOG(INFO) << "Model B - (sigma): (" << _su << ")";
    } else if (_type == ModelType::BH) {
      _su = std::get<gaussian_param_t>(_uncertainty_param).sigma;
      LOG(INFO) << "Model BH - (sigma): (" << _su << ")";
    } else if (_type == ModelType::C) {
      _su = std::get<asymmetric_param_t>(_uncertainty_param).sigma;
      _gu = std::get<asymmetric_param_t>(_uncertainty_param).gamma;
      LOG(INFO) << "Model C - (sigma, gamma): (" << _su << ", " << _gu << ")";
    }

    _gravity_vector = Vector3d(0, 0, -9.81);
    for (auto const pair : _anchor_poses) {
      _last_uwb_data[pair.first] = std::nullopt;
    }

    _degree = _param->harmonics.degree;
    _degree_container.push_back(0);
    _order_container.push_back(0);
    for (int i = 1; i < (_degree + 1) * (_degree + 1); i++) {
      int l = index2degree(i);
      int m = index2order(i, l);
      _degree_container.push_back(l);
      _order_container.push_back(m);
    }

    for (auto const& i : input.imu_dataset)
      _data.insert(std::pair<double, data_t>(i.timestamp, i));
    for (auto const& u : input.uwb_dataset)
      _data.insert(std::pair<double, data_t>(u.timestamp, u));

    auto T_cam_imu = _param->extrinsics.T_cam_imu;
    auto T_mocap_cam = _param->extrinsics.T_mocap_cam;
    auto T_mocap_tag = _param->extrinsics.T_mocap_tag;
    auto T_imu_tag = T_cam_imu.inverse() * T_mocap_cam.inverse() * T_mocap_tag;
    Matrix3d R = T_imu_tag.block(0, 0, 3, 3);
    _T_imu_tag = pose_t {
      .p = T_imu_tag.block(0, 3, 3, 1),
      .q = Quaterniond(R),
    };
  }

  void KalmanFilter::run() {
    size_t i = 0;
    _last_timestamp = _data.begin()->first;
    for (auto const data : _data) {
      double dt = data.first - _last_timestamp;
      if (data.second.index() == 0) {
        auto imu = std::get<imu_data_t>(data.second);
        if (dt != 0) {
          if (_last_imu) {
            imu = imu_data_t {
              .wx = (imu.wx + _last_imu->wx) / 2.0,
              .wy = (imu.wy + _last_imu->wy) / 2.0,
              .wz = (imu.wz + _last_imu->wz) / 2.0,
              .ax = (imu.ax + _last_imu->ax) / 2.0,
              .ay = (imu.ay + _last_imu->ay) / 2.0,
              .az = (imu.az + _last_imu->az) / 2.0,
            };
          }
          propagation(imu, dt);
        }
        _last_imu = imu;
      } else {
        if (!_last_imu) {
          continue;
        }
        auto uwb = std::get<uwb_data_t>(data.second);

        if (dt != 0)
          propagation(*_last_imu, dt);
        correction(uwb);

        if (i % 10 == 0) {
          posestamped_t pose {
            .pose =
              pose_t {
                .p = _p,
                .q = Eigen::Quaterniond(_R),
              },
            .timestamp = data.first,
          };
          _estimated_traj.push_back(pose);
        }
      }
      _last_timestamp = data.first;
      i++;
    }
  }

  void KalmanFilter::propagation(const imu_data_t imu, const double dt) {
    Vector3d am(imu.ax, imu.ay, imu.az);
    Vector3d wm(imu.wx, imu.wy, imu.wz);
    Vector3d bar_a = am - _ab;
    Vector3d bar_w = wm - _wb;
    Matrix<double, 15, 15> Q = _Q;
    Q.block(3, 3, 6, 6) = _Q.block(3, 3, 6, 6) * dt * dt;
    Q.block(9, 9, 6, 6) = _Q.block(9, 9, 6, 6) * dt;

    _p = _p + _v * dt + _R * get_Gamma2(bar_w * dt) * bar_a * dt * dt +
      0.5 * _gravity_vector * dt * dt;
    _v = _v + _R * get_Gamma1(bar_w * dt) * bar_a * dt + _gravity_vector * dt;
    _R = _R * get_Gamma0(bar_w * dt);

    Matrix<double, 15, 15> Phi = get_error_Phi(am, wm, dt);
    _P = Phi * _P * Phi.transpose() + Q;
  }

  void KalmanFilter::correction(const uwb_data_t uwb) {
    size_t id = uwb.anchor_idx;
    double d = uwb.distance;
    if (!_last_uwb_data[id]) {
      _last_uwb_data[id] = d;
    }

    pose_t tag_pose = pose_t {
      .p = _p + _R * _T_imu_tag.p,
      .q = Quaterniond(_R) * _T_imu_tag.q,
    };
    pose_t anchor_pose = _anchor_poses.at(id);
    Vector3d ray = tag_pose.p - anchor_pose.p;
    double bias = get_directional_bias(tag_pose, anchor_pose);

    Vector3d p = _p;
    Vector3d v = _v;
    Matrix3d R = _R;
    Vector3d ab = _ab;
    Vector3d wb = _wb;

    Vector15d update = Vector15d::Zero();
    Vector15d K = Vector15d::Zero();
    Matrix<double, 1, 15> H = Matrix<double, 1, 15>::Zero();
    if (_type == ModelType::A) {
      H = get_measurement_jacobian(p, R, anchor_pose);

      double ey = d - ray.norm();
      double s = H * _P * H.transpose() + _su * _su;
      K = _P * H.transpose() / s;
      update = K * ey;
    } else if (_type == ModelType::AH) {
      for (int i = 0; i < 3; i++) {
        tag_pose = pose_t {
          .p = p + R * _T_imu_tag.p,
          .q = Quaterniond(R) * _T_imu_tag.q,
        };

        H = get_measurement_jacobian(p, R, anchor_pose);
        ray = tag_pose.p - anchor_pose.p;

        double ey = d - ray.norm();
        double psi = get_psi_huber(ey, _su);
        double s = H * _P * H.transpose() + 1.0 / psi;
        K = _P * H.transpose() / s;
        update = K * (ey + H * update);

        p = _p + update.segment(0, 3);
        v = _v + update.segment(3, 3);
        R = _R * get_Gamma0(update.segment(6, 3));
        ab = _ab + update.segment(9, 3);
        wb = _wb + update.segment(12, 3);

        auto error =
          _P.inverse() * update - H.transpose() * psi * (d - ray.norm());
      }
    } else if (_type == ModelType::B) {
      H = get_measurement_jacobian(p, R, anchor_pose);

      double ey = d - ray.norm() - bias;
      double s = H * _P * H.transpose() + _su * _su;
      K = _P * H.transpose() / s;
      update = K * ey;
    } else if (_type == ModelType::BH) {
      for (int i = 0; i < 3; i++) {
        tag_pose = pose_t {
          .p = p + R * _T_imu_tag.p,
          .q = Quaterniond(R) * _T_imu_tag.q,
        };

        H = get_measurement_jacobian(p, R, anchor_pose);
        ray = tag_pose.p - anchor_pose.p;
        bias = get_directional_bias(tag_pose, anchor_pose);

        double ey = d - ray.norm() - bias;
        double psi = get_psi_huber(ey, _su);
        double s = H * _P * H.transpose() + 1.0 / psi;
        K = _P * H.transpose() / s;
        update = K * (ey + H * update);

        p = _p + update.segment(0, 3);
        v = _v + update.segment(3, 3);
        R = _R * get_Gamma0(update.segment(6, 3));
        ab = _ab + update.segment(9, 3);
        wb = _wb + update.segment(12, 3);
      }
    } else if (_type == ModelType::C) {
      for (int i = 0; i < 3; i++) {
        tag_pose = pose_t {
          .p = p + R * _T_imu_tag.p,
          .q = Quaterniond(R) * _T_imu_tag.q,
        };

        H = get_measurement_jacobian(p, R, anchor_pose);
        ray = tag_pose.p - anchor_pose.p;
        bias = get_directional_bias(tag_pose, anchor_pose);

        double ey = d - ray.norm() - bias;
        double psi = get_psi_asymm(ey, _su, _gu);
        double s = H * _P * H.transpose() + 1.0 / psi;
        K = _P * H.transpose() / s;
        update = K * (ey + H * update);

        p = _p + update.segment(0, 3);
        v = _v + update.segment(3, 3);
        R = _R * get_Gamma0(update.segment(6, 3));
        ab = _ab + update.segment(9, 3);
        wb = _wb + update.segment(12, 3);

        auto error =
          _P.inverse() * update - H.transpose() * psi * (d - ray.norm() - bias);
      }
    } else {
      LOG(ERROR) << "Not supported type";
    }

    _p = _p + update.segment(0, 3);
    _v = _v + update.segment(3, 3);
    _R = _R * get_Gamma0(update.segment(6, 3));
    _ab = _ab + update.segment(9, 3);
    _wb = _wb + update.segment(12, 3);

    if (_type != ModelType::A && _type != ModelType::B) {
      tag_pose = pose_t {
        .p = _p + _R * _T_imu_tag.p,
        .q = Quaterniond(_R) * _T_imu_tag.q,
      };
      H = get_measurement_jacobian(_p, _R, anchor_pose);
      ray = tag_pose.p - anchor_pose.p;
      bias = get_directional_bias(tag_pose, anchor_pose);

      double ey = d - ray.norm() - bias;
      double psi = get_psi_huber(ey, _su);
      if (_type == ModelType::C) {
        psi = get_psi_asymm(ey, _su, _gu);
      }
      double s = H * _P * H.transpose() + 1.0 / psi;
      K = _P * H.transpose() / s;
    }
    _P = (Matrix15d::Identity() - K * H) * _P;

    Matrix<double, 15, 15> G = Matrix<double, 15, 15>::Identity();
    G.block(6, 6, 3, 3) =
      Matrix3d::Identity() - hat(0.5 * update.segment(6, 3));
    _P = G * _P * G.transpose();
  }

  double KalmanFilter::get_directional_bias(
    pose_t tag_pose, pose_t anchor_pose) {
    Vector3d ray_world_coord = anchor_pose.p - tag_pose.p;
    Vector3d ray_tag_coord =
      tag_pose.q.normalized().toRotationMatrix().transpose() * ray_world_coord;
    Vector3d ray_anchor_coord =
      -anchor_pose.q.toRotationMatrix().transpose() * ray_world_coord;

    double ray_length = ray_world_coord.norm();

    double tag_theta = std::acos(ray_tag_coord(2) / ray_length);
    double tag_phi = std::atan2(ray_tag_coord(1), ray_tag_coord(0));
    double anchor_theta = std::acos(ray_anchor_coord(2) / ray_length);
    double anchor_phi = std::atan2(ray_anchor_coord(1), ray_anchor_coord(0));

    double bias =
      _tag_coeffs[0] * real_spherical_harmonics(0, 0, tag_theta, tag_phi);
    for (int i = 1; i < (_degree + 1) * (_degree + 1); i++) {
      int l = _degree_container[i];
      int m = _order_container[i];
      double tag_value = real_spherical_harmonics(l, m, tag_theta, tag_phi);
      double tag_coeffs = _tag_coeffs[i];
      bias += tag_coeffs * tag_value;

      double anchor_value =
        real_spherical_harmonics(l, m, anchor_theta, anchor_phi);
      double anchor_coeffs = _anchor_coeffs[i];
      bias += anchor_coeffs * anchor_value;
    }
    return bias;
  }

  Matrix<double, 1, 15> KalmanFilter::get_measurement_jacobian(
    Vector3d p, Matrix3d R, pose_t anchor_pose) {
    Matrix<double, 1, 15> H = Matrix<double, 1, 15>::Zero();

    pose_t tag_pose = pose_t {
      .p = p + R * _T_imu_tag.p,
      .q = Quaterniond(R) * _T_imu_tag.q,
    };
    Vector3d ray = tag_pose.p - anchor_pose.p;
    Matrix3d J_of_rayvector_wrt_p = Matrix3d::Identity();
    Matrix3d J_of_rayvector_wrt_dth =
      J_of_rotation_action_wrt_q(Quaterniond(R), _T_imu_tag.p) *
      J_of_q_wrt_dth(Quaterniond(R));
    auto J_of_distance_wrt_p = J_of_norm(ray) * J_of_rayvector_wrt_p;
    auto J_of_distance_wrt_dth = J_of_norm(ray) * J_of_rayvector_wrt_dth;

    H.segment(0, 3) = J_of_distance_wrt_p;
    H.segment(6, 3) = J_of_distance_wrt_dth;
    return H;
  }

  Matrix<double, 15, 15> KalmanFilter::get_error_Phi(
    const Vector3d am, const Vector3d wm, const double dt) {
    Matrix<double, 15, 15> result = Matrix<double, 15, 15>::Identity();

    Vector3d bar_w = wm - _wb;
    Vector3d bar_a = am - _ab;

    Matrix3d Pv = Matrix3d::Identity();
    Matrix3d Vth = -_R * hat(bar_a);
    Matrix3d Thth = -hat(bar_w);
    Matrix3d Va = -_R;
    Matrix3d Thw = -Matrix3d::Identity();

    Matrix3d Sigmma0 = get_Sigmma0(bar_w, dt, Thth);
    Matrix3d Sigmma1 = get_Sigmma1(bar_w, dt, Thth);
    Matrix3d Sigmma2 = get_Sigmma2(bar_w, dt, Thth);
    Matrix3d Sigmma3 = get_Sigmma3(bar_w, dt, Thth);

    result.block(0, 3, 3, 3) = Pv * dt;
    result.block(0, 6, 3, 3) = Pv * Vth * Sigmma2;
    result.block(0, 9, 3, 3) = 0.5 * Pv * Va * dt * dt;
    result.block(0, 12, 3, 3) = Pv * Vth * Sigmma3 * Thw;
    result.block(3, 6, 3, 3) = Vth * Sigmma1;
    result.block(3, 9, 3, 3) = Va * dt;
    result.block(3, 12, 3, 3) = Vth * Sigmma2 * Thw;
    result.block(6, 6, 3, 3) = Sigmma0;
    result.block(6, 12, 3, 3) = Sigmma1 * Thw;
    return result;
  }

  vector<posestamped_t> KalmanFilter::get_trajectory() {
    return _estimated_traj;
  }

  double KalmanFilter::get_RMSE() {
    std::vector<double> error_queue;
    double initial_t = _gt_poses.front().timestamp;
    double last_t = _gt_poses.back().timestamp;
    for (auto const& pose_ts : _estimated_traj) {
      double ts = pose_ts.timestamp;
      if (ts < initial_t)
        continue;
      if (ts > last_t)
        continue;
      Vector3d position = pose_ts.pose.p;
      Vector3d gt_position;
      for (size_t i = 0; i < _gt_poses.size(); i++) {
        auto gt_ts = _gt_poses.at(i);
        if (gt_ts.timestamp > ts) {
          double ts_l = _gt_poses.at(i - 1).timestamp;
          auto position_l = _gt_poses.at(i - 1).pose.p;
          double ts_r = _gt_poses.at(i).timestamp;
          auto position_r = _gt_poses.at(i).pose.p;
          double u = (ts - ts_l) / (ts_r - ts_l);
          gt_position = position_l * (1 - u) + position_r * u;
          break;
        }
      }
      auto position_error = position - gt_position;
      error_queue.push_back(position_error.norm());
    }
    double error_size = static_cast<double>(error_queue.size());
    double error_sqsum = 0;
    for (auto const& error : error_queue) {
      error_sqsum += error * error;
    }
    return std::sqrt(error_sqsum / error_size);
  }
}  // namespace inrol
