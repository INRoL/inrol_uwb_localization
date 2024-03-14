#include "libinroluwb/optimizer/batch.hpp"
#include "libinroluwb/optimizer/functor/anchor.hpp"
#include "libinroluwb/optimizer/functor/bias.hpp"
#include "libinroluwb/optimizer/functor/imu.hpp"
#include "libinroluwb/optimizer/functor/vision.hpp"
#include "libinroluwb/optimizer/functor/uwb.hpp"
#include "libinroluwb/utils/algorithm.hpp"

namespace inrol {
  using std::floor;
  using std::make_shared;

  FullBatchOptimizer::FullBatchOptimizer(batch_input_t batch_input)
      : _pose_knots(batch_input.pose_knots_initial_guess),
        _feature_points(batch_input.feature_points_initial_guess),
        _gravity_vector(batch_input.gravity_vector),
        _imu_dataset(batch_input.imu_dataset),
        _uwb_dataset(batch_input.uwb_dataset),
        _vision_dataset(batch_input.vision_dataset),
        _tag_coeffs(batch_input.tag_coeffs),
        _anchor_coeffs(batch_input.anchor_coeffs),
        _camera(make_shared<Camera>(batch_input.camera)),
        _param(batch_input.param),
        _degree(batch_input.param->harmonics.degree),
        _type(batch_input.type),
        _uncertainty_param(batch_input.uncertainty_param),
        _anchor_poses(batch_input.anchor_pose_prior),
        _duration(batch_input.pose_knots_initial_guess.back().timestamp) {
    size_t bias_knots_n = size_t(_duration / _param->spline.spline_bias_dt) + 3;
    _accel_bias_knots = vector<Vector3d>(bias_knots_n, Vector3d(0.0, 0.0, 0.0));
    _gyro_bias_knots = vector<Vector3d>(bias_knots_n, Vector3d(0.0, 0.0, 0.0));

    print_problem();
    add_residuals();
    _problem.SetParameterBlockConstant(_pose_knots.at(0).pose.p.data());
    _problem.SetParameterBlockConstant(
      _pose_knots.at(0).pose.q.coeffs().data());
  }

  void FullBatchOptimizer::print_problem() {
    size_t dimension = _pose_knots.size() * 6 + _accel_bias_knots.size() * 6 +
      _feature_points.size() * 3 + _anchor_poses.size() * 6 +
      (_degree + 1) * (_degree + 1) - 1;
    LOG(INFO) << "\n==========Optimization=========="
              << "\n  => total optimization variable dimension:" << dimension;
  }

  vector<posestamped_t> FullBatchOptimizer::get_pose_knots() {
    return _pose_knots;
  }

  vector<Vector3d> FullBatchOptimizer::get_accel_bias_knots() {
    return _accel_bias_knots;
  }

  vector<Vector3d> FullBatchOptimizer::get_gyro_bias_knots() {
    return _gyro_bias_knots;
  }

  aligned_unordered_map<size_t, Vector3d>
  FullBatchOptimizer::get_feature_points() {
    return _feature_points;
  }

  aligned_unordered_map<size_t, pose_t> FullBatchOptimizer::get_anchor_poses() {
    return _anchor_poses;
  }

  Vector3d FullBatchOptimizer::get_gravity_vector() {
    return _gravity_vector;
  }

  void FullBatchOptimizer::add_residuals() {
    add_vision_residuals();
    add_imu_residuals();
    add_bias_residuals();
    add_uwb_residuals();
    add_anchor_prior_residuals();
  }

  void FullBatchOptimizer::add_vision_residuals() {
    ceres::LossFunction* loss_function = new ceres::HuberLoss(2.0);
    ceres::Manifold* quaternion_manifold = new ceres::EigenQuaternionManifold;

    double pixel_noise = _param->sensor_noise.pixel;
    double dt = _param->spline.spline_pose_dt;
    for (auto const& data : _vision_dataset) {
      double ts = data.timestamp;
      if (ts < 0)
        continue;
      size_t landmark_idx = data.point3d_id;

      size_t pose_idx = floor(ts / dt);
      if (pose_idx + 3 >= _pose_knots.size())
        continue;

      double t1 = pose_idx * dt;
      double u = (ts - t1) / dt;
      auto l = _feature_points.at(landmark_idx);
      CHECK_NEAR(t1, _pose_knots.at(pose_idx + 1).timestamp, 1e-6);

      pose_t* pose0_ptr = &_pose_knots.at(pose_idx).pose;
      pose_t* pose1_ptr = &_pose_knots.at(pose_idx + 1).pose;
      pose_t* pose2_ptr = &_pose_knots.at(pose_idx + 2).pose;
      pose_t* pose3_ptr = &_pose_knots.at(pose_idx + 3).pose;

      // feature detection outlier rejection
      pose_t cam_pose =
        pose_b_spline(*pose0_ptr, *pose1_ptr, *pose2_ptr, *pose3_ptr, u);
      Vector3d l_B =
        cam_pose.q.toRotationMatrix().transpose() * (l - cam_pose.p);

      auto projected_uv = _camera->project(l_B);
      auto measured_uv = data.uv;
      double residuals = (projected_uv - measured_uv).norm();
      if (residuals > 4) {
        continue;
      }

      ceres::CostFunction* vision_cost_function =
        VisionResidualTerm::Create(data, pixel_noise, u, _camera);

      _problem.AddResidualBlock(
        vision_cost_function, loss_function, pose0_ptr->p.data(),
        pose0_ptr->q.coeffs().data(), pose1_ptr->p.data(),
        pose1_ptr->q.coeffs().data(), pose2_ptr->p.data(),
        pose2_ptr->q.coeffs().data(), pose3_ptr->p.data(),
        pose3_ptr->q.coeffs().data(), _feature_points.at(landmark_idx).data());

      _problem.SetManifold(pose0_ptr->q.coeffs().data(), quaternion_manifold);
      _problem.SetManifold(pose1_ptr->q.coeffs().data(), quaternion_manifold);
      _problem.SetManifold(pose2_ptr->q.coeffs().data(), quaternion_manifold);
      _problem.SetManifold(pose3_ptr->q.coeffs().data(), quaternion_manifold);
    }
  }

  void FullBatchOptimizer::add_imu_residuals() {
    ceres::LossFunction* loss_function = nullptr;
    ceres::Manifold* quaternion_manifold = new ceres::EigenQuaternionManifold;
    ceres::Manifold* sphere_manifold = new ceres::SphereManifold<3>;

    double accel_sigma = _param->sensor_noise.accel;
    double gyro_sigma = _param->sensor_noise.gyro;
    double pose_dt = _param->spline.spline_pose_dt;
    double bias_dt = _param->spline.spline_bias_dt;

    Eigen::Matrix3d R_cam_imu = _param->extrinsics.T_cam_imu.block<3, 3>(0, 0);
    Vector3d p_cam_imu = _param->extrinsics.T_cam_imu.block<3, 1>(0, 3);
    pose_t T_cam_imu = pose_t {
      .p = p_cam_imu,
      .q = Quaterniond(R_cam_imu),
    };

    for (auto const& data : _imu_dataset) {
      double ts = data.timestamp;

      size_t pose_idx = floor(ts / pose_dt);
      size_t bias_idx = floor(ts / bias_dt);
      if (ts < 0)
        continue;
      if (pose_idx + 3 >= _pose_knots.size())
        continue;
      if (bias_idx + 3 >= _accel_bias_knots.size())
        continue;

      double pose_t1 = pose_idx * pose_dt;
      double pose_u = (ts - pose_t1) / pose_dt;

      double bias_t1 = bias_idx * bias_dt;
      double bias_u = (ts - bias_t1) / bias_dt;
      CHECK_NEAR(pose_t1, _pose_knots.at(pose_idx + 1).timestamp, 1e-6);
      CHECK(pose_u >= 0);
      CHECK(pose_u <= 1);
      CHECK(bias_u >= 0);
      CHECK(bias_u <= 1);

      pose_t* pose0_ptr = &_pose_knots.at(pose_idx).pose;
      pose_t* pose1_ptr = &_pose_knots.at(pose_idx + 1).pose;
      pose_t* pose2_ptr = &_pose_knots.at(pose_idx + 2).pose;
      pose_t* pose3_ptr = &_pose_knots.at(pose_idx + 3).pose;
      Vector3d* a_bias0_ptr = &_accel_bias_knots.at(bias_idx);
      Vector3d* a_bias1_ptr = &_accel_bias_knots.at(bias_idx + 1);
      Vector3d* a_bias2_ptr = &_accel_bias_knots.at(bias_idx + 2);
      Vector3d* a_bias3_ptr = &_accel_bias_knots.at(bias_idx + 3);
      Vector3d* g_bias0_ptr = &_gyro_bias_knots.at(bias_idx);
      Vector3d* g_bias1_ptr = &_gyro_bias_knots.at(bias_idx + 1);
      Vector3d* g_bias2_ptr = &_gyro_bias_knots.at(bias_idx + 2);
      Vector3d* g_bias3_ptr = &_gyro_bias_knots.at(bias_idx + 3);

      ceres::CostFunction* accel_cost_function =
        AccelerometerResidualTerm::Create(
          data, accel_sigma, pose_u, bias_u, pose_dt, T_cam_imu);
      _problem.AddResidualBlock(
        accel_cost_function, loss_function, pose0_ptr->p.data(),
        pose0_ptr->q.coeffs().data(), pose1_ptr->p.data(),
        pose1_ptr->q.coeffs().data(), pose2_ptr->p.data(),
        pose2_ptr->q.coeffs().data(), pose3_ptr->p.data(),
        pose3_ptr->q.coeffs().data(), a_bias0_ptr->data(), a_bias1_ptr->data(),
        a_bias2_ptr->data(), a_bias3_ptr->data(), _gravity_vector.data());

      ceres::CostFunction* gyro_cost_function = GyroscopeResidualTerm::Create(
        data, gyro_sigma, pose_u, bias_u, pose_dt, T_cam_imu);
      _problem.AddResidualBlock(
        gyro_cost_function, loss_function, pose0_ptr->q.coeffs().data(),
        pose1_ptr->q.coeffs().data(), pose2_ptr->q.coeffs().data(),
        pose3_ptr->q.coeffs().data(), g_bias0_ptr->data(), g_bias1_ptr->data(),
        g_bias2_ptr->data(), g_bias3_ptr->data());

      _problem.SetManifold(pose0_ptr->q.coeffs().data(), quaternion_manifold);
      _problem.SetManifold(pose1_ptr->q.coeffs().data(), quaternion_manifold);
      _problem.SetManifold(pose2_ptr->q.coeffs().data(), quaternion_manifold);
      _problem.SetManifold(pose3_ptr->q.coeffs().data(), quaternion_manifold);
    }
    _problem.SetManifold(_gravity_vector.data(), sphere_manifold);
  }

  void FullBatchOptimizer::add_bias_residuals() {
    ceres::LossFunction* loss_function = nullptr;

    double accel_bias_sigma = _param->sensor_noise.bias_a;
    double gyro_bias_sigma = _param->sensor_noise.bias_w;
    double dt = _param->spline.spline_bias_dt;
    for (double ts = 0; ts < _duration; ts += dt / 10.0) {
      size_t bias_idx = floor(ts / dt);
      if (bias_idx + 3 >= _accel_bias_knots.size())
        continue;

      double t1 = bias_idx * dt;
      double u = (ts - t1) / dt;

      Vector3d* a_bias0_ptr = &_accel_bias_knots.at(bias_idx);
      Vector3d* a_bias1_ptr = &_accel_bias_knots.at(bias_idx + 1);
      Vector3d* a_bias2_ptr = &_accel_bias_knots.at(bias_idx + 2);
      Vector3d* a_bias3_ptr = &_accel_bias_knots.at(bias_idx + 3);
      Vector3d* g_bias0_ptr = &_gyro_bias_knots.at(bias_idx);
      Vector3d* g_bias1_ptr = &_gyro_bias_knots.at(bias_idx + 1);
      Vector3d* g_bias2_ptr = &_gyro_bias_knots.at(bias_idx + 2);
      Vector3d* g_bias3_ptr = &_gyro_bias_knots.at(bias_idx + 3);

      ceres::CostFunction* bias_a_cost_function =
        BiasResidualTerm::Create(accel_bias_sigma, u, dt);
      ceres::CostFunction* bias_g_cost_function =
        BiasResidualTerm::Create(gyro_bias_sigma, u, dt);

      _problem.AddResidualBlock(
        bias_a_cost_function, loss_function, a_bias0_ptr->data(),
        a_bias1_ptr->data(), a_bias2_ptr->data(), a_bias3_ptr->data());
      _problem.AddResidualBlock(
        bias_g_cost_function, loss_function, g_bias0_ptr->data(),
        g_bias1_ptr->data(), g_bias2_ptr->data(), g_bias3_ptr->data());
    }
  }

  void FullBatchOptimizer::add_uwb_residuals() {
    ceres::LossFunction* loss_function = nullptr;
    ceres::Manifold* quaternion_manifold = new ceres::EigenQuaternionManifold;

    auto T_c2t =
      _param->extrinsics.T_mocap_cam.inverse() * _param->extrinsics.T_mocap_tag;
    pose_t T_cam_tag {
      .p = T_c2t.block<3, 1>(0, 3), .q = Quaterniond(T_c2t.block<3, 3>(0, 0))};

    for (auto const& uwb : _uwb_dataset) {
      double d = uwb.distance;
      size_t id = uwb.anchor_idx;

      double ts = uwb.timestamp;
      double dt = _param->spline.spline_pose_dt;
      size_t idx = floor(ts / dt);
      if (ts < 0)
        continue;
      if (ts > _pose_knots.back().timestamp - dt)
        continue;

      double t1 = idx * dt;
      CHECK_NEAR(t1, _pose_knots.at(idx + 1).timestamp, 1e-6);

      double u = (ts - t1) / dt;

      pose_t* pose0_ptr = &_pose_knots.at(idx).pose;
      pose_t* pose1_ptr = &_pose_knots.at(idx + 1).pose;
      pose_t* pose2_ptr = &_pose_knots.at(idx + 2).pose;
      pose_t* pose3_ptr = &_pose_knots.at(idx + 3).pose;

      ceres::CostFunction* cost_function = PoseUWBErrorTerm::Create(
        uwb, _degree, u, T_cam_tag, _tag_coeffs, _anchor_coeffs, _type,
        _uncertainty_param);

      _problem.AddResidualBlock(
        cost_function, loss_function, pose0_ptr->p.data(),
        pose0_ptr->q.coeffs().data(), pose1_ptr->p.data(),
        pose1_ptr->q.coeffs().data(), pose2_ptr->p.data(),
        pose2_ptr->q.coeffs().data(), pose3_ptr->p.data(),
        pose3_ptr->q.coeffs().data(), _anchor_poses.at(uwb.anchor_idx).p.data(),
        _anchor_poses.at(uwb.anchor_idx).q.coeffs().data());

      _problem.SetManifold(pose0_ptr->q.coeffs().data(), quaternion_manifold);
      _problem.SetManifold(pose1_ptr->q.coeffs().data(), quaternion_manifold);
      _problem.SetManifold(pose2_ptr->q.coeffs().data(), quaternion_manifold);
      _problem.SetManifold(pose3_ptr->q.coeffs().data(), quaternion_manifold);
    }
    for (auto& anchor_pose : _anchor_poses) {
      _problem.SetManifold(
        anchor_pose.second.q.coeffs().data(), quaternion_manifold);
    }
  }

  void FullBatchOptimizer::add_anchor_prior_residuals() {
    ceres::LossFunction* loss_function = nullptr;
    for (auto& anchor_pose : _anchor_poses) {
      ceres::CostFunction* cost_function = AnchorPriorErrorTerm::Create(
        anchor_pose.second.q, _param->sensor_noise.anchor);
      _problem.AddResidualBlock(
        cost_function, loss_function, anchor_pose.second.q.coeffs().data());
    }
  }

  bool FullBatchOptimizer::solve() {
    ceres::Solver::Options options;
    options.max_num_iterations = 50;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.update_state_every_iteration = true;
    LoggingCallback logging_callback(true);
    options.callbacks.push_back(&logging_callback);
    ceres::Solver::Summary summary;
    LOG(INFO) << "Optimization start!";
    ceres::Solve(options, &_problem, &summary);

    LOG(INFO) << summary.FullReport();
    return summary.IsSolutionUsable();
  }
}  // namespace inrol
