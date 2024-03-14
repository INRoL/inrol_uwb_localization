#include "libinroluwb/geometry/measurements.hpp"
#include "libinroluwb/optimizer/batch.hpp"
#include "libinroluwb/utils/binary.hpp"
#include "libinroluwb/utils/directory.hpp"
#include "libinroluwb/utils/param.hpp"
#include "libinroluwb/utils/text.hpp"

#include <glog/logging.h>

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  google::ParseCommandLineFlags(&argc, &argv, true);
  FLAGS_alsologtostderr = 1;

  const std::string arg1_project_directories = argv[1];
  const std::string arg2_calibration_type = argv[2];
  auto const directories = inrol::get_directory_name(arg1_project_directories);
  auto type = inrol::string2modeltype(arg2_calibration_type);

  auto param = inrol::get_parameters(directories->config + "/parameter.yaml");
  param->print();

  std::string in_dirname;
  std::string out_dirname;
  switch (type) {
  case inrol::ModelType::A:
    in_dirname = directories->model_calibration_result_A;
    out_dirname = directories->optimized_result_A;
    break;
  case inrol::ModelType::B:
    in_dirname = directories->model_calibration_result_B;
    out_dirname = directories->optimized_result_B;
    break;
  case inrol::ModelType::C:
    in_dirname = directories->model_calibration_result_C;
    out_dirname = directories->optimized_result_C;
    break;
  default:
    LOG(ERROR) << "Not supported type";
    break;
  }
  std::string uncertainty_param_file_name =
    in_dirname + "/uncertainty_param.txt";
  auto uncertainty_param =
    inrol::get_measurement_uncertainty_param(uncertainty_param_file_name, type);

  auto imu_dataset = inrol::get_imu_data_from_txt(
    directories->parsed_anchor_dataset + "/imu.txt");
  auto uwb_dataset = inrol::get_uwb_data_from_txt(
    directories->parsed_anchor_dataset + "/uwb.txt");
  auto ts_txt = inrol::get_image_timestamp_from_txt(
    directories->parsed_anchor_dataset + "/images.txt");
  auto initial_knots = inrol::get_poses_from_txt(
    directories->initial_guess_scaled_state + "/scaled_traj.txt");
  auto scale = inrol::get_optimized_scale_from_txt(
    directories->initial_guess_scaled_state + "/scale.txt");
  auto anchor_position = inrol::get_anchor_position_from_txt(
    directories->initial_guess_scaled_state +
    "/initial_guess_anchor_position.txt");
  auto tag_coeffs = inrol::get_harmonics_coeffs_from_txt(
    directories->model_calibration_result_B + "/tag_coeffs.txt");
  auto anchor_coeffs = inrol::get_harmonics_coeffs_from_txt(
    directories->model_calibration_result_B + "/anchor_coeffs.txt");
  if (type == inrol::ModelType::C) {
    tag_coeffs =
      inrol::get_harmonics_coeffs_from_txt(in_dirname + "/tag_coeffs.txt");
    anchor_coeffs =
      inrol::get_harmonics_coeffs_from_txt(in_dirname + "/anchor_coeffs.txt");
  }
  double bias_dt = param->spline.spline_bias_dt;
  Eigen::Vector3d g(0.0, 0.0, 9.81);

  auto camera = inrol::readCameraBinary(directories->colmap_result);
  auto feature_points = inrol::get_feature_points_from_txt(
    directories->parsed_colmap_result + "/scale_free_feature_points.txt");
  auto frames = inrol::readFramesBinary(directories->colmap_result);

  std::vector<inrol::vision_data_t> vision_dataset;
  for (auto frame : frames) {
    frame.add_vision_data(ts_txt, vision_dataset);
  }

  feature_points = inrol::remove_unused_points(feature_points, vision_dataset);
  for (auto& f : feature_points) {
    f.second = f.second * scale;
  }

  auto anchor_orientation_prior =
    inrol::get_orientations(directories->anchor_prior);
  inrol::aligned_unordered_map<size_t, inrol::pose_t> anchor_pose_prior;
  for (auto const& a : anchor_position) {
    auto R_prior = anchor_orientation_prior[a.first];
    inrol::pose_t pose {
      .p = a.second,
      .q = Eigen::Quaterniond(R_prior).normalized(),
    };
    anchor_pose_prior[a.first] = pose;
  }

  inrol::batch_input_t batch_input {
    .pose_knots_initial_guess = initial_knots,
    .feature_points_initial_guess = feature_points,
    .anchor_position_initial_guess = anchor_position,
    .gravity_vector = g,
    .tag_coeffs = tag_coeffs,
    .anchor_coeffs = anchor_coeffs,
    .imu_dataset = imu_dataset,
    .uwb_dataset = uwb_dataset,
    .vision_dataset = vision_dataset,
    .camera = camera,
    .param = param,
    .type = type,
    .uncertainty_param = uncertainty_param,
    .anchor_pose_prior = anchor_pose_prior,
  };

  auto optimizer = inrol::FullBatchOptimizer(batch_input);

  optimizer.solve();
  auto optimized_knots = optimizer.get_pose_knots();
  auto accel_bias_knots = optimizer.get_accel_bias_knots();
  auto gyro_bias_knots = optimizer.get_gyro_bias_knots();
  feature_points = optimizer.get_feature_points();
  auto anchor_poses = optimizer.get_anchor_poses();
  auto gravity_vector = optimizer.get_gravity_vector();

  inrol::write_feature_points_as_txt(
    feature_points, out_dirname, "/optimized_feature_points.txt");
  inrol::write_posestamped_as_txt(
    optimized_knots, out_dirname, "/optimized_pose_knots.txt");
  inrol::write_bias_knots_as_txt(
    accel_bias_knots, bias_dt, out_dirname, "/optimized_accel_knots.txt");
  inrol::write_bias_knots_as_txt(
    gyro_bias_knots, bias_dt, out_dirname, "/optimized_gyro_knots.txt");
  inrol::write_anchor_poses_as_txt(
    anchor_poses, out_dirname, "/optimized_anchor_poses.txt");
  inrol::write_gravity_vector_as_txt(
    gravity_vector, out_dirname, "/optimized_gravity_vector.txt");

  return 0;
}
