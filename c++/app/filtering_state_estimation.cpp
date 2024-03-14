#include "libinroluwb/filter/kalman_filter.hpp"
#include "libinroluwb/utils/directory.hpp"
#include "libinroluwb/utils/param.hpp"
#include "libinroluwb/utils/text.hpp"

#include <glog/logging.h>

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  google::ParseCommandLineFlags(&argc, &argv, true);
  FLAGS_alsologtostderr = 1;

  const std::string arg1_project_directories = argv[1];
  auto const type = inrol::string2modeltype(argv[2]);
  auto const directories = inrol::get_directory_name(arg1_project_directories);

  auto param = inrol::get_parameters(directories->config + "/parameter.yaml");
  param->print();

  auto imu_dataset = inrol::get_imu_data_from_txt(
    directories->parsed_filter_dataset + "/imu.txt");
  auto uwb_dataset = inrol::get_uwb_data_from_txt(
    directories->parsed_filter_dataset + "/uwb.txt");
  auto anchor_poses = inrol::get_anchor_poses_from_txt(
    directories->parsed_filter_dataset + "/anchor.txt");

  std::string in_dirname;
  std::string out_dirname;
  std::string anchor_dirname;
  switch (type) {
  case inrol::ModelType::A:
    in_dirname = directories->model_calibration_result_A;
    out_dirname = directories->kalman_filter_result_A;
    anchor_dirname = directories->optimized_result_A;
    break;
  case inrol::ModelType::AH:
    in_dirname = directories->model_calibration_result_A;
    out_dirname = directories->kalman_filter_result_AH;
    anchor_dirname = directories->optimized_result_A;
    break;
  case inrol::ModelType::B:
    in_dirname = directories->model_calibration_result_B;
    out_dirname = directories->kalman_filter_result_B;
    anchor_dirname = directories->optimized_result_B;
    break;
  case inrol::ModelType::BH:
    in_dirname = directories->model_calibration_result_B;
    out_dirname = directories->kalman_filter_result_BH;
    anchor_dirname = directories->optimized_result_B;
    break;
  case inrol::ModelType::C:
    in_dirname = directories->model_calibration_result_C;
    out_dirname = directories->kalman_filter_result_C;
    anchor_dirname = directories->optimized_result_C;
    break;
  default:
    LOG(ERROR) << "Not supported type";
    break;
  }

  bool use_calibrated_anchor = false;
  if (use_calibrated_anchor) {
    anchor_poses = inrol::get_anchor_poses_from_txt(
      anchor_dirname + "/aligned_anchor_poses.txt");
  }

  std::string uncertainty_param_file_name =
    in_dirname + "/uncertainty_param.txt";
  auto uncertainty_param =
    inrol::get_measurement_uncertainty_param(uncertainty_param_file_name, type);

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

  auto gt_poses = inrol::get_poses_from_txt(
    directories->parsed_filter_dataset + "/imu_pose.txt");

  inrol::filter_input_t input {
    .imu_dataset = imu_dataset,
    .uwb_dataset = uwb_dataset,
    .anchor_poses = anchor_poses,
    .gt_poses = gt_poses,
    .type = type,
    .uncertainty_param = uncertainty_param,
    .tag_coeffs = tag_coeffs,
    .anchor_coeffs = anchor_coeffs,
    .param = param,
  };

  auto kalman_filter = inrol::KalmanFilter(input);
  kalman_filter.run();

  auto traj = kalman_filter.get_trajectory();
  double rmse = kalman_filter.get_RMSE();
  LOG(INFO) << "Position RMSE: " << rmse;

  inrol::write_posestamped_as_txt(traj, out_dirname, "/estimated_traj.txt");

  return 0;
}
