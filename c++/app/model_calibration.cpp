#include "libinroluwb/optimizer/calibration.hpp"
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
  auto const type = inrol::string2modeltype(arg2_calibration_type);

  auto param = inrol::get_parameters(directories->config + "/parameter.yaml");
  param->print();

  auto tag_poses = inrol::get_poses_from_txt(
    directories->parsed_cali_dataset + "/tag_pose.txt");
  auto anchor_poses = inrol::get_anchor_poses_from_txt(
    directories->parsed_cali_dataset + "/anchor.txt");
  auto uwb_dataset =
    inrol::get_uwb_data_from_txt(directories->parsed_cali_dataset + "/uwb.txt");

  auto dataset = inrol::model_cali_dataset_t {
    .tag_poses = tag_poses,
    .anchor_poses = anchor_poses,
    .uwb_dataset = uwb_dataset,
  };

  auto optimizer = inrol::ModelCalibrationOptimizer(dataset, param, type);

  optimizer.solve();

  std::string outdir;
  if (type == inrol::ModelType::A) {
    outdir = directories->model_calibration_result_A;
  } else if (type == inrol::ModelType::B) {
    outdir = directories->model_calibration_result_B;
  } else if (type == inrol::ModelType::C) {
    outdir = directories->model_calibration_result_C;
  }

  auto uncertainty_param = optimizer.get_uncertainty_param();
  inrol::write_model_param_as_txt(
    uncertainty_param, type, outdir, "/uncertainty_param.txt");

  if (type == inrol::ModelType::B || type == inrol::ModelType::C) {
    auto tag_coeffs = optimizer.get_tag_coeffs();
    auto anchor_coeffs = optimizer.get_anchor_coeffs();
    inrol::write_harmonics_coeffs_as_txt(tag_coeffs, outdir, "/tag_coeffs.txt");
    inrol::write_harmonics_coeffs_as_txt(
      anchor_coeffs, outdir, "/anchor_coeffs.txt");
  }
  return 0;
}
