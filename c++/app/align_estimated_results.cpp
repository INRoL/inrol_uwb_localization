#include "libinroluwb/geometry/pose.hpp"
#include "libinroluwb/optimizer/align.hpp"
#include "libinroluwb/utils/directory.hpp"
#include "libinroluwb/utils/param.hpp"
#include "libinroluwb/utils/text.hpp"

#include <glog/logging.h>

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  google::ParseCommandLineFlags(&argc, &argv, true);
  FLAGS_alsologtostderr = 1;

  const std::string arg1_project_directories = argv[1];
  auto const directories = inrol::get_directory_name(arg1_project_directories);

  auto param = inrol::get_parameters(directories->config + "/parameter.yaml");
  param->print();

  auto type = inrol::string2modeltype(argv[2]);
  std::string dirname;
  switch (type) {
  case inrol::ModelType::A:
    dirname = directories->optimized_result_A;
    break;
  case inrol::ModelType::B:
    dirname = directories->optimized_result_B;
    break;
  case inrol::ModelType::C:
    dirname = directories->optimized_result_C;
    break;
  default:
    LOG(ERROR) << "Not supported anchor calibration type";
    break;
  }

  LOG(INFO) << "Load optmized result in " << dirname;
  auto poses = inrol::get_poses_from_txt(dirname + "/optimized_pose_knots.txt");
  auto gt_poses = inrol::get_poses_from_txt(
    directories->parsed_anchor_dataset + "/cam_pose.txt");
  auto feature_points = inrol::get_feature_points_from_txt(
    dirname + "/optimized_feature_points.txt");
  auto anchor_poses =
    inrol::get_anchor_poses_from_txt(dirname + "/optimized_anchor_poses.txt");
  auto gravity_vector = inrol::get_gravity_vector_from_txt(
    dirname + "/optimized_gravity_vector.txt");

  auto optimizer = inrol::FrameTransformationOptimizer(
    poses, gt_poses, feature_points, anchor_poses, gravity_vector,
    param->spline.spline_pose_dt);

  optimizer.solve();

  poses = optimizer.get_poses();
  feature_points = optimizer.get_feature_points();
  anchor_poses = optimizer.get_anchor_poses();
  auto aligning_transform = optimizer.get_aligning_transformation();

  inrol::write_posestamped_as_txt(poses, dirname, "/aligned_pose_knots.txt");
  inrol::write_feature_points_as_txt(
    feature_points, dirname, "/aligned_feature_points.txt");
  inrol::write_anchor_poses_as_txt(
    anchor_poses, dirname, "/aligned_anchor_poses.txt");
  inrol::write_pose_as_txt(
    aligning_transform, dirname, "/aligning_transform.txt");
  return 0;
}
