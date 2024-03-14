#include "libinroluwb/geometry/pose.hpp"
#include "libinroluwb/optimizer/match.hpp"
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

  auto colmap_poses = inrol::get_poses_from_txt(
    directories->parsed_colmap_result + "/colmap_camera_traj.txt");

  double dt = param->spline.spline_pose_dt;
  auto initial_knots = inrol::get_interpolated_knots(colmap_poses, dt);

  auto optimizer =
    inrol::SplineMatchingOptimizer(initial_knots, colmap_poses, dt);
  optimizer.solve();

  auto knots = optimizer.get_pose_knots();

  inrol::write_posestamped_as_txt(
    knots, directories->scale_free_matched_bspline, "/matched_knots.txt");
  return 0;
}
