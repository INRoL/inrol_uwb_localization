#include "libinroluwb/optimizer/scale.hpp"
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

  auto uwb = inrol::get_uwb_data_from_txt(
    directories->parsed_anchor_dataset + "/uwb.txt");

  auto scaleless_knots = inrol::get_poses_from_txt(
    directories->scale_free_matched_bspline + "/matched_knots.txt");

  auto optimizer = inrol::ScaleFindingOptimizer(scaleless_knots, uwb, param);
  optimizer.solve();

  double scale = optimizer.get_scale_variable();
  auto anchor_position = optimizer.get_anchorposition_variable();

  auto scaled_traj = optimizer.get_scaled_trajectory();

  inrol::write_posestamped_as_txt(
    scaled_traj, directories->initial_guess_scaled_state, "/scaled_traj.txt");
  inrol::write_anchor_position_as_txt(
    anchor_position, directories->initial_guess_scaled_state,
    "/initial_guess_anchor_position.txt");
  inrol::write_scale_as_txt(
    scale, directories->initial_guess_scaled_state, "/scale.txt");

  return 0;
}
