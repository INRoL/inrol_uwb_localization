#include "libinroluwb/geometry/pose.hpp"
#include "libinroluwb/utils/binary.hpp"
#include "libinroluwb/utils/directory.hpp"
#include "libinroluwb/utils/text.hpp"

#include <glog/logging.h>

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  google::ParseCommandLineFlags(&argc, &argv, true);
  FLAGS_alsologtostderr = 1;

  const std::string arg1_project_directories = argv[1];
  auto const directories = inrol::get_directory_name(arg1_project_directories);

  auto const frames = inrol::readFramesBinary(directories->colmap_result);
  auto points_3D = inrol::readFeaturePointsBinary(directories->colmap_result);
  auto const timestamp_vector = inrol::get_image_timestamp_from_txt(
    directories->parsed_anchor_dataset + "/images.txt");

  CHECK(timestamp_vector.size() == frames.size())
    << "the numbers of images and frames are not matched";

  std::vector<inrol::posestamped_t> posestamed_v;
  for (size_t i = 0; i < frames.size(); i++) {
    auto frame = frames.at(i);
    double ts = timestamp_vector.at(i);
    posestamed_v.push_back(
      inrol::posestamped_t {.pose = frame.get_pose(), .timestamp = ts});
  }

  inrol::write_posestamped_as_txt(
    posestamed_v, directories->parsed_colmap_result, "/colmap_camera_traj.txt");

  std::vector<inrol::vision_data_t> vision;
  for (auto frame : frames) {
    frame.add_vision_data(timestamp_vector, vision);
  }
  points_3D = inrol::remove_unused_points(points_3D, vision);

  inrol::write_feature_points_as_txt(
    points_3D, directories->parsed_colmap_result,
    "/scale_free_feature_points.txt");
  inrol::write_vision_data_as_txt(
    vision, directories->parsed_colmap_result, "/vision_observation.txt");

  return 0;
}
