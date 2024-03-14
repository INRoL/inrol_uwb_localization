#include "libinroluwb/utils/binary.hpp"
#include "libinroluwb/geometry/feature.hpp"
#include "libinroluwb/utils/endian.hpp"

#include <sys/types.h>
#include <iostream>
#include <fstream>
#include <limits>

namespace inrol {
  Camera readCameraBinary(const string& path) {
    const string cam_path = path + "/cameras.bin";
    std::ifstream file(cam_path, std::ios::binary);
    CHECK(file.is_open()) << cam_path;

    const size_t num_cameras = readBinaryLittleEndian<uint64_t>(&file);
    const uint64_t camera_id = readBinaryLittleEndian<int>(&file);
    const int model_id = readBinaryLittleEndian<int>(&file);
    const int w = readBinaryLittleEndian<uint64_t>(&file);
    const int h = readBinaryLittleEndian<uint64_t>(&file);
    const double fx = readBinaryLittleEndian<double>(&file);
    const double fy = readBinaryLittleEndian<double>(&file);
    const double cx = readBinaryLittleEndian<double>(&file);
    const double cy = readBinaryLittleEndian<double>(&file);
    const double k1 = readBinaryLittleEndian<double>(&file);
    const double k2 = readBinaryLittleEndian<double>(&file);
    const double p1 = readBinaryLittleEndian<double>(&file);
    const double p2 = readBinaryLittleEndian<double>(&file);

    const vector<double> dist_coeffs = {k1, k2, p1, p2};
    file.close();
    return Camera(w, h, fx, fy, cx, cy, dist_coeffs);
  }

  aligned_unordered_map<size_t, Vector3d> readFeaturePointsBinary(
    const string& path) {
    aligned_unordered_map<size_t, Vector3d> result;

    const string points_path = path + "/points3D.bin";
    std::ifstream file(points_path, std::ios::binary);
    CHECK(file.is_open()) << points_path;

    const size_t num_points = readBinaryLittleEndian<uint64_t>(&file);

    for (size_t i = 0; i < num_points; i++) {
      const size_t id = readBinaryLittleEndian<uint64_t>(&file);

      double tx = readBinaryLittleEndian<double>(&file);
      double ty = readBinaryLittleEndian<double>(&file);
      double tz = readBinaryLittleEndian<double>(&file);

      readBinaryLittleEndian<uint8_t>(&file);
      readBinaryLittleEndian<uint8_t>(&file);
      readBinaryLittleEndian<uint8_t>(&file);

      double e = readBinaryLittleEndian<double>(&file);

      const size_t track_length = readBinaryLittleEndian<uint64_t>(&file);

      for (size_t j = 0; j < track_length; j++) {
        const uint32_t image_id = readBinaryLittleEndian<uint32_t>(&file);
        const uint32_t point2d_idx = readBinaryLittleEndian<uint32_t>(&file);
      }
      result.insert({id, Vector3d(tx, ty, tz)});
    }
    return result;
  }

  vector<Frame> readFramesBinary(const string& path) {
    vector<Frame> result;

    const string frames_path = path + "/images.bin";
    std::ifstream file(frames_path, std::ios::binary);
    CHECK(file.is_open()) << frames_path;

    const uint64_t invalid_point3d_id = std::numeric_limits<uint64_t>::max();

    const size_t num_reg_images = readBinaryLittleEndian<uint64_t>(&file);
    for (size_t i = 0; i < num_reg_images; i++) {
      uint32_t image_id = readBinaryLittleEndian<uint32_t>(&file);
      Frame frame(image_id);
      double qw = readBinaryLittleEndian<double>(&file);
      double qx = readBinaryLittleEndian<double>(&file);
      double qy = readBinaryLittleEndian<double>(&file);
      double qz = readBinaryLittleEndian<double>(&file);
      double tx = readBinaryLittleEndian<double>(&file);
      double ty = readBinaryLittleEndian<double>(&file);
      double tz = readBinaryLittleEndian<double>(&file);
      frame.set_pose(tx, ty, tz, qx, qy, qz, qw);
      uint32_t cam_idx = readBinaryLittleEndian<uint32_t>(&file);

      char name_char;
      string name;
      do {
        file.read(&name_char, 1);
        if (name_char != '\0') {
          name += name_char;
        }
      } while (name_char != '\0');

      size_t num_potential_points2D = readBinaryLittleEndian<uint64_t>(&file);

      std::unordered_map<double, size_t> x_coords;
      std::unordered_map<double, size_t> y_coords;
      vector<uint64_t> point3D_ids;
      for (size_t j = 0; j < num_potential_points2D; j++) {
        const double x = readBinaryLittleEndian<double>(&file);
        const double y = readBinaryLittleEndian<double>(&file);
        const size_t point3D_id = readBinaryLittleEndian<uint64_t>(&file);

        if (point3D_id == invalid_point3d_id)
          continue;

        auto search_x = x_coords.find(x);
        auto search_y = y_coords.find(y);
        if (
          (search_x != x_coords.end()) && (search_y != y_coords.end()) &&
          (search_x->second == search_y->second))
          continue;

        vector<uint64_t>::iterator it =
          std::find(point3D_ids.begin(), point3D_ids.end(), point3D_id);
        if (it != point3D_ids.end())
          continue;

        frame.add_feature_measurement(
          std::make_pair(point3D_id, Eigen::Vector2d(x, y)));

        point3D_ids.push_back(point3D_id);
        x_coords.insert_or_assign(x, j);
        y_coords.insert_or_assign(y, j);
      }
      result.push_back(frame);
    }
    return result;
  }

  aligned_unordered_map<size_t, Vector3d> remove_unused_points(
    aligned_unordered_map<size_t, Vector3d> feature_points,
    vector<vision_data_t> vision) {
    aligned_unordered_map<size_t, Eigen::Vector3d> result;
    for (auto const data : vision) {
      auto point_id = data.point3d_id;
      auto a = data.timestamp;
      auto v = data.uv;
      auto it = result.find(point_id);
      if (it == result.end()) {
        result.insert({point_id, feature_points.find(point_id)->second});
      }
    }
    return result;
  }
}  // namespace inrol
