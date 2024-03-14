#pragma once

#include "libinroluwb/geometry/feature.hpp"
#include "libinroluwb/geometry/camera.hpp"
#include "libinroluwb/geometry/measurements.hpp"

#include <unordered_map>
#include <Eigen/src/Core/util/Memory.h>

namespace inrol {
  using std::string;
  using std::vector;

  template <typename K, typename V>
  using aligned_unordered_map = std::unordered_map<
    K, V, std::hash<K>, std::equal_to<K>,
    Eigen::aligned_allocator<std::pair<K const, V>>>;

  Camera readCameraBinary(const string& path);

  aligned_unordered_map<size_t, Vector3d> readFeaturePointsBinary(
    const string& path);

  vector<Frame> readFramesBinary(const string& path);

  aligned_unordered_map<size_t, Vector3d> remove_unused_points(
    aligned_unordered_map<size_t, Vector3d> feature_points,
    vector<vision_data_t> vision);
}  // namespace inrol
