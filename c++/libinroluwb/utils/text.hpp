#pragma once

#include "libinroluwb/geometry/measurements.hpp"
#include "libinroluwb/geometry/pose.hpp"
#include "libinroluwb/optimizer/match.hpp"
#include "libinroluwb/optimizer/calibration.hpp"
#include "libinroluwb/optimizer/batch.hpp"
#include "libinroluwb/utils/param.hpp"
#include "libinroluwb/utils/binary.hpp"

#include <vector>

namespace inrol {
  using std::map;
  using std::string;
  using std::vector;

  vector<posestamped_t> get_poses_from_txt(string file_name);
  vector<imu_data_t> get_imu_data_from_txt(string file_name);
  vector<uwb_data_t> get_uwb_data_from_txt(string file_name);
  vector<double> get_image_timestamp_from_txt(string file_name);
  double get_optimized_scale_from_txt(string file_name);
  Vector3d get_gravity_vector_from_txt(string file_name);
  aligned_unordered_map<size_t, Vector3d> get_anchor_position_from_txt(
    string file_name);
  aligned_unordered_map<size_t, pose_t> get_anchor_poses_from_txt(
    string file_name);
  map<size_t, double> get_offset_from_txt(string file_name);
  vector<double> get_harmonics_coeffs_from_txt(string file_name);
  aligned_unordered_map<size_t, Vector3d> get_feature_points_from_txt(
    const string& file_name);
  uncertainty_param_t get_measurement_uncertainty_param(
    string file_name, ModelType type);
  pose_t get_pose_from_txt(string file_name);

  void write_posestamped_as_txt(
    const vector<posestamped_t> poses, const string& directory_name,
    const string& file_name);

  void write_poses_as_txt(
    const vector<pose_t> knots, const double dt, const string& directory_name,
    const string& file_name);

  void write_anchor_position_as_txt(
    aligned_unordered_map<size_t, Vector3d> anchor_position,
    const string& directory_name, const string& file_name);

  void write_anchor_poses_as_txt(
    aligned_unordered_map<size_t, pose_t> anchor_poses,
    const string& directory_name, const string& file_name);

  void write_scale_as_txt(
    const double scale, const string& directory_name, const string& file_name);

  void write_gravity_vector_as_txt(
    const Vector3d g, const string& directory_name, const string& file_name);

  void write_feature_points_as_txt(
    const aligned_unordered_map<size_t, Vector3d> feature_points,
    const string& directory_name, const string& file_name);

  void write_offset_as_txt(
    const vector<double> offset, const vector<size_t> used_anchor,
    const string& directory_name, const string& file_name);

  void write_vision_data_as_txt(
    const vector<vision_data_t> vision, const string& directory_name,
    const string& file_name);

  void write_bias_knots_as_txt(
    const vector<Vector3d> bias_knots, const double dt,
    const string& directory_name, const string& file_name);

  void write_harmonics_coeffs_as_txt(
    const vector<double> coeffs, const string& directory_name,
    const string& file_name);

  void write_uwb_error_as_txt(
    const vector<uwb_error_t> errors, const string& directory_name,
    const string& file_name);

  void write_model_param_as_txt(
    const uncertainty_param_t param, ModelType type,
    const string& directory_name, const string& file_name);

  void write_pose_as_txt(
    const pose_t pose, const string& directory_name, const string& file_name);
}  // namespace inrol
