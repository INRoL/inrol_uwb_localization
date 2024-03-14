#include "libinroluwb/utils/directory.hpp"

#include <yaml-cpp/yaml.h>
#include <glog/logging.h>

namespace inrol {
  using std::make_shared;

  string get_string_from_yaml_node(YAML::Node node, const string& name) {
    return node[name].as<string>();
  }

  shared_ptr<directory_name_t> get_directory_name(const string& filename) {
    const YAML::Node node = YAML::LoadFile(filename);

    return make_shared<directory_name_t>(directory_name_t {
      .config = get_string_from_yaml_node(node, "config"),
      .parsed_cali_dataset =
        get_string_from_yaml_node(node, "parsed_cali_dataset"),
      .parsed_anchor_dataset =
        get_string_from_yaml_node(node, "parsed_anchor_dataset"),
      .parsed_filter_dataset =
        get_string_from_yaml_node(node, "parsed_filter_dataset"),
      .parsed_test_dataset =
        get_string_from_yaml_node(node, "parsed_test_dataset"),
      .model_calibration_result_A =
        get_string_from_yaml_node(node, "model_calibration_result_A"),
      .model_calibration_result_B =
        get_string_from_yaml_node(node, "model_calibration_result_B"),
      .model_calibration_result_C =
        get_string_from_yaml_node(node, "model_calibration_result_C"),
      .colmap_result = get_string_from_yaml_node(node, "colmap_result"),
      .parsed_colmap_result =
        get_string_from_yaml_node(node, "parsed_colmap_result"),
      .scale_free_matched_bspline =
        get_string_from_yaml_node(node, "scale_free_matched_bspline"),
      .initial_guess_scaled_state =
        get_string_from_yaml_node(node, "initial_guess_scaled_state"),
      .optimized_result_A =
        get_string_from_yaml_node(node, "optimized_result_A"),
      .optimized_result_B =
        get_string_from_yaml_node(node, "optimized_result_B"),
      .optimized_result_C =
        get_string_from_yaml_node(node, "optimized_result_C"),
      .anchor_prior = get_string_from_yaml_node(node, "anchor_prior"),
      .kalman_filter_result_A =
        get_string_from_yaml_node(node, "kalman_filter_result_A"),
      .kalman_filter_result_AH =
        get_string_from_yaml_node(node, "kalman_filter_result_AH"),
      .kalman_filter_result_B =
        get_string_from_yaml_node(node, "kalman_filter_result_B"),
      .kalman_filter_result_BH =
        get_string_from_yaml_node(node, "kalman_filter_result_BH"),
      .kalman_filter_result_C =
        get_string_from_yaml_node(node, "kalman_filter_result_C"),
    });
  }
}  // namespace inrol
