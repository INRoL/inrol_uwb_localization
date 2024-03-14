#pragma once

#include <vector>
#include <string>
#include <memory>

namespace inrol {
  using std::shared_ptr;
  using std::string;

  struct directory_name_t {
    string config;
    string parsed_cali_dataset;
    string parsed_anchor_dataset;
    string parsed_filter_dataset;
    string parsed_test_dataset;
    string model_calibration_result_A;
    string model_calibration_result_B;
    string model_calibration_result_C;
    string colmap_result;
    string parsed_colmap_result;
    string scale_free_matched_bspline;
    string initial_guess_scaled_state;
    string optimized_result_A;
    string optimized_result_B;
    string optimized_result_C;
    string anchor_prior;
    string kalman_filter_result_A;
    string kalman_filter_result_AH;
    string kalman_filter_result_B;
    string kalman_filter_result_BH;
    string kalman_filter_result_C;
  };

  shared_ptr<directory_name_t> get_directory_name(const string& filename);
}  // namespace inrol
