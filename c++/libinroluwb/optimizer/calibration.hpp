#pragma once

#include "libinroluwb/geometry/measurements.hpp"
#include "libinroluwb/geometry/pose.hpp"
#include "libinroluwb/optimizer/math/spline.hpp"
#include "libinroluwb/utils/binary.hpp"
#include "libinroluwb/utils/param.hpp"

#include <Eigen/src/Core/Matrix.h>
#include <Eigen/src/Core/util/Memory.h>
#include <ceres/autodiff_cost_function.h>
#include <ceres/ceres.h>
#include <nlopt.hpp>

#include <map>
#include <optional>

namespace inrol {
  using std::optional;

  enum class ModelType {
    A,
    AH,
    B,
    BH,
    C,
  };

  struct cali_data_t {
    pose_t tag_pose;
    pose_t anchor_pose;
    uwb_data_t uwb;
  };

  struct model_cali_dataset_t {
    vector<posestamped_t> tag_poses;
    aligned_unordered_map<size_t, pose_t> anchor_poses;
    vector<uwb_data_t> uwb_dataset;
  };

  ModelType string2modeltype(std::string arg);

  class ModelCalibrationOptimizer {
  public:
    ModelCalibrationOptimizer(
      const model_cali_dataset_t& dataset, shared_ptr<param_t> param,
      ModelType type);

    bool solve();

    uncertainty_param_t get_uncertainty_param();
    vector<double> get_tag_coeffs();
    vector<double> get_anchor_coeffs();

  private:
    optional<size_t> find_tag_index_from_timestamp(double ts);
    void generate_dataset();

    model_cali_dataset_t _dataset;
    vector<size_t> _used_anchor;
    vector<cali_data_t> _aligned_dataset;

    size_t _anchor_number;
    int _degree;
    sensor_noise_param_t _noise;
    ModelType _type;

    double _tag_coeffs[100] = {0};
    double _anchor_coeffs[100] = {0};
    double _s = 0.1;
    double _g = 0.1;
  };
}  // namespace inrol
