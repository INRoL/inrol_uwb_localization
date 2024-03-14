#include "libinroluwb/optimizer/calibration.hpp"
#include "libinroluwb/optimizer/logging.hpp"
#include "libinroluwb/optimizer/functor/uwb.hpp"
#include "libinroluwb/geometry/pose.hpp"
#include "libinroluwb/utils/algorithm.hpp"
#include "libinroluwb/utils/param.hpp"

#include <ceres/cost_function.h>
#include <ceres/loss_function.h>
#include <ceres/manifold.h>
#include <ceres/types.h>

namespace inrol {
  ModelType string2modeltype(std::string arg) {
    std::map<std::string, ModelType> m;
    m["A"] = ModelType::A;
    m["AH"] = ModelType::AH;
    m["B"] = ModelType::B;
    m["BH"] = ModelType::BH;
    m["C"] = ModelType::C;
    return m[arg];
  }

  ModelCalibrationOptimizer::ModelCalibrationOptimizer(
    const model_cali_dataset_t& dataset, shared_ptr<param_t> param,
    ModelType type)
      : _dataset(dataset),
        _used_anchor(param->anchor.used_anchor),
        _degree(param->harmonics.degree),
        _noise(param->sensor_noise),
        _type(type) {
    _anchor_number = _used_anchor.size();

    generate_dataset();
  }

  optional<size_t> ModelCalibrationOptimizer::find_tag_index_from_timestamp(
    double ts) {
    size_t idx = 0;
    for (size_t j = 0; j < _dataset.tag_poses.size() - 1; j++) {
      if (_dataset.tag_poses.at(j + 1).timestamp > ts) {
        idx = j;
        break;
      }
    }
    if (idx == 0) {
      return std::nullopt;
    }
    return idx;
  }

  void ModelCalibrationOptimizer::generate_dataset() {
    for (auto const& uwb : _dataset.uwb_dataset) {
      double ts = uwb.timestamp;

      auto maybe_idx = find_tag_index_from_timestamp(ts);
      if (!maybe_idx)
        continue;

      posestamped_t p0 = _dataset.tag_poses.at(maybe_idx.value());
      posestamped_t p1 = _dataset.tag_poses.at(maybe_idx.value() + 1);

      auto tag_pose = pose_interpolation(p0, p1, ts);
      auto anchor_pose = _dataset.anchor_poses.at(uwb.anchor_idx);
      _aligned_dataset.push_back(cali_data_t {
        .tag_pose = tag_pose.pose,
        .anchor_pose = anchor_pose,
        .uwb = uwb,
      });
    }
  }

  bool ModelCalibrationOptimizer::solve() {
    ceres::GradientProblemSolver::Options options;
    options.max_num_iterations = 300;
    options.update_state_every_iteration = true;
    LoggingCallback logging_callback(true);
    options.callbacks.push_back(&logging_callback);
    ceres::GradientProblemSolver::Summary summary;

    if (_type == ModelType::A) {
      ceres::FirstOrderFunction* sg_cost_func3 =
        ModelACaliCost::Create(_aligned_dataset);
      ceres::GradientProblem _problem3(sg_cost_func3);

      double x[1] = {_s};
      ceres::Solve(options, _problem3, x, &summary);
      _s = x[0];
    } else if (_type == ModelType::B) {
      ceres::FirstOrderFunction* sg_cost_func3 =
        ModelBCaliCost::Create(_aligned_dataset, _degree);
      ceres::GradientProblem _problem3(sg_cost_func3);

      double x[201] = {0};
      x[0] = _s;
      x[1] = _tag_coeffs[0];
      for (int i = 1; i < (_degree + 1) * (_degree + 1); i++) {
        x[i + 1] = _tag_coeffs[i];
        x[i + (_degree + 1) * (_degree + 1)] = _anchor_coeffs[i];
      }
      ceres::Solve(options, _problem3, x, &summary);
      _s = x[0];
      _tag_coeffs[0] = x[1];
      for (int i = 1; i < (_degree + 1) * (_degree + 1); i++) {
        _tag_coeffs[i] = x[i + 1];
        _anchor_coeffs[i] = x[i + (_degree + 1) * (_degree + 1)];
      }
    } else if (_type == ModelType::C) {
      ceres::FirstOrderFunction* sg_cost_func3 =
        ModelCCaliCost::Create(_aligned_dataset, _degree);
      ceres::GradientProblem _problem3(sg_cost_func3);

      double x[202] = {0};
      x[0] = _s;
      x[1] = _g;
      x[2] = _tag_coeffs[0];
      for (int i = 1; i < (_degree + 1) * (_degree + 1); i++) {
        x[i + 2] = _tag_coeffs[i];
        x[i + (_degree + 1) * (_degree + 1) + 1] = _anchor_coeffs[i];
      }
      ceres::Solve(options, _problem3, x, &summary);
      _s = x[0];
      _g = x[1];
      _tag_coeffs[0] = x[2];
      for (int i = 1; i < (_degree + 1) * (_degree + 1); i++) {
        _tag_coeffs[i] = x[i + 2];
        _anchor_coeffs[i] = x[i + (_degree + 1) * (_degree + 1) + 1];
      }
    } else {
      LOG(INFO) << "Not supported calibration type";
    }
    LOG(INFO) << summary.FullReport();
    return summary.IsSolutionUsable();
  }

  uncertainty_param_t ModelCalibrationOptimizer::get_uncertainty_param() {
    uncertainty_param_t result;
    if (_type == ModelType::A) {
      LOG(INFO) << "Model A Calibration - sigma: " << _s;
      result = gaussian_param_t {
        .sigma = _s,
      };
    } else if (_type == ModelType::B) {
      LOG(INFO) << "Model B Calibration - sigma: " << _s;
      result = gaussian_param_t {
        .sigma = _s,
      };
    } else if (_type == ModelType::C) {
      LOG(INFO) << "Model C Calibration - sigma: " << _s << ", gamma: " << _g;
      result = asymmetric_param_t {
        .sigma = _s,
        .gamma = _g,
      };
    } else {
      LOG(INFO) << "Not supported calibration type";
    }

    return result;
  }

  vector<double> ModelCalibrationOptimizer::get_tag_coeffs() {
    vector<double> v;
    for (int i = 0; i < (_degree + 1) * (_degree + 1); i++) {
      v.push_back(_tag_coeffs[i]);
    }
    return v;
  }

  vector<double> ModelCalibrationOptimizer::get_anchor_coeffs() {
    vector<double> v;
    for (int i = 0; i < (_degree + 1) * (_degree + 1); i++) {
      v.push_back(_anchor_coeffs[i]);
    }
    return v;
  }
}  // namespace inrol
