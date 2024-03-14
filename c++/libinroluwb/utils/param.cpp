#include "libinroluwb/utils/param.hpp"

#include <yaml-cpp/yaml.h>

namespace inrol {
  using std::make_shared;

  Eigen::Matrix<double, 4, 4> convertYAMLto4x4Matrix(YAML::Node node) {
    Eigen::Matrix<double, 4, 4> result;
    result.row(0)
      << Eigen::Vector4d(node[0].as<vector<double>>().data()).transpose();
    result.row(1)
      << Eigen::Vector4d(node[1].as<vector<double>>().data()).transpose();
    result.row(2)
      << Eigen::Vector4d(node[2].as<vector<double>>().data()).transpose();
    result.row(3)
      << Eigen::Vector4d(node[3].as<vector<double>>().data()).transpose();
    return result;
  }

  Eigen::Matrix<double, 3, 3> convertYAMLto3x3Matrix(YAML::Node node) {
    Eigen::Matrix<double, 3, 3> result;
    result.row(0)
      << Eigen::Vector3d(node[0].as<vector<double>>().data()).transpose();
    result.row(1)
      << Eigen::Vector3d(node[1].as<vector<double>>().data()).transpose();
    result.row(2)
      << Eigen::Vector3d(node[2].as<vector<double>>().data()).transpose();
    return result;
  }

  sensor_noise_param_t get_sensor_noise_param(YAML::Node node) {
    return sensor_noise_param_t {
      .accel = node["accel"].as<double>(),
      .gyro = node["gyro"].as<double>(),
      .bias_a = node["bias_a"].as<double>(),
      .bias_w = node["bias_w"].as<double>(),
      .pixel = node["pixel"].as<double>(),
      .anchor = node["anchor"].as<double>(),
    };
  }
  camera_param_t get_camera_param(YAML::Node node) {
    return camera_param_t {
      .fx = node["fx"].as<double>(),
      .fy = node["fy"].as<double>(),
      .cx = node["cx"].as<double>(),
      .cy = node["cy"].as<double>(),
      .k1 = node["k1"].as<double>(),
      .k2 = node["k2"].as<double>(),
      .p1 = node["p1"].as<double>(),
      .p2 = node["p2"].as<double>(),
      .timeshift = node["timeshift"].as<double>(),
    };
  }
  spline_param_t get_spline_param(YAML::Node node) {
    return spline_param_t {
      .spline_pose_dt = node["spline_pose_dt"].as<double>(),
      .spline_bias_dt = node["spline_bias_dt"].as<double>(),
    };
  }
  harmonics_param_t get_harmonics_param(YAML::Node node) {
    return harmonics_param_t {
      .degree = node["degree"].as<size_t>(),
    };
  }
  anchor_param_t get_anchor_param(YAML::Node node) {
    return anchor_param_t {
      .used_anchor = node["used_anchor"].as<vector<size_t>>(),
    };
  }
  extrinsic_param_t get_extrinsic_param(YAML::Node node) {
    return extrinsic_param_t {
      .T_mocap_tag = convertYAMLto4x4Matrix(node["T_mocap_tag"]),
      .T_mocap_cam = convertYAMLto4x4Matrix(node["T_mocap_cam"]),
      .T_cam_imu = convertYAMLto4x4Matrix(node["T_cam_imu"]),
    };
  }

  shared_ptr<param_t> get_parameters(const string& filename) {
    const YAML::Node node = YAML::LoadFile(filename);

    return make_shared<param_t>(param_t {
      .sensor_noise = get_sensor_noise_param(node["sensor_noise"]),
      .camera = get_camera_param(node["camera"]),
      .spline = get_spline_param(node["spline"]),
      .harmonics = get_harmonics_param(node["harmonics"]),
      .anchor = get_anchor_param(node["anchor"]),
      .extrinsics = get_extrinsic_param(node["extrinsics"]),
    });
  }

  void param_t::print() {
    LOG(INFO) << "\n==========PARAMETERS=========="
              << "\n[sensor noise]"
              << "\n  - accel: " << sensor_noise.accel
              << "\n  - gyro: " << sensor_noise.gyro
              << "\n  - bias_a: " << sensor_noise.bias_a
              << "\n  - bias_w: " << sensor_noise.bias_w
              << "\n  - pixel: " << sensor_noise.pixel
              << "\n  - anchor : " << sensor_noise.anchor
              << "\n  - fx: " << camera.fx << "\n  - fy: " << camera.fy
              << "\n  - cx: " << camera.cx << "\n  - cy: " << camera.cy
              << "\n  - k1: " << camera.k1 << "\n  - k2: " << camera.k2
              << "\n  - p1: " << camera.p1 << "\n  - p2: " << camera.p2
              << "\n  - timeshift: " << camera.timeshift << "\n[spline]"
              << "\n  - spline_pose_dt: " << spline.spline_pose_dt
              << "\n  - spline_bias_dt: " << spline.spline_bias_dt
              << "\n[harmonics]"
              << "\n  - degree: " << harmonics.degree;
  }

  double get_alpha(asymmetric_param_t param) {
    double s = param.sigma;
    double g = param.gamma;
    double numerator = 2 * M_PI * g;
    double denominator = std::sqrt(2 * M_PI) * s + M_PI * g;
    return numerator / denominator;
  }

  aligned_unordered_map<size_t, Eigen::Matrix<double, 3, 3>> get_orientations(
    const string& filename) {
    aligned_unordered_map<size_t, Eigen::Matrix<double, 3, 3>> result;

    const YAML::Node node = YAML::LoadFile(filename);
    result[1] = convertYAMLto3x3Matrix(node["anchor1"]);
    result[2] = convertYAMLto3x3Matrix(node["anchor2"]);
    result[3] = convertYAMLto3x3Matrix(node["anchor3"]);
    result[4] = convertYAMLto3x3Matrix(node["anchor4"]);
    result[5] = convertYAMLto3x3Matrix(node["anchor5"]);
    result[6] = convertYAMLto3x3Matrix(node["anchor6"]);
    result[7] = convertYAMLto3x3Matrix(node["anchor7"]);
    result[8] = convertYAMLto3x3Matrix(node["anchor8"]);
    return result;
  }
}  // namespace inrol
