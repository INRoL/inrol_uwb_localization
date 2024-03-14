#include "libinroluwb/utils/text.hpp"
#include "libinroluwb/geometry/pose.hpp"
#include "libinroluwb/utils/binary.hpp"

#include <glog/logging.h>
#include <filesystem>
#include <fstream>

namespace inrol {
  vector<double> split(string str) {
    vector<double> result;
    std::stringstream ss(str);
    string tmp;

    while (std::getline(ss, tmp, ' ')) {
      double d;
      std::stringstream tmpss(tmp);
      tmpss >> d;
      result.push_back(d);
    }
    return result;
  }

  vector<vector<double>> get_double_vvector(string fn) {
    vector<vector<double>> result;

    std::ifstream in(fn);
    string line;
    string buffer;
    char delimiter = ' ';
    bool init_flag = true;
    while (in) {
      std::getline(in, line);
      if (init_flag) {
        init_flag = false;
        continue;
      }
      if (line.empty()) {
        continue;
      }
      result.push_back(split(line));
    }

    return result;
  }

  posestamped_t posestamped_from_double_vector(vector<double> dv) {
    CHECK(dv.size() == 8) << "pose txt column size not mathced";
    double timestamp = dv.at(0);
    double tx = dv.at(1);
    double ty = dv.at(2);
    double tz = dv.at(3);
    double qx = dv.at(4);
    double qy = dv.at(5);
    double qz = dv.at(6);
    double qw = dv.at(7);
    Vector3d v(tx, ty, tz);
    Quaterniond q(qw, qx, qy, qz);
    pose_t p {.p = v, .q = q};
    return posestamped_t {.pose = p, .timestamp = timestamp};
  }

  imu_data_t imu_data_from_double_vector(vector<double> dv) {
    CHECK(dv.size() == 8) << "pose txt column size not mathced";
    return imu_data_t {
      .timestamp = dv.at(1),
      .wx = dv.at(2),
      .wy = dv.at(3),
      .wz = dv.at(4),
      .ax = dv.at(5),
      .ay = dv.at(6),
      .az = dv.at(7),
    };
  }

  uwb_data_t uwb_data_from_double_vector(vector<double> dv) {
    CHECK(dv.size() == 4) << "pose txt column size not mathced";
    return uwb_data_t {
      .timestamp = dv.at(1),
      .anchor_idx = size_t(dv.at(2)),
      .distance = dv.at(3),
    };
  }

  vector<posestamped_t> get_poses_from_txt(const string file_name) {
    vector<posestamped_t> result;

    std::filesystem::path p(file_name);
    CHECK(std::filesystem::exists(p)) << file_name << " does not exist!";

    auto dvv = get_double_vvector(file_name);
    for (auto const& dv : dvv) {
      result.push_back(posestamped_from_double_vector(dv));
    }
    return result;
  }

  double get_optimized_scale_from_txt(string file_name) {
    std::filesystem::path p(file_name);
    CHECK(std::filesystem::exists(p)) << file_name << " does not exist!";

    auto dvv = get_double_vvector(file_name);
    return dvv[0][0];
  }

  vector<imu_data_t> get_imu_data_from_txt(string file_name) {
    vector<imu_data_t> result;

    std::filesystem::path p(file_name);
    CHECK(std::filesystem::exists(p)) << file_name << " does not exist!";

    auto dvv = get_double_vvector(file_name);
    for (auto const& dv : dvv) {
      result.push_back(imu_data_from_double_vector(dv));
    }
    return result;
  }

  vector<uwb_data_t> get_uwb_data_from_txt(string file_name) {
    vector<uwb_data_t> result;

    std::filesystem::path p(file_name);
    CHECK(std::filesystem::exists(p)) << file_name << " does not exist!";

    auto dvv = get_double_vvector(file_name);
    for (auto const& dv : dvv) {
      result.push_back(uwb_data_from_double_vector(dv));
    }
    return result;
  }

  vector<double> get_image_timestamp_from_txt(string file_name) {
    vector<double> result;

    std::filesystem::path p(file_name);
    CHECK(std::filesystem::exists(p)) << file_name << " does not exist!";

    auto dvv = get_double_vvector(file_name);
    for (auto const& dv : dvv) {
      result.push_back(dv[1]);
    }

    return result;
  }

  Vector3d get_gravity_vector_from_txt(string file_name) {
    std::filesystem::path p(file_name);
    CHECK(std::filesystem::exists(p)) << file_name << " does not exist!";

    auto dvv = get_double_vvector(file_name);
    return Vector3d(dvv[0][0], dvv[0][1], dvv[0][2]);
  }

  aligned_unordered_map<size_t, Vector3d> get_anchor_position_from_txt(
    string file_name) {
    aligned_unordered_map<size_t, Vector3d> result;

    std::filesystem::path p(file_name);
    CHECK(std::filesystem::exists(p)) << file_name << " does not exist!";

    auto dvv = get_double_vvector(file_name);
    for (auto const& dv : dvv) {
      result[static_cast<size_t>(dv[0])] = Vector3d(dv[1], dv[2], dv[3]);
    }
    return result;
  }

  aligned_unordered_map<size_t, pose_t> get_anchor_poses_from_txt(
    string file_name) {
    aligned_unordered_map<size_t, pose_t> result;

    std::filesystem::path p(file_name);
    CHECK(std::filesystem::exists(p)) << file_name << " does not exist!";

    auto dvv = get_double_vvector(file_name);
    for (auto const& dv : dvv) {
      double tx = dv.at(1);
      double ty = dv.at(2);
      double tz = dv.at(3);
      double qx = dv.at(4);
      double qy = dv.at(5);
      double qz = dv.at(6);
      double qw = dv.at(7);

      result[static_cast<size_t>(dv[0])] = pose_t {
        .p = Vector3d(tx, ty, tz),
        .q = Quaterniond(qw, qx, qy, qz),
      };
    }
    return result;
  }

  map<size_t, double> get_offset_from_txt(string file_name) {
    map<size_t, double> result;

    std::filesystem::path p(file_name);
    CHECK(std::filesystem::exists(p)) << file_name << " does not exist!";

    auto dvv = get_double_vvector(file_name);
    for (auto const& dv : dvv) {
      result[static_cast<size_t>(dv[0])] = dv[1];
    }
    return result;
  }

  vector<double> get_harmonics_coeffs_from_txt(string file_name) {
    vector<double> result;

    std::filesystem::path p(file_name);
    CHECK(std::filesystem::exists(p)) << file_name << " does not exist!";

    auto dvv = get_double_vvector(file_name);
    for (auto const& dv : dvv) {
      result.push_back(dv[1]);
    }

    return result;
  }

  aligned_unordered_map<size_t, Vector3d> get_feature_points_from_txt(
    const string& file_name) {
    aligned_unordered_map<size_t, Vector3d> result;

    std::filesystem::path p(file_name);
    CHECK(std::filesystem::exists(p)) << file_name << " does not exist!";

    auto dvv = get_double_vvector(file_name);
    for (auto const& dv : dvv) {
      result[static_cast<size_t>(dv[0])] = Vector3d(dv[1], dv[2], dv[3]);
    }

    return result;
  }

  uncertainty_param_t get_measurement_uncertainty_param(
    string file_name, ModelType type) {
    uncertainty_param_t result;

    std::filesystem::path p(file_name);
    CHECK(std::filesystem::exists(p)) << file_name << " does not exist!";

    auto dvv = get_double_vvector(file_name);
    for (auto const& dv : dvv) {
      switch (type) {
      case ModelType::A:
      case ModelType::AH:
        result = gaussian_param_t {
          .sigma = dv[0],
        };
        break;
      case ModelType::B:
      case ModelType::BH:
        result = gaussian_param_t {
          .sigma = dv[0],
        };
        break;
      case ModelType::C:
        result = asymmetric_param_t {
          .sigma = dv[0],
          .gamma = dv[1],
        };
        break;
      default:
        LOG(ERROR) << "Not supported anchor calibration type";
        break;
      }
    }

    return result;
  }

  pose_t get_pose_from_txt(string file_name) {
    pose_t result;

    std::filesystem::path p(file_name);
    CHECK(std::filesystem::exists(p)) << file_name << " does not exist!";

    auto dvv = get_double_vvector(file_name);
    auto dv = dvv[0];
    double tx = dv.at(0);
    double ty = dv.at(1);
    double tz = dv.at(2);
    double qx = dv.at(3);
    double qy = dv.at(4);
    double qz = dv.at(5);
    double qw = dv.at(6);
    Vector3d position(tx, ty, tz);
    Quaterniond quaternion(qw, qx, qy, qz);

    return pose_t {.p = position, .q = quaternion};
  }

  std::filesystem::path create_outfile(const string& path, const string file) {
    std::filesystem::path p(path);
    if (!std::filesystem::exists(p)) {
      std::filesystem::create_directory(p);
    }
    p = std::filesystem::path(path + file);
    if (std::filesystem::exists(p)) {
      std::filesystem::remove(p);
    }
    return p;
  }

  void write_posestamped_as_txt(
    const vector<posestamped_t> poses, const string& directory_name,
    const string& file_name) {
    auto p = create_outfile(directory_name, file_name);
    std::ofstream ofile(p);
    if (ofile.is_open()) {
      ofile << "# timestamp tx ty tz qx qy qz qw\n";
      int i = 0;
      for (auto const posestamped : poses) {
        double t = posestamped.timestamp;
        auto pose = posestamped.pose;
        auto position = pose.p;
        auto rotation = pose.q;
        double tx = position.x();
        double ty = position.y();
        double tz = position.z();
        double qx = rotation.x();
        double qy = rotation.y();
        double qz = rotation.z();
        double qw = rotation.w();
        ofile << std::fixed << std::setprecision(12) << t << " " << tx << " "
              << ty << " " << tz << " " << qx << " " << qy << " " << qz << " "
              << qw << "\n";
        i++;
      }
    }

    LOG(INFO) << "write " << poses.size() << " posestamped at " << p.string();
  }

  void write_poses_as_txt(
    const vector<pose_t> knots, const double dt, const string& directory_name,
    const string& file_name) {
    auto p = create_outfile(directory_name, file_name);
    std::ofstream ofile(p);
    if (ofile.is_open()) {
      ofile << "# timestamp tx ty tz qx qy qz qw\n";
      int i = 0;
      for (auto const knot : knots) {
        double t = (i - 1) * dt;
        auto position = knot.p;
        auto rotation = knot.q;
        double tx = position.x();
        double ty = position.y();
        double tz = position.z();
        double qx = rotation.x();
        double qy = rotation.y();
        double qz = rotation.z();
        double qw = rotation.w();
        ofile << t << " " << tx << " " << ty << " " << tz << " " << qx << " "
              << qy << " " << qz << " " << qw << "\n";
        i++;
      }
    }
  }

  void write_anchor_position_as_txt(
    aligned_unordered_map<size_t, Vector3d> anchor_positions,
    const string& directory_name, const string& file_name) {
    auto p = create_outfile(directory_name, file_name);
    std::ofstream ofile(p);
    if (ofile.is_open()) {
      ofile << "# anchor_idx x y z\n";
      for (auto const anchor_position : anchor_positions) {
        size_t anchor_idx = anchor_position.first;
        Vector3d position = anchor_position.second;
        ofile << std::fixed << std::setprecision(12) << anchor_idx << " "
              << position.x() << " " << position.y() << " " << position.z()
              << "\n";
      }
    }

    LOG(INFO) << "write " << anchor_positions.size() << " anchor positions at "
              << p.string();
  }

  void write_anchor_poses_as_txt(
    aligned_unordered_map<size_t, pose_t> anchor_poses,
    const string& directory_name, const string& file_name) {
    auto p = create_outfile(directory_name, file_name);
    std::ofstream ofile(p);
    if (ofile.is_open()) {
      ofile << "# anchor_idx tx ty tz qx qy qz qw\n";
      for (auto const anchor_pose : anchor_poses) {
        size_t anchor_idx = anchor_pose.first;
        pose_t pose = anchor_pose.second;
        ofile << std::fixed << std::setprecision(12) << anchor_idx << " "
              << pose.p.x() << " " << pose.p.y() << " " << pose.p.z() << " "
              << pose.q.x() << " " << pose.q.y() << " " << pose.q.z() << " "
              << pose.q.w() << "\n";
      }
    }

    LOG(INFO) << "write " << anchor_poses.size() << " anchor poses at "
              << p.string();
  }

  void write_scale_as_txt(
    const double scale, const string& directory_name, const string& file_name) {
    auto p = create_outfile(directory_name, file_name);
    std::ofstream ofile(p);
    if (ofile.is_open()) {
      ofile << "# s\n";
      ofile << std::fixed << std::setprecision(12) << scale;
    }

    LOG(INFO) << "write scale " << scale << " at " << p.string();
  }

  void write_gravity_vector_as_txt(
    const Vector3d g, const string& directory_name, const string& file_name) {
    auto p = create_outfile(directory_name, file_name);
    std::ofstream ofile(p);
    if (ofile.is_open()) {
      ofile << "# gx gy gz\n";
      ofile << std::fixed << std::setprecision(12) << g.x() << " " << g.y()
            << " " << g.z();
    }

    LOG(INFO) << "write gravity vector " << g.x() << ", " << g.y() << ", "
              << g.z() << " at " << p.string();
  }

  void write_feature_points_as_txt(
    const aligned_unordered_map<size_t, Vector3d> feature_points,
    const string& directory_name, const string& file_name) {
    auto p = create_outfile(directory_name, file_name);
    std::ofstream ofile(p);
    if (ofile.is_open()) {
      ofile << "# id tx ty tz\n";
      for (auto const& f : feature_points) {
        ofile << std::fixed << std::setprecision(12) << f.first << " "
              << f.second.x() << " " << f.second.y() << " " << f.second.z()
              << "\n";
      }
    }

    LOG(INFO) << "write " << feature_points.size() << " points at "
              << p.string();
  }

  void write_offset_as_txt(
    const vector<double> offset, const vector<size_t> used_anchor,
    const string& directory_name, const string& file_name) {
    CHECK(used_anchor.size() == offset.size()) << "";
    auto p = create_outfile(directory_name, file_name);
    std::ofstream ofile(p);
    if (ofile.is_open()) {
      ofile << "# anchor_idx offset\n";
      for (size_t i = 0; i < used_anchor.size(); i++) {
        size_t anchor_idx = used_anchor.at(i);
        ofile << std::fixed << std::setprecision(12) << anchor_idx << " "
              << offset.at(i) << "\n";
      }
    }

    LOG(INFO) << "write " << used_anchor.size() << " anchor offset at "
              << p.string();
  }

  void write_vision_data_as_txt(
    const vector<vision_data_t> vision, const string& directory_name,
    const string& file_name) {
    auto p = create_outfile(directory_name, file_name);
    std::ofstream ofile(p);
    if (ofile.is_open()) {
      ofile << "# timestamp point_id u v\n";
      for (auto const& data : vision) {
        ofile << std::fixed << std::setprecision(12) << data.timestamp << " "
              << data.point3d_id << " " << data.uv[0] << " " << data.uv[1]
              << "\n";
      }
    }
    LOG(INFO) << "write " << vision.size() << " vision observation at "
              << p.string();
  }

  void write_bias_knots_as_txt(
    const vector<Vector3d> bias_knots, const double dt,
    const string& directory_name, const string& file_name) {
    auto p = create_outfile(directory_name, file_name);
    std::ofstream ofile(p);
    int i = -1;
    if (ofile.is_open()) {
      ofile << "# timestamp point_id u v\n";
      for (auto const& knot : bias_knots) {
        double t = i * dt;
        ofile << std::fixed << std::setprecision(12) << t << " " << knot.x()
              << " " << knot.y() << " " << knot.z() << "\n";
        i++;
      }
    }

    LOG(INFO) << "write " << bias_knots.size() << " bias knots at "
              << p.string();
  }

  void write_harmonics_coeffs_as_txt(
    const vector<double> coeffs, const string& directory_name,
    const string& file_name) {
    auto p = create_outfile(directory_name, file_name);
    std::ofstream ofile(p);
    int i = 0;
    if (ofile.is_open()) {
      ofile << "# idx coeff\n";
      for (auto const& coeff : coeffs) {
        ofile << std::fixed << std::setprecision(12) << i << " " << coeff
              << "\n";
        i++;
      }
    }

    LOG(INFO) << "write " << coeffs.size() << " coefficients at " << p.string();
  }

  void write_uwb_error_as_txt(
    const vector<uwb_error_t> errors, const string& directory_name,
    const string& file_name) {
    auto p = create_outfile(directory_name, file_name);
    std::ofstream ofile(p);
    double squared_error_sum = 0;
    if (ofile.is_open()) {
      ofile << "# timestamp anchor_idx error\n";
      for (auto const& error : errors) {
        ofile << std::fixed << std::setprecision(12) << error.timestamp << " "
              << error.anchor_idx << " " << error.error << "\n";
        squared_error_sum += error.error * error.error;
      }
    }

    LOG(INFO) << "write " << errors.size() << " errors at " << p.string();
    LOG(INFO) << "RMSE: " << std::sqrt(squared_error_sum / errors.size());
  }

  void write_model_param_as_txt(
    const uncertainty_param_t param, ModelType type,
    const string& directory_name, const string& file_name) {
    auto p = create_outfile(directory_name, file_name);
    std::ofstream ofile(p);
    if (ofile.is_open()) {
      if (type == ModelType::A) {
        ofile << "sigma\n";
        auto sg = std::get<gaussian_param_t>(param);
        ofile << sg.sigma << "\n";
      } else if (type == ModelType::B) {
        ofile << "sigma\n";
        auto dg = std::get<gaussian_param_t>(param);
        ofile << dg.sigma << "\n";
      } else if (type == ModelType::C) {
        ofile << "sigma gamma\n";
        auto da = std::get<asymmetric_param_t>(param);
        ofile << da.sigma << " " << da.gamma << "\n";
      } else {
        LOG(INFO) << "Not supported calibration type";
      }
    }
    LOG(INFO) << "write uncertainty param at " << p.string();
  }

  void write_pose_as_txt(
    const pose_t pose, const string& directory_name, const string& file_name) {
    auto p = create_outfile(directory_name, file_name);
    std::ofstream ofile(p);
    if (ofile.is_open()) {
      ofile << "# tx ty tz qx qy qz qw\n";
      ofile << std::fixed << std::setprecision(12) << pose.p.x() << " "
            << pose.p.y() << " " << pose.p.z() << " " << pose.q.x() << " "
            << pose.q.y() << " " << pose.q.z() << " " << pose.q.w() << "\n";
    }
    LOG(INFO) << "write a pose at " << p.string();
  }
}  // namespace inrol
