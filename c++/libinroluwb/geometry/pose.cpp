#include "libinroluwb/geometry/pose.hpp"
#include "libinroluwb/optimizer/math/spline.hpp"

namespace inrol {
  pose_t pose_t::product(pose_t pose) {
    return pose_t {.p = p + q.toRotationMatrix() * pose.p, .q = q * pose.q};
  }

  posestamped_t pose_interpolation(
    posestamped_t ps_a, posestamped_t ps_b, double ts) {
    if (ps_a.timestamp == ps_b.timestamp) {
      return ps_a;
    } else {
      CHECK(ps_a.timestamp <= ts) << ts << " is less than " << ps_a.timestamp;
      CHECK(ps_b.timestamp >= ts)
        << ts << " is greater than " << ps_b.timestamp;

      Vector3d pa = ps_a.pose.p;
      Vector3d pb = ps_b.pose.p;
      Quaterniond qa = ps_a.pose.q;
      Quaterniond qb = ps_b.pose.q;
      double t = (ts - ps_a.timestamp) / (ps_b.timestamp - ps_a.timestamp);

      Vector3d p = pa * (1 - t) + pb * t;
      Quaterniond q = qa.slerp(t, qb);
      return posestamped_t {.pose = pose_t {.p = p, .q = q}, .timestamp = ts};
    }
  }

  vector<posestamped_t> get_interpolated_knots(
    const vector<posestamped_t>& input_traj, double dt) {
    double duration = input_traj.back().timestamp;
    size_t knots_n = duration / dt + 4;
    size_t input_traj_size = input_traj.size();

    vector<posestamped_t> result;

    for (int i = 0; i < knots_n; i++) {
      double ts = dt * (i - 1);

      size_t front_idx;
      size_t back_idx;
      if (ts <= 0) {
        front_idx = 0;
        back_idx = 0;
      } else if (ts > duration) {
        front_idx = input_traj_size - 1;
        back_idx = input_traj_size - 1;
      } else {
        for (int j = 0; j < input_traj_size; j++) {
          if (input_traj.at(j).timestamp > ts) {
            front_idx = j - 1;
            back_idx = j;
            break;
          }
        }
      }

      posestamped_t posestamped_a = input_traj.at(front_idx);
      posestamped_t posestamped_b = input_traj.at(back_idx);

      result.push_back(pose_interpolation(posestamped_a, posestamped_b, ts));
    }
    result.front().timestamp = -dt;
    result.at(1).timestamp = 0.0;
    result.at(result.size() - 2).timestamp = double(result.size() - 3) * dt;
    result.back().timestamp = double(result.size() - 2) * dt;
    return result;
  }

  pose_t pose_b_spline(
    pose_t pose0, pose_t pose1, pose_t pose2, pose_t pose3, double u) {
    pose_t result;

    double u2 = u * u;
    double u3 = u2 * u;
    double b1 = 5.0 / 6.0 + u / 2.0 - u2 / 2.0 + u3 / 6.0;
    double b2 = 1.0 / 6.0 + u / 2.0 + u2 / 2.0 - u3 / 3.0;
    double b3 = u3 / 6.0;

    Quaterniond q0 = pose0.q;
    Quaterniond q1 = pose1.q;
    Quaterniond q2 = pose2.q;
    Quaterniond q3 = pose3.q;
    Vector3d p0 = pose0.p;
    Vector3d p1 = pose1.p;
    Vector3d p2 = pose2.p;
    Vector3d p3 = pose3.p;

    Vector3d qd01 = quaterniond_logarithm(q0.conjugate() * q1) * b1;
    Vector3d qd12 = quaterniond_logarithm(q1.conjugate() * q2) * b2;
    Vector3d qd23 = quaterniond_logarithm(q2.conjugate() * q3) * b3;
    Eigen::Quaterniond A1 = quaterniond_exponential<double>(qd01);
    Eigen::Quaterniond A2 = quaterniond_exponential<double>(qd12);
    Eigen::Quaterniond A3 = quaterniond_exponential<double>(qd23);
    Vector3d pd01 = (p1 - p0) * b1;
    Vector3d pd12 = (p2 - p1) * b2;
    Vector3d pd23 = (p3 - p2) * b3;

    result.q = q0 * A1 * A2 * A3;
    result.p = p0 + pd01 + pd12 + pd23;
    return result;
  }

  pose_t convert_eigen_to_pose_t(Matrix4d m) {
    Vector3d p = m.block<3, 1>(0, 3);
    Matrix3d R = m.block<3, 3>(0, 0);
    Quaterniond q = Quaterniond(R);
    return pose_t {
      .p = p,
      .q = q,
    };
  }
}  // namespace inrol
