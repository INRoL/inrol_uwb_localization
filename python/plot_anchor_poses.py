import argparse
import numpy as np

from utils import text

import matplotlib.pyplot as plt
import pyquaternion
import math
from scipy.spatial.transform import Rotation


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gt_poses1", dest="gt_poses1", help="gt anchor poses1 file", required="true"
    )
    parser.add_argument(
        "--gt_poses2", dest="gt_poses2", help="gt anchor poses2 file", required="true"
    )
    parser.add_argument(
        "--poses2", dest="poses2", help="anchor poses2 file", required="true"
    )
    parser.add_argument(
        "--gt_poses3", dest="gt_poses3", help="gt anchor poses3 file", required="true"
    )
    parser.add_argument(
        "--poses3", dest="poses3", help="anchor poses3 file", required="true"
    )
    return parser.parse_args()


def plot_anchors(idx, x, y, z, qx, qy, qz, qw, length, ax):
    result = dict()
    for eid, ex, ey, ez, eqx, eqy, eqz, eqw, in zip(idx, x, y, z, qx, qy, qz, qw):
        ax.scatter(ex, ey, ez, linewidth=0.7, color=(0.0, 0.0, 0.0))
        R = pyquaternion.Quaternion(eqw, eqx, eqy, eqz).rotation_matrix
        r = length * R @ np.array([[1], [0], [0]])
        g = length * R @ np.array([[0], [1], [0]])
        b = length * R @ np.array([[0], [0], [1]])
        rx = [ex, ex + r[0][0]]
        ry = [ey, ey + r[1][0]]
        rz = [ez, ez + r[2][0]]
        gx = [ex, ex + g[0][0]]
        gy = [ey, ey + g[1][0]]
        gz = [ez, ez + g[2][0]]
        bx = [ex, ex + b[0][0]]
        by = [ey, ey + b[1][0]]
        bz = [ez, ez + b[2][0]]
        ax.plot3D(rx, ry, rz, linewidth=2.0, color="r")
        ax.plot3D(gx, gy, gz, linewidth=2.0, color="g")
        ax.plot3D(bx, by, bz, linewidth=2.0, color="b")
        result[eid] = (ex, ey, ez, eqx, eqy, eqz, eqw)
    ax.set_xlim(-2.3, 2.3)
    ax.set_ylim(-2.3, 2.3)
    ax.set_zlim(-0.5, 3)
    ax.set_aspect("equal")
    ax.view_init(elev=20, azim=-150)
    return result


def run():
    args = get_arguments()

    fig = plt.figure(figsize=(18, 12))
    ax1 = fig.add_subplot(1, 3, 1, projection='3d')
    ax2 = fig.add_subplot(1, 3, 2, projection='3d')
    ax3 = fig.add_subplot(1, 3, 3, projection='3d')

    idx, x, y, z, qx, qy, qz, qw = text.readAnchorPoses(args.gt_poses1)
    dict1 = plot_anchors(idx, x, y, z, qx, qy, qz, qw, 0.4, ax1)

    idx, x, y, z, qx, qy, qz, qw = text.readAnchorPoses(args.gt_poses2)
    dict2 = plot_anchors(idx, x, y, z, qx, qy, qz, qw, 0.4, ax2)

    idx, x, y, z, qx, qy, qz, qw = text.readAnchorPoses(args.poses2)
    dict3 = plot_anchors(idx, x, y, z, qx, qy, qz, qw, 0.2, ax2)

    idx, x, y, z, qx, qy, qz, qw = text.readAnchorPoses(args.gt_poses3)
    dict4 = plot_anchors(idx, x, y, z, qx, qy, qz, qw, 0.4, ax3)

    idx, x, y, z, qx, qy, qz, qw = text.readAnchorPoses(args.poses3)
    dict5 = plot_anchors(idx, x, y, z, qx, qy, qz, qw, 0.2, ax3)

    ep1_queue = []
    ep2_queue = []
    eo1_queue = []
    eo2_queue = []
    for eid in idx:
        (x1, y1, z1, qx1, qy1, qz1, qw1) = dict2[eid]
        (x2, y2, z2, qx2, qy2, qz2, qw2) = dict3[eid]
        (x3, y3, z3, qx3, qy3, qz3, qw3) = dict4[eid]
        (x4, y4, z4, qx4, qy4, qz4, qw4) = dict5[eid]
        ep = (x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2
        ep2 = (x3 - x4) ** 2 + (y3 - y4) ** 2 + (z3 - z4) ** 2
        q1 = pyquaternion.Quaternion(qw1, qx1, qy1, qz1)
        q2 = pyquaternion.Quaternion(qw2, qx2, qy2, qz2)
        R1 = q1.rotation_matrix
        R2 = q2.rotation_matrix
        dR = Rotation.from_matrix(R1 @ R2.T)
        theta = dR.as_rotvec()
        q1 = pyquaternion.Quaternion(qw3, qx3, qy3, qz3)
        q2 = pyquaternion.Quaternion(qw4, qx4, qy4, qz4)
        R1 = q1.rotation_matrix
        R2 = q2.rotation_matrix
        dR = Rotation.from_matrix(R1 @ R2.T)
        theta2 = dR.as_rotvec()
        ep1_queue.append(ep)
        ep2_queue.append(ep2)
        eo1_queue.append(np.sum(theta**2))
        eo2_queue.append(np.sum(theta2**2))

    print("(env1) anchor position RMSE: ")
    print(np.sqrt(np.mean(ep1_queue)))
    print("(env1) anchor orientation RMSE: ")
    print(np.sqrt(np.mean(eo1_queue)) * 180.0 / np.pi)
    print("(env1) anchor position RMSE: ")
    print(np.sqrt(np.mean(ep2_queue)))
    print("(env1) anchor orientation RMSE: ")
    print(np.sqrt(np.mean(eo2_queue)) * 180.0 / np.pi)
    plt.savefig('anchor_config.svg', format='svg')
    plt.show()


run()
