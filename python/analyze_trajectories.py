import argparse
import numpy as np

from utils import text

import matplotlib.pyplot as plt


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gt", dest="gt", help="ground turth file", required=True
    )
    parser.add_argument(
        "--traj1", dest="traj1", help="trajectory file 1", required=True
    )
    parser.add_argument(
        "--traj2", dest="traj2", help="trajectory file 2", required=True
    )
    parser.add_argument(
        "--traj3", dest="traj3", help="trajectory file 3", required=True
    )
    parser.add_argument(
        "--traj4", dest="traj4", help="trajectory file 4", required=True
    )
    parser.add_argument(
        "--traj5", dest="traj5", help="trajectory file 5", required=True
    )
    parser.add_argument(
        "--start", dest="start", help="start time", required=True
    )
    parser.add_argument(
        "--end", dest="end", help="end time", required=True
    )
    return parser.parse_args()


def run():
    args = get_arguments()

    fig = plt.figure(figsize=(20, 10))
    ax = plt.axes(projection="3d")

    st = float(args.start)
    et = float(args.end)

    timestamps, ts, qs = text.readTimestampAndPose(args.gt)
    times = np.array(timestamps).reshape(-1, 1)
    ts = np.array(ts).reshape(-1, 3)
    ts = np.concatenate((times, ts), axis=1)
    ts = ts[ts[:, 0] < et, :]
    ts = ts[ts[:, 0] > st, :]
    x = ts[:, 1]
    y = ts[:, 2]
    z = ts[:, 3]
    ax.plot3D(x, y, z, 'k', linewidth=2.0, label="ground truth")

    timestamps, ts, qs = text.readTimestampAndPose(args.traj1)
    times = np.array(timestamps).reshape(-1, 1)
    ts = np.array(ts).reshape(-1, 3)
    ts = np.concatenate((times, ts), axis=1)
    ts = ts[ts[:, 0] < et, :]
    ts = ts[ts[:, 0] > st, :]
    x = ts[:, 1]
    y = ts[:, 2]
    z = ts[:, 3]
    ax.plot3D(x, y, z, color=(1, 0, 0), linewidth=2.0, alpha=0.5, label="A")

    timestamps, ts, qs = text.readTimestampAndPose(args.traj2)
    times = np.array(timestamps).reshape(-1, 1)
    ts = np.array(ts).reshape(-1, 3)
    ts = np.concatenate((times, ts), axis=1)
    ts = ts[ts[:, 0] < et, :]
    ts = ts[ts[:, 0] > st, :]
    x = ts[:, 1]
    y = ts[:, 2]
    z = ts[:, 3]
    ax.plot3D(x, y, z, '--', color=(0.5, 0, 0),
              linewidth=2.0, alpha=0.5, label="AH")

    timestamps, ts, qs = text.readTimestampAndPose(args.traj3)
    times = np.array(timestamps).reshape(-1, 1)
    ts = np.array(ts).reshape(-1, 3)
    ts = np.concatenate((times, ts), axis=1)
    ts = ts[ts[:, 0] < et, :]
    ts = ts[ts[:, 0] > st, :]
    x = ts[:, 1]
    y = ts[:, 2]
    z = ts[:, 3]
    ax.plot3D(x, y, z, color=(0, 1, 0), linewidth=2.0, alpha=0.5, label="B")

    timestamps, ts, qs = text.readTimestampAndPose(args.traj4)
    times = np.array(timestamps).reshape(-1, 1)
    ts = np.array(ts).reshape(-1, 3)
    ts = np.concatenate((times, ts), axis=1)
    ts = ts[ts[:, 0] < et, :]
    ts = ts[ts[:, 0] > st, :]
    x = ts[:, 1]
    y = ts[:, 2]
    z = ts[:, 3]
    ax.plot3D(x, y, z, '--', color=(0, 0.5, 0),
              linewidth=2.0, alpha=0.5, label="BH")

    timestamps, ts, qs = text.readTimestampAndPose(args.traj5)
    times = np.array(timestamps).reshape(-1, 1)
    ts = np.array(ts).reshape(-1, 3)
    ts = np.concatenate((times, ts), axis=1)
    ts = ts[ts[:, 0] < et, :]
    ts = ts[ts[:, 0] > st, :]
    x = ts[:, 1]
    y = ts[:, 2]
    z = ts[:, 3]
    ax.plot3D(x, y, z, '-', color=(0.0, 0.0, 1.0),
              linewidth=2.0, alpha=0.5, label="C")

    ax.set_aspect("equal")
    ax.legend()
    ax.view_init(elev=20, azim=-80)
    plt.show()


run()
