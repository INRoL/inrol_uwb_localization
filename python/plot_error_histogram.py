import argparse
import numpy as np

from utils import text

import math
import pyquaternion

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import scipy


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--uwb", dest="uwb",
                        help="uwb data file", required="true")
    parser.add_argument("--tag_pose", dest="tag_pose",
                        help="tag pose data file", required="true")
    parser.add_argument("--anchor", dest="anchor",
                        help="anchor pose file", required="true")
    parser.add_argument("--a_dir", dest="a_dir",
                        help="A calibration result directory", required="true")
    parser.add_argument("--b_dir", dest="b_dir",
                        help="B calibration result directory", required="true")
    parser.add_argument("--c_dir", dest="c_dir",
                        help="C calibration result directory", required="true")
    return parser.parse_args()


def gaussian(x, sigma, alpha):
    return (2 - alpha) / np.sqrt(2 * np.pi * sigma**2) * np.exp(-(x**2) / (2 * sigma**2))


def cauchy(x, gamma, alpha):
    return alpha / (np.pi * gamma * (1 + (x**2) / (gamma**2)))


def index_to_degree(k):
    l = 0
    while True:
        if k < (l + 1) * (l + 1):
            break
        l = l + 1
    return l


def index_to_order(k, l):
    return (k - l * l) - l


def legendre(l, m, x):
    assert m >= 0
    assert m <= l

    Pmm = np.ones_like(x)
    if m > 0:
        somx2 = np.sqrt((1.0 - x) * (1.0 + x))
        for i in range(1, m + 1):
            Pmm = -Pmm * (2 * i - 1) * somx2

    if l == m:
        return Pmm

    Pmmp1 = x * (2.0 * m + 1.0) * Pmm
    if l == m + 1:
        return Pmmp1

    Plm = 0.0
    for ll in range(m + 2, l + 1):
        Plm = ((2.0 * ll - 1.0) * x * Pmmp1 - (ll + m - 1.0) * Pmm) / (ll - m)
        Pmm = Pmmp1
        Pmmp1 = Plm
    return Plm


def normal_term(l, m):
    factor = (
        (2.0 * l + 1.0) * scipy.special.gamma(l - m + 1) /
        (4.0 * np.pi * scipy.special.gamma(l + m + 1))
    )
    return math.sqrt(factor)


def get_real_spherical_hamonics(l, m, theta, phi):
    x = np.cos(theta)
    phase = m * phi
    sqrt2 = math.sqrt(2.0)

    sign = 1.0
    if m % 2 != 0:
        sign = -1.0

    if m == 0:
        Plm = legendre(l, m, x)
        N = normal_term(l, m)
        return N * Plm
    elif m > 0:
        Plm = legendre(l, m, x)
        N = normal_term(l, m)
        return sign * sqrt2 * N * Plm * np.cos(phase)
    else:
        Plm = legendre(l, -m, x)
        N = normal_term(l, -m)
        return sign * sqrt2 * N * Plm * np.sin(-phase)


def run():
    args = get_arguments()

    uts, uids, urs = text.readUWBData(args.uwb)
    tts, tt, tq = text.readTimestampAndPose(args.tag_pose)
    anchor_ids, axs, ays, azs, aqxs, aqys, aqzs, aqws = text.readAnchorPoses(
        args.anchor)
    a_sigma, _ = text.readModelParam(
        args.a_dir + "/uncertainty_param.txt", "A")
    b_sigma, _ = text.readModelParam(
        args.b_dir + "/uncertainty_param.txt", "B")
    c_sigma, d_gamma = text.readModelParam(
        args.c_dir + "/uncertainty_param.txt", "C")
    b_t = text.readSphHarmCoeffs(args.b_dir + "/tag_coeffs.txt")
    b_a = text.readSphHarmCoeffs(args.b_dir + "/anchor_coeffs.txt")
    c_t = text.readSphHarmCoeffs(args.c_dir + "/tag_coeffs.txt")
    c_a = text.readSphHarmCoeffs(args.c_dir + "/anchor_coeffs.txt")

    traj_timestamps = np.array(tts)

    a_error = []
    b_error = []
    c_error = []
    for ts, id, v in zip(uts, uids, urs):
        if ts < traj_timestamps[0]:
            continue
        if ts > traj_timestamps[-1]:
            continue

        traj_idx = len(traj_timestamps) - (np.sum(traj_timestamps > ts)) - 1
        ts0 = traj_timestamps[traj_idx]
        ts1 = traj_timestamps[traj_idx + 1]
        t = (ts - ts0) / (ts1 - ts0)

        p0 = np.array(tt[traj_idx])
        p1 = np.array(tt[traj_idx + 1])
        q0 = tq[traj_idx]
        q1 = tq[traj_idx + 1]
        q0 = pyquaternion.Quaternion(q0[3], q0[0], q0[1], q0[2])
        q1 = pyquaternion.Quaternion(q1[3], q1[0], q1[1], q1[2])

        tsp = (1.0 - t) * p0 + t * p1
        tsq = pyquaternion.Quaternion.slerp(q0, q1, amount=t)

        anchor_idx = anchor_ids.index(id)
        ax = axs[anchor_idx]
        ay = ays[anchor_idx]
        az = azs[anchor_idx]
        aqx = aqxs[anchor_idx]
        aqy = aqys[anchor_idx]
        aqz = aqzs[anchor_idx]
        aqw = aqws[anchor_idx]
        aq = pyquaternion.Quaternion(aqw, aqx, aqy, aqz)
        ap = np.array([ax, ay, az])
        ray_world_coord = (ap - tsp).reshape(3, 1)
        ray_tag_coord = tsq.rotation_matrix.T @ ray_world_coord
        ray_anc_coord = -aq.rotation_matrix.T @ ray_world_coord

        r = np.linalg.norm(ray_world_coord)

        tag_the = math.acos(ray_tag_coord[2] / r)
        tag_phi = math.atan2(ray_tag_coord[1], ray_tag_coord[0])
        anc_the = math.acos(ray_anc_coord[2] / r)
        anc_phi = math.atan2(ray_anc_coord[1], ray_anc_coord[0])

        b_bias = 0
        c_bias = 0

        for i, (bt, ba, ct, ca) in enumerate(zip(b_t, b_a, c_t, c_a)):
            if i == 0:
                b_bias = b_bias + bt * get_real_spherical_hamonics(0, 0, 0, 0)
                c_bias = c_bias + ct * get_real_spherical_hamonics(0, 0, 0, 0)
            else:
                l = index_to_degree(i)
                m = index_to_order(i, l)

                tag_v = get_real_spherical_hamonics(l, m, tag_the, tag_phi)
                anc_v = get_real_spherical_hamonics(l, m, anc_the, anc_phi)
                b_bias = b_bias + bt * tag_v + ba * anc_v
                c_bias = c_bias + ct * tag_v + ca * anc_v

        a_error.append(v - r)
        b_error.append(v - r - b_bias)
        c_error.append(v - r - c_bias)

    minx = -0.5
    maxx = 0.5
    a_count, a_bin = np.histogram(a_error, bins=100, range=(minx, maxx))
    b_count, b_bin = np.histogram(b_error, bins=100, range=(minx, maxx))
    c_count, c_bin = np.histogram(c_error, bins=100, range=(minx, maxx))

    x = np.arange(minx, maxx, 0.01)
    a_y = gaussian(x, a_sigma, 1)
    b_y = gaussian(x, b_sigma, 1)
    xl = np.arange(minx, 0.0, 0.01)
    xr = np.arange(-0.0, maxx, 0.01)
    alpha = (2 * np.pi * d_gamma) / \
        (np.sqrt(2 * np.pi) * c_sigma + np.pi * d_gamma)
    c_yl = gaussian(xl, c_sigma, alpha)
    c_yr = cauchy(xr, d_gamma, alpha)
    c_y = np.concatenate((c_yl, c_yr), axis=None)

    fig = plt.figure(figsize=(6, 6))
    ax1 = plt.subplot(311)
    ax2 = plt.subplot(312)
    ax3 = plt.subplot(313)
    plt.tight_layout()
    ax1.hist(
        a_bin[:-1],
        a_bin,
        weights=a_count,
        density=True,
        alpha=0.5,
        color=(1, 0, 0),
    )
    ax2.hist(
        b_bin[:-1],
        b_bin,
        weights=b_count,
        density=True,
        alpha=0.5,
        color=(0, 1, 0),
    )
    ax3.hist(
        c_bin[:-1],
        c_bin,
        weights=c_count,
        density=True,
        alpha=0.5,
        color=(0, 0, 1),
    )
    ax1.plot(x, a_y, '--', color=(0, 0, 0),
             linewidth=2.0, label='calibrated pdf')
    ax2.plot(x, b_y, '--', color=(0, 0, 0),
             linewidth=2.0, label='calibrated pdf')
    ax3.plot(x, c_y, '--', color=(0, 0, 0),
             linewidth=2.0, label='calibrated pdf')
    ax1.set_xlim([minx, maxx])
    ax2.set_xlim([minx, maxx])
    ax3.set_xlim([minx, maxx])
    ax1.set_ylim([0, 6.0])
    ax2.set_ylim([0, 6.0])
    ax3.set_ylim([0, 6.0])
    ax1.tick_params(direction='in')
    ax2.tick_params(direction='in')
    ax3.tick_params(direction='in')

    plt.show()


run()
