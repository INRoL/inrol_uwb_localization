'''
The following code is adapted from: https://github.com/uzh-rpg/rpg_vision-based_slam/blob/main/scripts/python/rpg_vision_based_slam/utils.py
Origianl author: Giovanni Cioffi <cioffi at ifi dot uzh dot ch>
'''

import os
import sys
import numpy as np
import yaml


def writeTrajEstToTxt(out_file, timestamps, pose_list):
    assert len(timestamps) == len(pose_list)

    f = open(out_file, "w")
    f.write("# timestamp tx ty tz qx qy qz qw\n")
    for i in range(len(timestamps)):
        ts = timestamps[i]
        t = pose_list[i].t
        q = pose_list[i].q_wxyz()
        f.write(
            "{:.12f} {:.12f} {:.12f} {:.12f} {:.12f} {:.12f} {:.12f} {:.12f}\n".format(
                ts, t[0][0], t[1][0], t[2][0], q[1], q[2], q[3], q[0]
            )
        )
    f.close()

    print("Written {} estimates to {}.".format(len(timestamps), out_file))


def writeUWBOffsets(out_file, offsets):
    f = open(out_file, "w")
    f.write("# anchor_id offset\n")
    for i in range(1, 9):
        o = offsets[i - 1]
        f.write("{:d} {:.12f}\n".format(i, o))
    f.close()

    print("Written {} offsets to {}.".format(len(offsets), out_file))


def readImageIdAndTimestamps(in_file):
    f_img_infos = open(in_file, "r")
    lines = f_img_infos.readlines()
    ids = []
    timestamps = []
    for cnt, line in enumerate(lines):
        if cnt > 0:
            ids.append(int(line.split(" ")[0]))
            timestamps.append(float(line.split(" ")[1]))
    f_img_infos.close()
    return ids, timestamps


def readImageIdTimestampAndName(in_file):
    f_img_infos = open(in_file, "r")
    lines = f_img_infos.readlines()
    ids = []
    timestamps = []
    names = []
    for cnt, line in enumerate(lines):
        if cnt > 0:
            ids.append(int(line.split(" ")[0]))
            timestamps.append(float(line.split(" ")[1]))
            names.append(line.split(" ")[2][4:])
    f_img_infos.close()
    return ids, timestamps, names


def readTimestampAndPose(in_file):
    f_img_infos = open(in_file, "r")
    lines = f_img_infos.readlines()
    timestamps = []
    t = []
    q = []
    for cnt, line in enumerate(lines):
        if cnt > 0:
            timestamps.append(float(line.split(" ")[0]))
            tx = float(line.split(" ")[1])
            ty = float(line.split(" ")[2])
            tz = float(line.split(" ")[3])
            qx = float(line.split(" ")[4])
            qy = float(line.split(" ")[5])
            qz = float(line.split(" ")[6])
            qw = float(line.split(" ")[7])
            t.append([tx, ty, tz])
            q.append([qx, qy, qz, qw])

    return timestamps, t, q


def readAnchorPosition(in_file):
    f_img_infos = open(in_file, "r")
    lines = f_img_infos.readlines()
    idx = []
    x = []
    y = []
    z = []
    for cnt, line in enumerate(lines):
        if cnt > 0:
            idx.append(float(line.split(" ")[0]))
            x.append(float(line.split(" ")[1]))
            y.append(float(line.split(" ")[2]))
            z.append(float(line.split(" ")[3]))

    return idx, x, y, z


def readAnchorPoses(in_file):
    f_img_infos = open(in_file, "r")
    lines = f_img_infos.readlines()
    idx = []
    x = []
    y = []
    z = []
    qx = []
    qy = []
    qz = []
    qw = []
    for cnt, line in enumerate(lines):
        if cnt > 0:
            idx.append(float(line.split(" ")[0]))
            x.append(float(line.split(" ")[1]))
            y.append(float(line.split(" ")[2]))
            z.append(float(line.split(" ")[3]))
            qx.append(float(line.split(" ")[4]))
            qy.append(float(line.split(" ")[5]))
            qz.append(float(line.split(" ")[6]))
            qw.append(float(line.split(" ")[7]))

    return idx, x, y, z, qx, qy, qz, qw


def readIMUData(in_file):
    f_img_infos = open(in_file, "r")
    lines = f_img_infos.readlines()
    ts = []
    a = []
    w = []
    for cnt, line in enumerate(lines):
        if cnt > 0:
            ts.append(float(line.split(" ")[1]))
            wx = float(line.split(" ")[2])
            wy = float(line.split(" ")[3])
            wz = float(line.split(" ")[4])
            ax = float(line.split(" ")[5])
            ay = float(line.split(" ")[6])
            az = float(line.split(" ")[7])
            a.append([ax, ay, az])
            w.append([wx, wy, wz])

    return ts, a, w


def readUWBData(in_file):
    f_img_infos = open(in_file, "r")
    lines = f_img_infos.readlines()
    ts = []
    id = []
    r = []
    for cnt, line in enumerate(lines):
        if cnt > 0:
            ts.append(float(line.split(" ")[1]))
            id.append(int(line.split(" ")[2]))
            r.append(float(line.split(" ")[3]))

    return ts, id, r


def readUWBErrorData(in_file):
    f_img_infos = open(in_file, "r")
    lines = f_img_infos.readlines()
    ts = []
    id = []
    r = []
    for cnt, line in enumerate(lines):
        if cnt > 0:
            ts.append(float(line.split(" ")[0]))
            id.append(int(line.split(" ")[1]))
            r.append(float(line.split(" ")[2]))

    return ts, id, r


def readSphHarmCoeffs(in_file):
    f_img_infos = open(in_file, "r")
    lines = f_img_infos.readlines()
    coeffs = []
    for cnt, line in enumerate(lines):
        if cnt > 0:
            coeffs.append(float(line.split(" ")[1]))

    return coeffs


def readOffsets(in_file):
    f_img_infos = open(in_file, "r")
    lines = f_img_infos.readlines()
    offsets = {}
    for cnt, line in enumerate(lines):
        if cnt > 0:
            offsets[float(line.split(" ")[0])] = float(line.split(" ")[1])

    return offsets


def readModelParam(in_file, type):
    f_img_infos = open(in_file, "r")
    lines = f_img_infos.readlines()
    param_list = lines[1].split(" ")
    if (type == "A"):
        return float(param_list[0]), None
    elif (type == "B"):
        return float(param_list[0]), None
    elif (type == "C"):
        return float(param_list[0]), float(param_list[1])
    else:
        return -1
