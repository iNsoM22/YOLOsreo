"""
Script for handling KITTI calibration files
"""

import numpy as np


def get_projection_matrix(cam_idx, calib_file):
    """
    Get projection matrix P_rect_0X for camera X (RGB camera)
    and transform to 3 x 4 matrix.
    """
    P_key = f'P{cam_idx}:'
    for line in open(calib_file):
        if P_key in line:
            cam_P = line.strip().split(' ')[1:]  # Skip the key
            cam_P = np.asarray([float(num) for num in cam_P])
            return cam_P.reshape((3, 4))

    file_not_found(calib_file)


def get_P(calib_file):
    """
    Get the projection matrix P_rect_00 for the first camera (RGB camera).
    """
    return get_projection_matrix(0, calib_file)


def get_R0(calib_file):
    """
    Get the rectification rotation matrix R0.
    """
    for line in open(calib_file):
        if 'R0_rect:' in line:
            R0 = line.strip().split(' ')[1:]  # Skip the key
            R0 = np.asarray([float(num) for num in R0])
            return R0.reshape((3, 3))

    file_not_found(calib_file)


def get_Tr_velo_to_cam(calib_file):
    """
    Get the transformation matrix from Velodyne to camera coordinates.
    """
    for line in open(calib_file):
        if 'Tr_velo_to_cam:' in line:
            Tr = line.strip().split(' ')[1:]  # Skip the key
            Tr = np.asarray([float(num) for num in Tr])
            Tr_to_velo = Tr.reshape((3, 4))
            return np.vstack((Tr_to_velo, [0, 0, 0, 1]))  # Add the last row

    file_not_found(calib_file)


def get_Tr_imu_to_velo(calib_file):
    """
    Get the transformation matrix from IMU to Velodyne coordinates.
    """
    for line in open(calib_file):
        if 'Tr_imu_to_velo:' in line:
            Tr = line.strip().split(' ')[1:]  # Skip the key
            Tr = np.asarray([float(num) for num in Tr])
            Tr_to_imu = Tr.reshape((3, 4))
            return np.vstack((Tr_to_imu, [0, 0, 0, 1]))  # Add the last row

    file_not_found(calib_file)


def file_not_found(filename):
    print(f"\nError! Can't read calibration file, does {filename} exist?")
    exit()
