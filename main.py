import numpy as np
import matplotlib.pyplot as plt
import math
import types
# from types import NoneType
import json
import cv2 as cv
from cv2 import aruco
from projection_M_builder import getProjection
from parameters import camera_parameters
from detect_pos_video import arucoDetect

if __name__ == "__main__":

    np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})

    # Load cameras parameters
    K0, R0, T0, res0, dis0 = camera_parameters('data/params/0.json')
    K1, R1, T1, res1, dis1 = camera_parameters('data/params/1.json')
    K2, R2, T2, res2, dis2 = camera_parameters('data/params/2.json')
    K3, R3, T3, res3, dis3 = camera_parameters('data/params/3.json')

    mProj = []
    mProj.append(getProjection(K0, R0, T0))
    mProj.append(getProjection(K1, R1, T1))
    mProj.append(getProjection(K2, R2, T2))
    mProj.append(getProjection(K3, R3, T3))

    arucoPos = []
    arucoPos.append(arucoDetect("data/videos/camera-00.mp4"))
    arucoPos.append(arucoDetect("data/videos/camera-01.mp4"))
    arucoPos.append(arucoDetect("data/videos/camera-02.mp4"))
    arucoPos.append(arucoDetect("data/videos/camera-03.mp4"))

    frame_count = len(arucoPos[0])

    x3d = []
    y3d = []
    z3d = []
    for frame in range(frame_count):
        added_points = 0
        for cam in range(4):
            if arucoPos[cam][frame] is not None:
                if added_points == 0:
                    B_matrix = np.append(mProj[cam], -1 * arucoPos[cam][frame].T, axis = 1)

                else:
                    B_matrix = np.append(B_matrix, np.zeros((3 * added_points, 1)), axis = 1)
                    new_lines = np.concatenate((mProj[cam], np.zeros((3, added_points)), -1 * arucoPos[cam][frame].T), axis = 1)
                    B_matrix = np.append(B_matrix, new_lines, axis = 0)

                added_points = added_points + 1

        # Perform SVD(A) = U.S.Vt to estimate the homography
        U, S, Vt = np.linalg.svd(B_matrix)

        # Reshape last column of V as the 3 dimension point
        V_last_column = Vt[len(Vt)-1]
        point_3D = V_last_column[:4]
        point_3D = point_3D/point_3D[3]

        x3d.append(point_3D[0])
        y3d.append(point_3D[1])
        z3d.append(point_3D[2])
        
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x3d, y3d, z3d)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()