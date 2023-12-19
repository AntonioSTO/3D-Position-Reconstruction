import numpy as np
import matplotlib.pyplot as plt
from projection_M_builder import getProjection
from parameters import camera_parameters
from detect_pos_video import arucoDetect
from calculate_3d_coordinates import calculate_3d_coordinates

def main():
    np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})

    K = []
    R = []
    T = []

    # Carrega os parâmetros de cada câmera
    for idx in range(4):
        k, r, t, _, _ = camera_parameters(f'data/params/{idx}.json')
        K.append(k)
        R.append(r)
        T.append(t)
    # K0, R0, T0, _, _ = camera_parameters('data/params/0.json')
    # K1, R1, T1, _, _ = camera_parameters('data/params/1.json')
    # ...


    #mProj = [P1, P2, P3, P4]  -> matrizes de projeção das 4 câmeras
    mProj = []
    for idx in range(4):
        mProj.append(getProjection(K[idx], R[idx], T[idx]))
    # mProj.append(getProjection(K0, R0, T0))
    # mProj.append(getProjection(K1, R1, T1))
    # ...


    #arucoPos = [pontos_cam0, pontos_cam1, pontos_cam2, pontos_cam3]
    arucoPos = []
    for idx in range(4):
        arucoPos.append(arucoDetect(f"data/videos/camera-0{idx}.mp4"))
    #arucoPos.append(arucoDetect("data/videos/camera-00.mp4"))
    #arucoPos.append(arucoDetect("data/videos/camera-01.mp4"))
    #...


    # Cálculo das coordenadas 3d (pontos no mundo) com as matrizes de projeção e
    #as posições detectadas nas câmeras pelos marcadores Aruco
    x3d, y3d, z3d = calculate_3d_coordinates(arucoPos, mProj)

    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x3d, y3d, z3d)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.axes.set_zlim3d(bottom=-1, top=1)
    plt.show()

if __name__ == "__main__":
    main()