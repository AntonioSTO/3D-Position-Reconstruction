import cv2
import numpy as np
from cv2 import aruco
import cv2 as cv
import matplotlib.pyplot as plt
import sys



def arucoDetect(file_name):

    aruco_dict = aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_50)
    parameters = aruco.DetectorParameters()
    detector = cv.aruco.ArucoDetector(aruco_dict, parameters)
    vid = cv2.VideoCapture(file_name)

    arucoCoord = []

    while True:
        ret, img = vid.read()
        if img is None:
            print("Empty Frame or End of Video")
            break

        corners, ids, rejectedImgPoints = detector.detectMarkers(img)
        frame_markers = aruco.drawDetectedMarkers(img.copy(), corners, ids)
        cv.imshow('output', frame_markers)

        if cv.waitKey(1) == ord('q'):
            break
        
        if ids is not None and 0 in ids:
            idx = np.where(ids == 0)[0][0]
            x,y = np.mean(corners[idx][0], axis=0).astype(int)
            
            

            arucoCoord.append(np.array([[x,y,1]]))
        else:
            arucoCoord.append(None)
    
    return arucoCoord
