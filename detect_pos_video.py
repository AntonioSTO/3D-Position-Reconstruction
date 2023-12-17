import cv2
import numpy as np
from cv2 import aruco
import matplotlib.pyplot as plt
import sys



def arucoDetect(file_name):

    parameters =  aruco.DetectorParameters()
    dictionary = aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    arucoDetector = aruco.ArucoDetector(dictionary, parameters)
    vid = cv2.VideoCapture(file_name)

    arucoCoord = []

    while True:
        _, img = vid.read()
        if img is None:
            print("Empty Frame")
            break

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        corners, ids, rejectedImgPoints = arucoDetector.detectMarkers(img)
        frame_markers = aruco.drawDetectedMarkers(img.copy(), corners, ids)
        cv2.imshow('output', frame_markers)
        
        
        
        if ids is not None and 0 in ids:
            idx = np.where(ids == 0)[0][0]
            x,y = np.mean(corners[idx][0], axis=0).astype(int)

            arucoCoord.append(np.array([[x,y,1]]))
        else:
            arucoCoord.append(None)
    
    return arucoCoord
