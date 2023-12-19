import numpy as np
import cv2 as cv

def arucoDetect(file_name):

    aruco_dict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_50)
    parameters = cv.aruco.DetectorParameters()
    detector = cv.aruco.ArucoDetector(aruco_dict, parameters)
    vid = cv.VideoCapture(file_name)

    arucoCoord = []

    while True:
        ret, img = vid.read()
        if img is None:
            print("Empty Frame or End of Video")
            break

        corners, ids, rejectedImgPoints = detector.detectMarkers(img)
        frame_markers = cv.aruco.drawDetectedMarkers(img.copy(), corners, ids)
        cv.imshow('output', frame_markers)

        if cv.waitKey(1) == ord('q'):
            break
        
        if ids is not None and 0 in ids:
            idx = np.where(ids == 0)[0][0]
            x,y = np.mean(corners[idx][0], axis=0).astype(int)
            arucoCoord.append(np.array([[x,y,1]]))              # m~[i] = [U[i], V[i], 1]^T 
        else:
            arucoCoord.append(None)
    
    return arucoCoord
