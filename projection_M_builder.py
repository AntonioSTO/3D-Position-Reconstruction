import numpy as np

def getProjection(K, R, T):
    pi = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0]])
    extM = np.vstack((np.hstack((R, T)), np.array([[0, 0, 0, 1]])))

    Projection = K @ pi @ np.linalg.inv(extM)
    
    return Projection