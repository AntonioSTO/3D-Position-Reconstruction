import numpy as np

def getProjection(K, R, T):
    pi = np.array([[1,0,0,0],
                   [0,1,0,0],
                   [0,0,1,0]])
    
    extM = np.vstack((np.hstack((R, T)), np.array([[0, 0, 0, 1]])))     #matriz de parâmetros extrínsecos

    Projection = K @ pi @ np.linalg.inv(extM)       #P[i] = K[i] . pi . [R[i], T[i]]^-1
    
    return Projection