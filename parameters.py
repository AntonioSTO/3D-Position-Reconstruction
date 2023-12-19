import numpy as np
import json

# Função de leitura dos parametros intrínsecos e extrínsecos de cada camera
 
def camera_parameters(file):
    camera_data = json.load(open(file))
    K = np.array(camera_data['intrinsic']['doubles']).reshape(3, 3)   #matriz de parametros intrínsecos
    res = [camera_data['resolution']['width'],
           camera_data['resolution']['height']]                       #resolução de cada câmera
    tf = np.array(camera_data['extrinsic']['tf']['doubles']).reshape(4, 4)
    R = tf[:3, :3]
    T = tf[:3, 3].reshape(3, 1)
    dis = np.array(camera_data['distortion']['doubles'])
    return K, R, T, res, dis





