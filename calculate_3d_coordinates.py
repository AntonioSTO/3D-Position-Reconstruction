import numpy as np

def calculate_3d_coordinates(arucoPos, M_Proj):
    
    #listas de coordenadas 3D
    z3d_coord = []
    y3d_coord = []
    x3d_coord = []

    #len(arucoPos[0]) == qtd de pontos detectados no vídeo da câmera 0
    for frame in range(len(arucoPos[0])):       
        cnt_pnts_add = 0
        for camera_i in range(4):
            if arucoPos[camera_i][frame] is not None:   #se encontrou [U,V,1] -> executa
                #primeira iteração de camera_i
                if cnt_pnts_add == 0:   
                    B = np.append(M_Proj[camera_i], -arucoPos[camera_i][frame].T, axis=1)
                #demais iterações de camera_i
                else:                   
                    B = np.append(B, np.zeros((3*cnt_pnts_add, 1)), axis=1)
                    concat_line = np.concatenate((M_Proj[camera_i], np.zeros((3, cnt_pnts_add)), -arucoPos[camera_i][frame].T), axis=1)
                    B = np.append(B, concat_line, axis=0)
                cnt_pnts_add += 1

        # SVD da matriz B
        _, _, Vt = np.linalg.svd(B)
        # Selecionar os 4 primeiros elementos da última linha de V transposto
        world_point = Vt[-1,:][:4]
        # Passar as coordenadas 3D para coordenadas homogêneas
        world_point = world_point/world_point[3]
        #append das coordenadas 3D às respectivas listas de coordenadas 
        x3d_coord.append(world_point[0])
        y3d_coord.append(world_point[1])
        z3d_coord.append(world_point[2])
    
    return x3d_coord, y3d_coord, z3d_coord