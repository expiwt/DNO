import numpy as np

def cutting_line(X, F, line="long_1"):
    """
    Cutting line function for mesh processing
    """
    tmp = []
    I = []
    new_I = []
    X = X.T
    F_initial = F.copy()
    
    z_min = X[:, 2].min()
    z_max = X[:, 2].max()
    x_min = X[:, 0].min()
    x_max = X[:, 0].max()
    y_min = X[:, 1].min()
    y_max = X[:, 1].max()
    
    new_x = X
    z_mid = (z_max + z_min) / 2
    
    idx_open_x = []
    idx_close_x = []
    coundry_start_point = []
    
    if line == "short":
        # Short edge cutting
        for i, x in enumerate(X):
            if (x == [x_max, y_min, z_mid]).all():
                idx_open_x.append(i)
            if (x == [x_max, y_max, z_mid]).all():
                idx_close_x.append(i)
            if (x == [x_min, y_min, z_mid]).all():
                coundry_start_point.append(i)
        
        idx_open_x = idx_open_x[0]
        idx_close_x = idx_close_x[0]
        
        idx_mid = X.shape[0]
        idx_mid_list = []
        
        for i, x in enumerate(X):
            if x[2] == z_mid:
                if x[0] == x_max and (x[1] > y_min and x[1] < y_max):
                    continue
                else:
                    tmp.append(x)
                    I.append(i)
                    idx_mid_list.append(idx_mid)
                    idx_mid += 1
                    
    elif line == "long_0":
        # Long edge cutting
        for i, x in enumerate(X):
            if (x == [x_min, y_min, z_mid]).all():
                idx_open_x.append(i)
            if (x == [x_max, y_min, z_mid]).all():
                idx_close_x.append(i)
            if (x == [x_min, y_max, z_mid]).all():
                coundry_start_point.append(i)
        
        idx_open_x = idx_open_x[0]
        idx_close_x = idx_close_x[0]
        
        idx_mid = X.shape[0]
        idx_mid_list = []
        
        for i, x in enumerate(X):
            if x[2] == z_mid:
                if x[1] == y_min and (x[0] > x_min and x[0] < x_max):
                    continue
                else:
                    tmp.append(x)
                    I.append(i)
                    idx_mid_list.append(idx_mid)
                    idx_mid += 1
                    
    elif line == "long_1":
        # Long edge cutting (alternative)
        for i, x in enumerate(X):
            if (x == [x_min, y_max, z_mid]).all():
                idx_open_x.append(i)
            if (x == [x_max, y_max, z_mid]).all():
                idx_close_x.append(i)
            if (x == [x_min, y_min, z_mid]).all():
                coundry_start_point.append(i)
        
        idx_open_x = idx_open_x[0]
        idx_close_x = idx_close_x[0]
        
        idx_mid = X.shape[0]
        idx_mid_list = []
        
        for i, x in enumerate(X):
            if x[2] == z_mid:
                if x[1] == y_max and (x[0] > x_min and x[0] < x_max):
                    continue
                else:
                    tmp.append(x)
                    I.append(i)
                    idx_mid_list.append(idx_mid)
                    idx_mid += 1
    
    # Add new points to the mesh
    new_x = np.append(X, np.array(tmp), axis=0)
    cutting_boundry = []
    
    start_point = idx_open_x
    stop_point = idx_close_x
    new_I = I.copy()
    
    if idx_close_x in new_I:
        new_I.remove(idx_close_x)
    if idx_open_x in new_I:
        new_I.remove(idx_open_x)
        
    cutting_boundry.append(idx_open_x)
    F = F.T
    cutting_faces = []
    faces_number = []
    
    # Find cutting boundary points and faces
    for i in range(len(new_I)):
        for num, j in enumerate(F):
            if j[0] == start_point or j[1] == start_point or j[2] == start_point:
                if j[0] in new_I:            
                    start_tmp = j[0]    
                    cutting_faces.append(j)  
                    faces_number.append(num)
                elif j[1] in new_I:     
                    start_tmp = j[1]
                    cutting_faces.append(j)
                    faces_number.append(num)
                elif j[2] in new_I:
                    start_tmp = j[2]  
                    cutting_faces.append(j)
                    faces_number.append(num)
        
        if i == 0 and idx_close_x not in new_I:
            new_I.append(idx_close_x)
            
        if start_point != start_tmp:   
            cutting_boundry.append(int(start_tmp))
        stop_point = start_point
        start_point = start_tmp
        
        if start_tmp in new_I:
            new_I.remove(start_tmp)
    
    cutting_faces = np.array(cutting_faces)        
    I_tmp_list = cutting_boundry.copy()
    
    if idx_open_x in I_tmp_list:
        I_tmp_list.remove(idx_open_x)
    if idx_close_x in I_tmp_list:
        I_tmp_list.remove(idx_close_x)
    
    # Update faces with new points
    for i in faces_number: 
        for j in F[i]:
            if j not in cutting_boundry:           
                if new_x[int(j)][2] <= z_mid:
                    for m, f in enumerate(F[i]):
                        if f in I_tmp_list:
                            F[i][m] = idx_mid_list[np.where(np.array(I) == f)[0][0]]   
    
    # Update all faces below cutting line
    for i, f in enumerate(F):
        f = f.astype('int64')
        if f[0] in I_tmp_list:
            if new_x[f[1]][2] <= z_mid and new_x[f[2]][2] <= z_mid:
                F[i][0] = idx_mid_list[np.where(np.array(I) == f[0])[0][0]]
        elif f[1] in I_tmp_list:
            if new_x[f[0]][2] <= z_mid and new_x[f[2]][2] <= z_mid:
                F[i][1] = idx_mid_list[np.where(np.array(I) == f[1])[0][0]]
        elif f[2] in I_tmp_list:
            if new_x[f[1]][2] <= z_mid and new_x[f[0]][2] <= z_mid:
                F[i][2] = idx_mid_list[np.where(np.array(I) == f[2])[0][0]]
    
    return new_x.T, F.T, coundry_start_point