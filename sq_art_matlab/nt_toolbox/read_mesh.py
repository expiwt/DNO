import numpy as np

def read_mesh(name):
    '''
    Read from OFF file in 3 dimensions
    Returns X0 (coordinates) and F (faces)
    '''
    with open(name, mode="r") as file:
        # Check type of file
        file_type = file.readline().strip()
        if file_type != "OFF":
            raise Exception("Wrong type of file, only reads OFF files")
    
        # Number of vertices/faces/edges
        n_verts, n_faces, n_edges = tuple([int(s) for s in file.readline().strip().split(' ')])
        
        # Vertices
        X0 = []
        for _ in range(n_verts):
            X0.append(file.readline().strip().split(' '))
            
        # Faces
        F = []
        for i in range(n_faces):
            F.append(file.readline().strip().split(' ')[1:])
    
    return np.transpose(np.asarray(X0).astype(float)), np.transpose(np.asarray(F).astype(int))