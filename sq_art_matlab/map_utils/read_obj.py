import numpy as np

def read_obj(filename):
    """
    read_obj - load a .obj file.
    [vertex,face] = read_obj(filename);
    faces: list of triangle elements
    vertex: node coordinates
    """
    faces_s = []
    vertex_s = []
    
    with open(filename, "r") as file:
        for line in file.readlines():
            if line[0] == 'v':
                line = line.replace("\n", "").replace('v', "")
                vertex_s.append(line[1:].split())
            elif line[0] == 'f':
                line = line.replace('f', "").replace("\n", "")
                faces_s.append(line[1:].split())
    
    # Convert to arrays
    vertex = []
    for i in range(len(vertex_s)):
        vertex.append(list(map(lambda x: round(float(x), 6), vertex_s[i])))
    
    faces = []
    for i in range(len(faces_s)):
        if (faces_s[i][0] != faces_s[i][1]) and (faces_s[i][1] != faces_s[i][2]) and (faces_s[i][0] != faces_s[i][2]):
            # Take only vertex indices (ignore texture/normal)
            face_vertices = [int(part.split('/')[0]) for part in faces_s[i]]
            faces.append(face_vertices)
    
    vertex = np.array(vertex).T
    faces = np.array(faces).T
    
    return vertex, faces