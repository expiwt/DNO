import numpy as np
from scipy import sparse
from nt_toolbox.compute_edge_face_ring import compute_edge_face_ring
import warnings

def compute_boundary(face, input_point=False, boundary_start_point=0):
    """
    compute_boundary - compute the vertices on the boundary of a 3D mesh
    boundary = compute_boundary(face, input_point, boundary_start_point);
    """
    # Compute edges (i,j) that are adjacent to only 1 face
    A = compute_edge_face_ring(face)
    i, j, v = sparse.find(A)
    
    # Retrieve python indices starting from 0
    v[v > 0] = v[v > 0] - 1 
    
    i = i[v == -1]
    j = j[v == -1]

    # Build the boundary by traversing the edges
    if input_point:
        start_idx = np.where(i == boundary_start_point)[0]
        if len(start_idx) > 0:
            start_idx = start_idx[0]
            j = np.hstack((j[start_idx:], j[0:start_idx]))
            i = np.hstack((i[start_idx:], i[0:start_idx]))
        
        boundary = [i[0]] 
        i = i[1:]
        j = j[1:]
    else:
        boundary = [i[0]] 
        i = i[1:]
        j = j[1:]

    while len(i) > 0:
        b = boundary[-1]
        I = np.where(i == b)[0]
        if len(I) == 0:
            I = np.where(j == b)[0]
            if len(I) == 0:
                warnings.warn('Problem with boundary')
                break
            boundary = boundary + [i[I][0]]
        else:
            boundary = boundary + [j[I][0]]

        i = np.delete(i, I)
        j = np.delete(j, I)
    
    return boundary