import numpy as np
from scipy import sparse

def compute_edge_face_ring(faces):
    """
    compute_edge_face_ring - compute faces adjacent to each edge
    e2f = compute_edge_face_ring(faces);
    e2f(i,j) and e2f(j,i) are the number of the two faces adjacent to edge (i,j)
    """
    n = np.max(faces) + 1
    m = np.shape(faces)[1]
    
    i = np.hstack((faces[0, :], faces[1, :], faces[2, :]))
    j = np.hstack((faces[1, :], faces[2, :], faces[0, :]))
    s = np.hstack((np.arange(0, m), np.arange(0, m), np.arange(0, m)))

    # First without duplicates
    tmp, I = np.unique(i + (np.max(i) + 1) * j, return_index=True)
    
    # Remaining items
    J = np.setdiff1d(np.arange(0, len(s)), I)
    
    # Flip the duplicates
    i1 = np.hstack((i[I], j[J]))
    j1 = np.hstack((j[I], i[J]))
    s = np.hstack((s[I], s[J]))

    # Remove duplicates
    tmp, I = np.unique(i1 + (np.max(i1) + 1) * j1, return_index=True)
    i1 = i1[I]
    j1 = j1[I]
    s = s[I]
    
    nn = n.astype(np.int32)
    i11 = i1.astype(np.int32)
    j11 = j1.astype(np.int32)
    
    A = sparse.coo_matrix((s + 1, (i11, j11)), shape=(nn, nn))
    
    # Add missing points
    B = A.toarray()
    I = np.where(np.ravel(B.T) != 0)[0]
    J = np.where(np.ravel(B)[I] == 0)[0]
    np.ravel(B)[I[J]] = -1
    
    return sparse.coo_matrix(B)