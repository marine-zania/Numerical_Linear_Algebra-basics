import numpy as np
from lu_decomposition import LU

def LDV(A):
    # Perform LDV Matrix decomposition: A = LDV
    L, U = LU(A)
    # Handle singular matrices where LU fails
    if L is None:
        return None, None, None
        
    n = len(A)
    
    # Initialize D and V
    D = np.zeros_like(A, dtype=np.float64)
    V = np.zeros_like(A, dtype=np.float64)
    
    for i in range(n):
        # the pivots are on the diagonal of U
        D[i, i] = U[i, i]
        
        # for diagonal matrix D, D^-1 * U is just scaling rows of U by 1/D[i,i]
        # if D[i,i] is zero, a zero pivot error must be handled in LU or here
        V[i, i:] = U[i, i:] / D[i, i]
        
    return L, D, V