import numpy as np
from cholesky_decomposition import cholesky

def LDLT(A):
    # Perform LDL^T decomposition (building on top of Cholesky)
    # A = LDL^T where L is unit lower triangular and D is diagonal
    # This decomposition only works if the Cholesky factor exists (SPD matrix)
    C = cholesky(A)
    if C is None:
        return None, None
    n = len(A)
    
    # Standard derivation: A = CC^T = (L*sqrt(D)) (L*sqrt(D))^T = LDL^T
    # So sqrt(D)_ii = C_ii and L_ij = C_ij / C_jj
    D = np.zeros_like(A, dtype=np.float64)
    L = np.zeros_like(A, dtype=np.float64)
    
    for i in range(n):
        # the squared diagonal elements of Cholesky matrix C are our D values
        D[i, i] = C[i, i]**2
        
        # for diagonal matrix D_sqrt, normalizing L_unit is just dividing C's rows by D_sqrt
        L[:, i] = C[:, i] / C[i, i]
        
    return L, D