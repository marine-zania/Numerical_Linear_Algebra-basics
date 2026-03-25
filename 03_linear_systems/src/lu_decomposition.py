import numpy as np
from .substitution import forward_sub, back_sub

def LU(A):
    # Perform LU decomposition: A = LU (no pivoting)
    n = len(A)
    L = np.eye(n, dtype=np.float64)
    U = A.astype(np.float64).copy()
    
    for i in range(n):
        # pivot must be non-zero to continue
        if U[i, i] == 0:
            print("Error: Zero pivot detected")
            return None, None
            
        for k in range(i + 1, n):
            # calculate multiplier for elimination
            multiplier = U[k, i] / U[i, i]
            U[k, i:] = U[k, i:] - multiplier * U[i, i:]
            L[k, i] = multiplier
            
    return L, U

def solve_system_lu(A, b):
    # Solve linear system Ax = b using LU decomposition
    L, U = LU(A)
    # Ly = b
    y = forward_sub(L, b)
    # Ux = y
    x = back_sub(U, y)
    return x