import numpy as np
from .substitution import forward_sub, back_sub

def cholesky(A):
    # Perform a Cholesky decomposition of symmetric positive-definite matrix A
    # A = LL^T where L is lower triangular
    n = len(A)
    L = np.zeros_like(A, dtype=np.float64)
    
    for i in range(n):
        for j in range(i + 1):
            sum_val = 0
            for k in range(j):
                # dot product of shared row parts of L
                sum_val += L[i, k] * L[j, k]
            
            if i == j: # diagonal
                val = A[i, i] - sum_val
                if val <= 0:
                    print("Error: Matrix is not positive-definite")
                    return None
                L[i, j] = np.sqrt(val)
            else: # off-diagonal
                L[i, j] = (A[i, j] - sum_val) / L[j, j]
                
    return L

def solve_system_cholesky(A, b):
    # Solve system Ax = b using Cholesky factor
    L = cholesky(A)
    # 1. Ly = b
    y = forward_sub(L, b)
    # 2. L^T x = y
    x = back_sub(L.T, y)
    return x