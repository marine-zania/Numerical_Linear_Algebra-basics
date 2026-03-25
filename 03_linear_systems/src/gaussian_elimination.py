import numpy as np
from .substitution import back_sub

def forward_elimination(A, b):
    # n is matrix dimension
    n = len(b)
    
    # Work on copies to avoid modifying original inputs
    U = A.astype(float).copy()
    c = b.astype(float).copy()
    
    for i in range(n):
        # check for zero on the diagonal (pivoting)
        if U[i, i] == 0:
            raise ValueError(f"Zero pivot at ({i}, {i}). Gaussian elimination failed.")
            
        for j in range(i + 1, n):
            # calculate the ratio for elimination
            ratio = U[j, i] / U[i, i]
            
            # Update the rest of the row
            U[j, i:] = U[j, i:] - ratio * U[i, i:]
            
            # Update the constant vector c
            c[j] = c[j] - ratio * c[i]
            
    return U, c

def solve_gaussian_elimination(A, b):
    """
    Solves Ax = b using Gaussian elimination (without pivoting).
    """
    # 1. Forward Elimination (Convert to Upper Triangular form Ux = c)
    U_tri, c_new = forward_elimination(A, b)
    
    # 2. Back Substitution (Solve the triangular system)
    x_sol = back_sub(U_tri, c_new)
    
    return x_sol
