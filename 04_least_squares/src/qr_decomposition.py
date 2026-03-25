import numpy as np
import sys
import os

# Add Module 03 to path to enable importing for solving triangular systems
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '03_linear_systems', 'src')))
from substitution import back_sub

def qr_factor_mgs(A):
    """
    Factor A = QR using Modified Gram-Schmidt.
    Stable and straightforward for educational purposes.
    """
    m, n = A.shape
    Q = np.zeros((m, n), dtype=np.float64)
    R = np.zeros((n, n), dtype=np.float64)
    
    V = A.astype(np.float64).copy()
    
    for i in range(n):
        # 1. calculate the norm of the ith column
        R[i, i] = np.linalg.norm(V[:, i])
        
        # 2. normalize the column to get q_i
        if R[i, i] == 0:
            print("Warning: Rank deficient matrix detected")
            continue
        Q[:, i] = V[:, i] / R[i, i]
        
        # 3. orthogonalize the remaining columns against q_i
        for j in range(i + 1, n):
            R[i, j] = np.dot(Q[:, i], V[:, j])
            V[:, j] = V[:, j] - R[i, j] * Q[:, i]
            
    return Q, R

def solve_least_squares_qr(A, b):
    """
    Solve Ax = b using QR: Rx = Q.T * b
    Better numerical stability than Normal Equations.
    """
    # 1. Decompose A = QR
    Q, R = qr_factor_mgs(A)
    
    # 2. Compute Q^T * b
    c = np.dot(Q.T, b)
    
    # 3. Solve the upper triangular system Rx = c
    x = back_sub(R, c)
    
    # Residual norm for fit quality
    residual = np.linalg.norm(b - np.dot(A, x))
    
    return x, residual
