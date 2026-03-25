import numpy as np
import sys
import os

# Add Module 03 to path to enable importing its solvers
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '03_linear_systems', 'src')))

try:
    from cholesky_decomposition import solve_system_cholesky
except ImportError:
    print("Error: Could not find 03_linear_systems solvers. Make sure path is correct.")

def least_squares(A, b):
    """
    Solve Least Squares problem Ax = b using Normal Equations:
    (A^T A) x = A^T b
    
    Returns:
        x (np.array): The least squares solution.
        residual (float): The norm of the error vector (b - Ax).
    """
    # 1. Construct the Normal Equations system
    # A_T_A x = A_T_b
    A_T_A = np.dot(A.T, A)
    A_T_b = np.dot(A.T, b)
    
    # 2. Solve the symmetric positive definite system using Cholesky
    # This is standard for Normal Equations solvers
    x = solve_system_cholesky(A_T_A, A_T_b)
    
    if x is None:
        raise ValueError("Matrix A is likely rank-deficient (Normal Equations failed).")
    
    # 3. Calculate residual norm ||b - Ax||
    # This represents 'how well' the model fits the data
    residual = np.linalg.norm(b - np.dot(A, x))
    
    return x, residual