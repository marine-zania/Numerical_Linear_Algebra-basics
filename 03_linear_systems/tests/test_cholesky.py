import sys
import os
import numpy as np
import pytest

# set sys.path to point to src folder
curr_dir = os.path.dirname(__file__)
sys.path.insert(0, os.path.abspath(os.path.join(curr_dir, '..', 'src')))
from cholesky_decomposition import cholesky, solve_system_cholesky

def test_cholesky_reconstruction():
    # Symmetric Positive-Definite Matrix
    B = np.array([[1, 2, 3], [0, 4, 5], [0, 0, 6]], dtype=float)
    A = np.dot(B.T, B)
    
    L = cholesky(A)
    
    # Check L is lower triangular
    assert np.all(np.triu(L, k=1) == 0)
    
    # Check reconstruction L * L^T = A
    np.testing.assert_allclose(np.dot(L, L.T), A)

def test_cholesky_solver():
    B = np.array([[2, 1], [0, 3]], dtype=float)
    A = np.dot(B.T, B) # SPD Matrix
    b = np.array([5, 8], dtype=float)
    
    x_man = solve_system_cholesky(A, b)
    x_np = np.linalg.solve(A, b)
    
    np.testing.assert_allclose(x_man, x_np, atol=1e-10)

def test_cholesky_non_spd():
    # Matrix that is not SPD (singular or indefinite)
    # 2x2 with zero diagonal
    A_bad = np.array([[0, 1], [1, 0]], dtype=float)
    assert cholesky(A_bad) is None
