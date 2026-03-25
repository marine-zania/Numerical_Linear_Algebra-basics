import sys
import os
import numpy as np
import pytest

# set sys.path for local src imports
curr_dir = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(curr_dir, '..')))
from src.lu_decomposition import LU, solve_system_lu

def test_lu_reconstruction():
    # Test reconstruction A = LU
    A = np.array([[2, 1, 1], [4, 3, 3], [8, 7, 9]], dtype=float)
    L, U = LU(A)
    
    # Check L is unit lower triangular
    np.testing.assert_allclose(np.diag(L), 1.0)
    assert np.all(np.triu(L, k=1) == 0)
    
    # Check U is upper triangular
    assert np.all(np.tril(U, k=-1) == 0)
    
    # Check reconstruction
    np.testing.assert_allclose(np.dot(L, U), A)

def test_lu_solver():
    A = np.array([[2, 1, 1], [4, 3, 3], [8, 7, 9]], dtype=float)
    b = np.array([4, 10, 24], dtype=float)
    
    x_man = solve_system_lu(A, b)
    x_np = np.linalg.solve(A, b)
    
    np.testing.assert_allclose(x_man, x_np, atol=1e-10)

def test_lu_singular():
    A_sing = np.array([[1, 2], [2, 4]], dtype=float)
    # The current LU returns None, None for zero pivot
    L, U = LU(A_sing)
    assert L is None and U is None
