import sys
import os
import numpy as np
import pytest

# set sys.path for local src imports
curr_dir = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(curr_dir, '..')))
from src.ldv_decomposition import LDV
from src.ldlt_decomposition import LDLT

def test_ldv_reconstruction():
    # Test reconstruction A = LDV
    A = np.array([[2, 1, 1], [4, 3, 3], [8, 7, 9]], dtype=float)
    L, D, V = LDV(A)
    
    # Check dimensions
    assert L.shape == A.shape
    assert D.shape == A.shape
    assert V.shape == A.shape
    
    # Check L is unit lower triangular
    np.testing.assert_allclose(np.diag(L), 1.0)
    assert np.all(np.triu(L, k=1) == 0)
    
    # Check V is unit upper triangular
    np.testing.assert_allclose(np.diag(V), 1.0)
    assert np.all(np.tril(V, k=-1) == 0)
    
    # Check D is diagonal
    assert np.all(D - np.diag(np.diag(D)) == 0)
    
    # Check reconstruction L * D * V = A
    reconstructed = np.dot(np.dot(L, D), V)
    np.testing.assert_allclose(reconstructed, A)

def test_ldlt_reconstruction():
    # Symmetric Positive-Definite Matrix
    B = np.array([[2, 1], [0, 4]], dtype=float)
    A = np.dot(B.T, B) # Symmetric and SPD
    
    L, D = LDLT(A)
    
    # Check L is unit lower triangular
    np.testing.assert_allclose(np.diag(L), 1.0)
    assert np.all(np.triu(L, k=1) == 0)
    
    # Check D is diagonal
    assert np.all(D - np.diag(np.diag(D)) == 0)
    
    # Check reconstruction L * D * L^T = A
    reconstructed = np.dot(np.dot(L, D), L.T)
    np.testing.assert_allclose(reconstructed, A)

def test_ldv_singular():
    A_sing = np.array([[1, 2], [2, 4]], dtype=float)
    L, D, V = LDV(A_sing)
    # the LU decomposition returns None if singular
    assert L is None and D is None and V is None
