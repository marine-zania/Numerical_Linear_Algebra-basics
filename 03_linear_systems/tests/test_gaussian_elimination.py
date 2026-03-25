import sys
import os
import numpy as np
import pytest

# set sys.path to point to src folder
curr_dir = os.path.dirname(__file__)
sys.path.insert(0, os.path.abspath(os.path.join(curr_dir, '..', 'src')))
from gaussian_elimination import forward_elimination, solve_gaussian_elimination

def test_mysolver_small():
    A = np.array([[2, 1], [1, 3]], dtype=float)
    b = np.array([5, 5], dtype=float)
    x_expected = np.array([2, 1], dtype=float)
    
    x_man = solve_gaussian_elimination(A, b)
    
    # checking manual result against expected
    np.testing.assert_allclose(x_man, x_expected, atol=1e-12)

def test_random_systems():
    # checking random matrices of different sizes
    np.random.seed(42)
    for sz in [3, 5, 10]:
        A_rand = np.random.rand(sz, sz) + np.eye(sz) * 5
        b_rand = np.random.rand(sz)
        
        x_man = solve_gaussian_elimination(A_rand, b_rand)
        x_np = np.linalg.solve(A_rand, b_rand)
        
        # compare manual against numpy.linalg.solve result
        np.testing.assert_allclose(x_man, x_np, atol=1e-10)

def test_singular_error():
    # checking for error handling on singular matrices
    A_sing = np.array([[1, 2], [2, 4]], dtype=float)
    b_vec = np.array([3, 6], dtype=float)
    
    with pytest.raises(ValueError, match="Zero pivot"):
        forward_elimination(A_sing, b_vec)
