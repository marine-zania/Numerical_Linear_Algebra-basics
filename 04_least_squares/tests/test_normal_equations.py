import sys
import os
import numpy as np
import pytest

# set sys.path for local src imports
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(script_dir, '..', 'src'))
import normal_equations

def test_normal_equations_solver():
    # Simple linear regression solve via Normal Equations
    # (A^T A) x = A^T b
    A = np.array([[1, 1], [1, 2], [1, 3]], dtype=float)
    b = np.array([1, 2.1, 2.9], dtype=float)
    
    x_man, res_man = normal_equations.least_squares(A, b)
    
    # Compare with NumPy
    x_np, res_np, _, _ = np.linalg.lstsq(A, b, rcond=None)
    
    # 1. Check Solution
    np.testing.assert_allclose(x_man, x_np, atol=1e-12)
    # 2. Check Residual Norm (NumPy returns sum of squared residuals)
    np.testing.assert_allclose(res_man, np.sqrt(res_np[0]), atol=1e-12)

def test_normal_equations_fitting():
    # Fit y = 2x + 1
    x = np.linspace(0, 10, 50)
    y = 2*x + 1 + np.random.normal(0, 0.1, size=x.shape)
    
    A = np.ones((len(x), 2))
    A[:, 1] = x
    
    x_man, _ = normal_equations.least_squares(A, y)
    
    # Should be close to [1, 2]
    assert np.abs(x_man[0] - 1.0) < 0.5
    assert np.abs(x_man[1] - 2.0) < 0.1
