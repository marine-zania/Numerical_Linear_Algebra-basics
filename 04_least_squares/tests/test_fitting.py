import sys
import os
import numpy as np
import pytest

# set sys.path for local src imports
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(script_dir, '..', 'src'))
from fitting import linear_model, polynomial_model, exponential_model

def test_linear_model():
    x = np.array([1, 2, 3])
    # Model: y = a0 + a1*x
    A = linear_model(x)
    A_expected = np.array([
        [1, 1],
        [1, 2],
        [1, 3]
    ], dtype=float)
    np.testing.assert_allclose(A, A_expected)

def test_polynomial_model():
    x = np.array([1, 2, 3])
    # Order 2: y = a0 + a1*x + a2*x^2
    A = polynomial_model(x, order=2)
    A_expected = np.array([
        [1, 1, 1],
        [1, 2, 4],
        [1, 3, 9]
    ], dtype=float)
    np.testing.assert_allclose(A, A_expected)

def test_exponential_model():
    x = np.array([1, 2, 3])
    # The exponential model constructor in fitting.py is just a linear model
    # Because we fit in log-space: ln(y) = ln(a) + b*x.
    A = exponential_model(x)
    A_expected = np.array([
        [1, 1],
        [1, 2],
        [1, 3]
    ], dtype=float)
    np.testing.assert_allclose(A, A_expected)
