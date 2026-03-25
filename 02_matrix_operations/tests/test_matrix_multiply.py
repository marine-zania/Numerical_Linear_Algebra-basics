"""Verifying matrix multiplication operations against NumPy."""
import numpy as np
import sys
import os

# point directly to src for isolated imports
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(script_dir, '..', 'src'))
import matrix_multiply

def test_mat_vec_multiply():
    A = [[1.0, 2.0], [3.0, 4.0]]
    x = [5.0, 6.0]
    ours = matrix_multiply.mat_vec_multiply(A, x)
    theirs = np.dot(np.array(A), np.array(x)).tolist()
    
    print(f"  Mat-Vec Multiply: {A} * {x}")
    print(f"    - Ours:  {ours}")
    print(f"    - NumPy: {theirs}")
    assert np.allclose(ours, theirs)
    print("    ✓ VERIFIED MATCH\n")

def test_mat_mat_multiply():
    A = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    B = [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]
    ours = matrix_multiply.mat_mat_multiply(A, B)
    theirs = np.dot(np.array(A), np.array(B)).tolist()
    
    print(f"  Mat-Mat Multiply: A(2x3) * B(3x2)")
    print(f"    - Ours:  {ours}")
    print(f"    - NumPy: {theirs}")
    assert np.allclose(ours, theirs)
    print("    ✓ VERIFIED MATCH\n")

if __name__ == "__main__":
    print("=" * 60)
    print("VERIFICATION: UNIT 02 — MATRIX MULTIPLICATION")
    print("=" * 60)
    test_mat_vec_multiply()
    test_mat_mat_multiply()
    print("ALL MATRIX MULTIPLICATION VERIFIED ✓")
