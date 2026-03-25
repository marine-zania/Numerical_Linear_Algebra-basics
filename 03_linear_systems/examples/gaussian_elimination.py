# Gaussian Elimination Example

import numpy as np
import sys
import os

# setup sys.path for local src imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.gaussian_elimination import forward_elimination, solve_gaussian_elimination

def run_example():
    np.random.seed(42)
    sz = 4
    A_mat = np.random.randint(1, 10, (sz, sz)).astype(float)
    b_vec = np.random.randint(1, 10, (sz, 1)).astype(float)
    
    print("--- 1. Input System [A|b] ---")
    print(f"Matrix A:\n{A_mat}\nVector b:\n{b_vec}\n")

    # 1. Step-by-step check
    print("--- 2. Step-by-Step Solving ---")
    U_tri, c_new = forward_elimination(A_mat, b_vec)
    print(f"Upper Triangular Matrix U:\n{U_tri}")
    print(f"Modified Vector c:\n{c_new}")
    
    # 2. Main solver shortcut
    x_man = solve_gaussian_elimination(A_mat, b_vec)
    print(f"\nSolution x (Manual):\n{x_man}\n")

    # 3. Verification with NumPy
    print("--- 3. Verification ---")
    x_np = np.linalg.solve(A_mat, b_vec)
    print(f"NumPy Solution x:\n{x_np}")
    
    # check error norm
    diff = np.linalg.norm(x_man - x_np)
    print(f"\nDifference (L2 Norm): {diff:.2e}")

if __name__ == "__main__":
    run_example()
